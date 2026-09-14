#!/usr/bin/env python3
"""WFV HYBRYD boost + drzewa (2026-09-14, decyzja uzytkownika: "raczej hybrydy drzew i boostow").

Po ensemble_szukaj.py (jedno ciecie, wybor na polowie testu) hybrydy boost+drzewa na duzym shorcie wyszly
na plus w 2 przebiegach, ale z inna konfiguracja za kazdym razem -> tu test walk-forward:
- 12 okien po 45 dni; w KAZDYM oknie 6 modeli (LGB, XGB, CAT, ET, RF, HGB) uczonych od nowa WYLACZNIE na danych
  sprzed (poczatek okna - 7 dni), kazdy sam dobiera --ile cech bez dubli (|rho|<=0.5) — jak ensemble_szukaj;
- prognozy kazdego modelu -> PRZYCZYNOWY percentyl (wzgledem jego prognoz z poprzednich 168 h, bez przyszlosci);
- hybrydy z LISTY USTALONEJ Z GORY (bez wyboru na tescie): rodzina = srednia percentyli modeli,
  "A+B|srednia" = srednia rodzin, "A+B|zgoda" = minimum rodzin;
- prog wejscia: przyczynowy kwantyl wyniku hybrydy z poprzednich 168 h pod ~--tx wejsc/dzien na 48 symboli;
- ocena szybka (wynik r_* w ATR po kosztach vs losowy short, per okno) + silnik (test_regul_silnik.py).
Konfiguracje L+ERH, LX+EH, X+E pochodza z ensemble_szukaj na okresie pokrywajacym sie z oknami — oznaczone.
"""
import argparse, os, sys, time, json, subprocess, numpy as np, pandas as pd
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True); ap.add_argument("--katalog", required=True)
ap.add_argument("--etykieta", default="6.0,1.5,48,S"); ap.add_argument("--okna", type=int, default=12)
ap.add_argument("--dni", type=int, default=45); ap.add_argument("--embargo", type=int, default=7)
ap.add_argument("--tx", type=float, default=10.0); ap.add_argument("--ile", type=int, default=10)
ap.add_argument("--wiersze", type=int, default=600_000); ap.add_argument("--procesy", type=int, default=3)
ap.add_argument("--bez-silnika", action="store_true")
ap.add_argument("--koniec", default="", help="koniec ostatniego okna (te same okna na GH i podzie)")
ap.add_argument("--zbior", default="", help="GH: zbudowanie --przygotowany z magazynu, gdy go nie ma")
ap.add_argument("--tylko-okno", type=int, default=0, help="GH: tylko jedno okno (6 modeli) -> okno_NN.parquet, bez skladania")
ap.add_argument("--polacz", action="store_true", help="bez treningu: sklada okno_*.parquet z --katalog (wyniki z GH)")
a = ap.parse_args()
NJ = int(os.environ.get("ENS_NJ", "9"))
TP, SL, HZ, ST = a.etykieta.split(","); TP, SL, HZ = float(TP), float(SL), int(HZ)
KOL = f"r_{TP}_{SL}_{HZ}_{ST}"
RYNKOWE = {"fear_greed", "btc_trend_1h", "btc_trend_4h", "btc_trend_1d", "btc_rsi_4h", "btc_dominance_chg",
           "hour_sin", "hour_cos", "day_of_week", "x_weekend"}
TYPY = ("lgb", "xgb", "cat", "et", "rf", "hgb")
SKROT = {"lgb": "L", "xgb": "X", "cat": "C", "et": "E", "rf": "R", "hgb": "H"}
# (nazwa, rodzina A, rodzina B lub None, sposob) — ustalone PRZED testem
KONFIGI = [("LXC|srednia", "LXC", None, "srednia"), ("ERH|srednia", "ERH", None, "srednia"),
           ("LXC+ERH|srednia", "LXC", "ERH", "srednia"), ("LXC+ERH|zgoda", "LXC", "ERH", "zgoda"),
           ("L+ERH|srednia*", "L", "ERH", "srednia"), ("LX+EH|srednia*", "LX", "EH", "srednia"),
           ("LX+EH|zgoda*", "LX", "EH", "zgoda"), ("X+E|srednia*", "X", "E", "srednia")]
KONFIGI += [(f"{SKROT[t]}", SKROT[t], None, "srednia") for t in TYPY]


def zbuduj(t):
    if t == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=600,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=NJ)
    if t == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.7, colsample_bytree=0.7,
                                 min_child_weight=200, reg_lambda=5.0, n_jobs=NJ, verbosity=0)
    if t == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, l2_leaf_reg=10, verbose=0, thread_count=NJ)
    if t == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_leaf=500, max_features="sqrt", n_jobs=NJ, random_state=1)
    if t == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=500, max_features="sqrt",
                                      max_samples=0.5, n_jobs=NJ, random_state=1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=800,
                                          l2_regularization=5.0, random_state=1)


def Xp(d, c, t):
    x = d[c].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return x.fillna(0) if t in ("et", "rf") else x


def waznosc(m, t, X, y):
    if t in ("et", "rf"):
        return np.asarray(m.feature_importances_)
    if t == "hgb":
        from sklearn.inspection import permutation_importance
        n = min(15_000, len(X))
        return permutation_importance(m, X.iloc[:n], y.iloc[:n], n_repeats=2, random_state=1, scoring="roc_auc",
                                      n_jobs=min(NJ, 6)).importances_mean
    if t == "lgb":
        c = m.predict(X.iloc[:20000], pred_contrib=True)
    elif t == "xgb":
        import xgboost as xgb
        c = m.get_booster().predict(xgb.DMatrix(X.iloc[:20000]), pred_contribs=True)
    else:
        from catboost import Pool
        c = m.get_feature_importance(Pool(X.iloc[:20000]), type="ShapValues")
    return np.abs(np.asarray(c)[:, :-1]).mean(axis=0)


def przyczynowy_pct(t, v):
    """Percentyl kazdej wartosci wzgledem wartosci z poprzednich 168 h (sama przeszlosc, bez biezacej godziny)."""
    df = pd.DataFrame({"t": t, "v": v, "i": np.arange(len(v))}).sort_values("t")
    out = np.full(len(v), np.nan); okno = deque(); ref = np.array([])
    for tt, g in df.groupby("t", sort=True):
        zm = False
        while okno and okno[0][0] <= tt - pd.Timedelta(hours=168):
            okno.popleft(); zm = True
        if zm or len(ref) != sum(len(x) for _, x in okno):
            ref = np.sort(np.concatenate([x for _, x in okno])) if okno else np.array([])
        if len(ref) >= 500:
            out[g.i.values] = np.searchsorted(ref, g.v.values, side="right") / len(ref)
        okno.append((tt, g.v.values)); ref = np.sort(np.concatenate([ref, g.v.values]))
    return out


def zadanie(args):
    nr, start, t, cechy, kor = args
    Z = pd.read_parquet(a.przygotowany, columns=cechy + ["symbol", "_t", KOL])
    Z = Z[Z[KOL].notna() & ~Z.symbol.isin(PZ.MARTWE_COINY)]
    tr = Z[Z._t < start - pd.Timedelta(days=a.embargo)]
    if len(tr) > a.wiersze:
        tr = tr.sample(a.wiersze, random_state=nr)
    y = (tr[KOL] > 0).astype(int)
    s0 = tr.sample(min(250_000, len(tr)), random_state=2)
    m0 = zbuduj(t); m0.fit(Xp(s0, cechy, t), (s0[KOL] > 0).astype(int))
    w = waznosc(m0, t, Xp(s0, cechy, t), (s0[KOL] > 0).astype(int))
    wyb = []
    for j in np.argsort(-w):
        if all(kor[cechy[j]][g] <= 0.5 for g in wyb):
            wyb.append(cechy[j])
        if len(wyb) == a.ile:
            break
    m = zbuduj(t); m.fit(Xp(tr, wyb, t), y)
    kon = start + pd.Timedelta(days=a.dni)
    P = Z[(Z._t >= start - pd.Timedelta(days=7)) & (Z._t < kon)][["symbol", "_t"]].copy()
    P["p"] = m.predict_proba(Xp(Z.loc[P.index], wyb, t))[:, 1]
    P["pct"] = przyczynowy_pct(P._t.values, P.p.values)
    from sklearn.metrics import roc_auc_score
    ok_ = Z.loc[P.index][P._t >= start]
    auc = roc_auc_score((ok_[KOL] > 0).astype(int), P.loc[ok_.index, "p"]) if ok_[KOL].nunique() > 1 else np.nan
    import joblib
    os.makedirs(f"{a.katalog}/modele", exist_ok=True)
    joblib.dump({"model": m, "cechy": wyb, "okno": nr, "start": str(start), "etykieta": a.etykieta},
                f"{a.katalog}/modele/{t}_okno{nr:02d}.pkl")
    print(f"  okno {nr:2d} ({start.date()}) {t}: AUC okna {auc:.3f} | trening do {tr._t.max()} | {', '.join(wyb[:4])}...", flush=True)
    P["okno"] = nr; P["model"] = t
    return P, {"okno": nr, "model": t, "cechy": wyb, "auc_okno": float(auc), "koniec_treningu": str(tr._t.max()), "start": str(start)}


def main():
    t0 = time.time(); os.makedirs(a.katalog, exist_ok=True)
    if a.zbior and not os.path.exists(a.przygotowany):
        PZ.GEOMETRIE = [(2.5, 1.5), (4.0, 1.0), (6.0, 1.5)]; PZ.HORYZONTY = [24, 48, 72]   # jak gwaranci_cech
        PZ.przygotuj(a.zbior, []).to_parquet(a.przygotowany, index=False)
        print(f"zbior przygotowany w {time.time() - t0:.0f} s", flush=True)
    import pyarrow.parquet as pq, re, glob
    wk = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
    kol = pq.read_schema(a.przygotowany).names
    kand = [k for k in kol if not wk.match(k) and not k.startswith(("_", "label_", "trade_", "cel_", "__"))
            and k not in ("timestamp", "symbol", "close") and k not in PZ.MAKRO_USUNIETE and k not in RYNKOWE]
    Zs = pd.read_parquet(a.przygotowany, columns=kand + ["_t", "symbol"])
    Zs = Zs[~Zs.symbol.isin(PZ.MARTWE_COINY)]
    koniec = pd.Timestamp(a.koniec) if a.koniec else Zs._t.max().normalize() - pd.Timedelta(days=3) - pd.Timedelta(hours=HZ)
    assert koniec <= Zs._t.max() - pd.Timedelta(hours=HZ), f"--koniec {koniec} za daleko: dane do {Zs._t.max()}"
    starty = [koniec - pd.Timedelta(days=a.dni * k) for k in range(a.okna, 0, -1)]
    tr0 = Zs[Zs._t < starty[0] - pd.Timedelta(days=a.embargo)]
    cechy = [k for k in kand if pd.api.types.is_numeric_dtype(Zs[k]) and Zs[k].nunique() > 10 and tr0[k].notna().mean() > 0.5]
    kor = tr0[cechy].sample(min(200_000, len(tr0)), random_state=3).replace([np.inf, -np.inf], np.nan).rank().corr().abs().fillna(0).to_dict()
    del Zs, tr0
    print(f"WFV hybryd: {len(cechy)} cech coina, {a.okna}x{a.dni} dni {starty[0].date()} -> {koniec.date()}, "
          f"etykieta {KOL}, {a.tx} tx/dzien/48 sym.", flush=True)
    czesci, meta = [], []
    if a.polacz:
        for f in sorted(glob.glob(f"{a.katalog}/**/okno_*.parquet", recursive=True)):
            czesci.append(pd.read_parquet(f)); meta += json.load(open(f[:-8] + ".json"))
        ok = sorted({int(o) for o in pd.concat(czesci).okno.unique()})
        print(f"polaczono {len(czesci)} plikow okien z {a.katalog}: okna {ok}", flush=True)
        assert ok == list(range(1, a.okna + 1)), "brakuje okien"
    else:
        zad = [(nr, st, t, cechy, kor) for nr, st in enumerate(starty, 1) for t in TYPY
               if not a.tylko_okno or nr == a.tylko_okno]
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.procesy) as pool:
            for P, mt in pool.imap_unordered(zadanie, zad):
                czesci.append(P); meta.append(mt)
        if a.tylko_okno:
            f = f"{a.katalog}/okno_{a.tylko_okno:02d}.parquet"
            pd.concat(czesci).to_parquet(f, index=False); json.dump(meta, open(f[:-8] + ".json", "w"), indent=1)
            print(f"zapisano {f} ({time.time() - t0:.0f} s)"); return
    json.dump(meta, open(f"{a.katalog}/cechy_okien.json", "w"), indent=1)
    M = pd.DataFrame(meta)
    zle = (pd.to_datetime(M.koniec_treningu) >= pd.to_datetime(M.start) - pd.Timedelta(days=a.embargo)).sum()
    print(f"\nKONTROLA PODZIALU: zadan z treningiem siegajacym w okno/embargo: {zle} (musi byc 0)")
    print("AUC okna srednio:", M.groupby("model").auc_okno.mean().round(3).to_dict())
    # tabela: wiersz = (symbol, godzina, okno), kolumny = percentyle modeli
    D = pd.concat(czesci).pivot_table(index=["symbol", "_t", "okno"], columns="model", values="pct").reset_index()
    D = D.dropna(subset=list(TYPY))
    W = pd.read_parquet(a.przygotowany, columns=["symbol", "_t", "_koszt_atr", KOL])
    D = D.merge(W, on=["symbol", "_t"], how="left")
    D = D[D._t >= D.okno.map({nr: st for nr, st in enumerate(starty, 1)})]   # tylko okno (bez tygodnia rozbiegu)
    rodz = lambda s: D[[{v: k for k, v in SKROT.items()}[c] for c in s]].mean(axis=1)
    n_sym = D.symbol.nunique()
    q = 1 - min(0.5, a.tx / 48 / 24)
    los = (D[KOL] - D._koszt_atr)
    print(f"\nlosowy {ST} w oknach: ev {los.mean():+.3f} ATR, WR {(D[KOL] > 0).mean():.1%} ({n_sym} symboli)")
    wyn, syg = [], {}
    for nm, A, B, sp in KONFIGI:
        sc = rodz(A) if B is None else (pd.concat([rodz(A), rodz(B)], axis=1).mean(axis=1) if sp == "srednia"
                                        else np.minimum(rodz(A), rodz(B)))
        pc = przyczynowy_pct(D._t.values, sc.values)
        m = pc >= q
        S = D[m]; ev = S[KOL] - S._koszt_atr
        po_oknach = ev.groupby(S.okno).mean()
        dni = ev.groupby(S._t.dt.normalize()).mean()
        rng = np.random.default_rng(7)
        bs = [dni.values[rng.integers(0, len(dni), len(dni))].mean() for _ in range(300)] if len(dni) > 5 else [np.nan]
        wyn.append(dict(konfig=nm, n=len(S), tpd48=len(S) / max(1, D._t.dt.normalize().nunique()) * 48 / n_sym,
                        wr=(S[KOL] > 0).mean(), ev=ev.mean(), ev_p05=np.percentile(bs, 5),
                        okien_plus=int((po_oknach > 0).sum()), okien=int(len(po_oknach)),
                        najgorsze_okno=po_oknach.min(), najlepsze_okno=po_oknach.max()))
        X = S[["symbol", "_t"]].copy(); X["akcja"] = -1 if ST == "S" else 1
        X["ts_ms"] = X["_t"].astype("datetime64[ms]").astype("int64"); syg[nm] = X
    R = pd.DataFrame(wyn).sort_values("ev", ascending=False)
    R.to_csv(f"{a.katalog}/hybrydy.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== WFV HYBRYD {KOL} (ocena szybka: wynik r_* po kosztach; * = konfiguracja z ensemble_szukaj na tym samym okresie) ===")
    print(R.round(3).to_string(index=False))
    print(f"(czas {time.time() - t0:.0f} s)", flush=True)
    if a.bez_silnika:
        return
    for nm in [r.konfig for r in R.itertuples() if r.ev > 0][:6]:
        f = f"{a.katalog}/sygnaly_{nm.replace('|', '_').replace('+', 'p').replace('*', '')}.parquet"
        syg[nm].to_parquet(f, index=False)
        for zakres in ("wl48", "wszystkie"):
            print(f"\n######## SILNIK: {nm}, symbole {zakres}", flush=True)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_regul_silnik.py"),
                            "--regula", "plik", "--plik", f, "--tp", str(TP), "--sl", str(SL), "--od", str(starty[0].date()),
                            "--symbole", zakres, "--procesy", "8", "--wyniki", f[:-8] + f"_{zakres}.pkl"])


if __name__ == "__main__":
    main()

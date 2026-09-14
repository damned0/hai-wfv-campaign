#!/usr/bin/env python3
"""Szukanie konfiguracji ENSEMBLE: najpierw boostingi, potem drzewa, potem polaczenie (2026-09-14).

Decyzja uzytkownika: "znajdz najlepszy boost konfig ensembla, pozniej drzew, zas skompilujemy do kupy".

Dla kazdej etykiety (strona x geometria x horyzont):
- 6 modeli bazowych na CECHACH COINA (bez makro i bez cech wspolnych dla rynku): LGB, XGB, CAT, ET, RF, HGB;
  kazdy sam dobiera 12 cech (waznosc na treningu, bez dubli |rho|<=0.5), mocna regularyzacja
  (poprzednie okna: AUC trening 0.58-0.64 vs poza probka 0.51-0.54 -> przeuczenie);
- konfiguracje: podzbiory boostow {L,X,C} i drzew {E,R,H} x laczenie {srednia rang, zgoda (min rang),
  glos 2 z n}; potem najlepszy boost + najlepsze drzewa i cala szostka.
Uczciwosc: trening przed cieciem kalendarzowym (+7 dni przerwy). Test dzielony na POLOWE WALIDACYJNA
(tu wybieramy najlepsza konfiguracje) i POLOWE SPRAWDZAJACA (tylko raport wybranych) — inaczej wybor
najlepszej z ~50 konfiguracji na tych samych danych bylby optymistyczny.
"""
import argparse, os, sys, re, time, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True); ap.add_argument("--wyniki", required=True)
ap.add_argument("--etykiety", default="2.5,1.5,48,S;2.5,1.5,72,S;4.0,1.0,48,S;6.0,1.5,48,S;6.0,1.5,72,S;"
                                      "2.5,1.5,48,L;2.5,1.5,72,L;6.0,1.5,48,L")
ap.add_argument("--procesy", type=int, default=3); ap.add_argument("--ile", type=int, default=12)
ap.add_argument("--wiersze", type=int, default=600_000)
a = ap.parse_args()
NJ = int(os.environ.get("ENS_NJ", "9"))
RYNKOWE = {"fear_greed", "btc_trend_1h", "btc_trend_4h", "btc_trend_1d", "btc_rsi_4h", "btc_dominance_chg",
           "hour_sin", "hour_cos", "day_of_week", "x_weekend"}
BOOST, DRZEWA = ("lgb", "xgb", "cat"), ("et", "rf", "hgb")
SKROT = {"lgb": "L", "xgb": "X", "cat": "C", "et": "E", "rf": "R", "hgb": "H"}


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


def ocen(sp, wynik, kol, hz, tx):
    dni = max((sp._t.max() - sp._t.min()).days, 1); ns = sp.symbol.nunique()
    wz = PZ.wybierz_transakcje(sp, wynik, hz, tx, dni, ns)
    if len(wz) < 40:
        return None
    r = wz[kol].values; ev = r - wz._koszt_atr.values
    kod, ud = pd.factorize(wz._t.dt.normalize()); rng = np.random.default_rng(7)
    idx = [np.where(kod == k)[0] for k in range(len(ud))]
    bs = [ev[np.concatenate([idx[k] for k in rng.integers(0, len(ud), len(ud))])].mean() for _ in range(200)]
    rz = wz._rez.values
    return dict(n=len(wz), tpd48=len(wz) / dni * 48 / ns, wr=(r > 0).mean(), ev=ev.mean(), ev_p05=np.percentile(bs, 5),
                ev_wzrost=ev[rz == 1].mean() if (rz == 1).any() else np.nan,
                ev_spadek=ev[rz == 0].mean() if (rz == 0).any() else np.nan)


def zadanie(args):
    tp, sl, hz, st, cechy, kor = args
    kol = f"r_{tp}_{sl}_{hz}_{st}"
    Z = pd.read_parquet(a.przygotowany, columns=cechy + ["symbol", "_t", "_koszt_atr", kol])
    Z = Z[Z[kol].notna()].reset_index(drop=True)
    b = pd.read_parquet(f"{PZ.ROOT}/data_warehouse/ohlcv/binance/1h/BTC.parquet")
    b["_t"] = pd.to_datetime(b.timestamp).astype("datetime64[ns]"); b = b.drop_duplicates("_t").set_index("_t").sort_index()
    rez = (b.close / b.close.shift(168) - 1 > 0).astype(int)
    Z["_rez"] = rez.reindex(pd.DatetimeIndex(Z._t.values).astype("datetime64[ns]")).fillna(-1).astype(int).values
    tr, te = PZ.podziel(Z)
    if len(tr) > a.wiersze:
        tr = tr.sample(a.wiersze, random_state=1)
    y = (tr[kol] > 0).astype(int)
    pol = te._t.min() + (te._t.max() - te._t.min()) / 2
    wal, spr = te[te._t < pol], te[te._t >= pol]
    pred, meta = {}, {}
    for t in BOOST + DRZEWA:
        s0 = tr.sample(min(250_000, len(tr)), random_state=2)
        m0 = zbuduj(t); m0.fit(Xp(s0, cechy, t), (s0[kol] > 0).astype(int))
        w = waznosc(m0, t, Xp(s0, cechy, t), (s0[kol] > 0).astype(int))
        wyb = []
        for j in np.argsort(-w):
            if all(kor[cechy[j]][g] <= 0.5 for g in wyb):
                wyb.append(cechy[j])
            if len(wyb) == a.ile:
                break
        m = zbuduj(t); m.fit(Xp(tr, wyb, t), y)
        p = m.predict_proba(Xp(te, wyb, t))[:, 1]
        pred[t] = pd.Series(p, index=te.index).rank(pct=True)   # rangi — rozne skale prawdopodobienstw
        meta[t] = wyb
    # konfiguracje ensemble
    def konfigi(rodzina):
        out = {}
        for k in range(1, len(rodzina) + 1):
            for sub in itertools.combinations(rodzina, k):
                nm = "".join(SKROT[s] for s in sub)
                R = np.column_stack([pred[s].values for s in sub])
                out[f"{nm}|srednia"] = R.mean(axis=1)
                if k >= 2:
                    out[f"{nm}|zgoda"] = R.min(axis=1)
                    g = (R >= 0.97).sum(axis=1)
                    out[f"{nm}|glos2"] = np.where(g >= 2, R.mean(axis=1), 0.0)
        return out
    K = {**{("boost", k): v for k, v in konfigi(BOOST).items()}, **{("drzewa", k): v for k, v in konfigi(DRZEWA).items()}}
    wyn = []
    for (rodz, nm), sc in K.items():
        s = pd.Series(sc, index=te.index)
        for tx in (10, 15):
            ow = ocen(wal, s.loc[wal.index].values, kol, hz, tx); os_ = ocen(spr, s.loc[spr.index].values, kol, hz, tx)
            if ow and os_:
                wyn.append(dict(tp=tp, sl=sl, hz=hz, strona=st, rodzina=rodz, konfig=nm, tx=tx,
                                **{f"wal_{k}": v for k, v in ow.items()}, **{f"spr_{k}": v for k, v in os_.items()}))
    W = pd.DataFrame(wyn)
    # polaczenie: najlepszy boost + najlepsze drzewa (wybor na WALIDACJI, tx=10) i cala szostka
    best = {}
    for rodz in ("boost", "drzewa"):
        w = W[(W.rodzina == rodz) & (W.tx == 10)]
        if len(w):
            best[rodz] = w.sort_values("wal_ev", ascending=False).konfig.iloc[0]
    if len(best) == 2:
        sb = pd.Series(K[("boost", best["boost"])], index=te.index).rank(pct=True)
        sd = pd.Series(K[("drzewa", best["drzewa"])], index=te.index).rank(pct=True)
        szost = np.column_stack([pred[t].values for t in BOOST + DRZEWA]).mean(axis=1)
        for nm, sc in ((f"{best['boost']}+{best['drzewa']}|srednia", ((sb + sd) / 2).values),
                       (f"{best['boost']}+{best['drzewa']}|zgoda", np.minimum(sb, sd).values),
                       ("LXCERH|srednia", szost)):
            s = pd.Series(sc, index=te.index)
            for tx in (10, 15):
                ow = ocen(wal, s.loc[wal.index].values, kol, hz, tx); os_ = ocen(spr, s.loc[spr.index].values, kol, hz, tx)
                if ow and os_:
                    W = pd.concat([W, pd.DataFrame([dict(tp=tp, sl=sl, hz=hz, strona=st, rodzina="polaczone", konfig=nm, tx=tx,
                                  **{f"wal_{k}": v for k, v in ow.items()}, **{f"spr_{k}": v for k, v in os_.items()})])])
    W["wr_los_wal"] = (wal[kol] > 0).mean(); W["wr_los_spr"] = (spr[kol] > 0).mean()
    W["cechy"] = str({SKROT[t]: meta[t] for t in meta})
    print(f"  {st} {tp}/{sl} {hz}h: {len(W)} konfiguracji, najlepszy boost {best.get('boost')}, drzewa {best.get('drzewa')}", flush=True)
    return W


def main():
    t0 = time.time()
    import pyarrow.parquet as pq
    wk = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
    kol = pq.read_schema(a.przygotowany).names
    kand = [k for k in kol if not wk.match(k) and not k.startswith(("_", "label_", "trade_", "cel_", "__"))
            and k not in ("timestamp", "symbol", "close") and k not in PZ.MAKRO_USUNIETE and k not in RYNKOWE]
    Zs = pd.read_parquet(a.przygotowany, columns=kand + ["_t"])
    os.environ["POSZ_CIECIE"] = os.environ.get("POSZ_CIECIE_STALE", "2024-10-31 15:00:00")
    tr0, _ = PZ.podziel(Zs)
    cechy = [k for k in kand if pd.api.types.is_numeric_dtype(Zs[k]) and Zs[k].nunique() > 10 and tr0[k].notna().mean() > 0.5]
    kor = tr0[cechy].sample(200_000, random_state=3).replace([np.inf, -np.inf], np.nan).rank().corr().abs().fillna(0).to_dict()
    del Zs, tr0
    et = [tuple(e.split(",")) for e in a.etykiety.split(";") if e]
    zad = [(float(t), float(s), int(h), st, cechy, kor) for t, s, h, st in et]
    print(f"ensemble: {len(cechy)} cech coina, {len(zad)} etykiet, 6 modeli bazowych, ciecie {os.environ['POSZ_CIECIE']}", flush=True)
    import multiprocessing as mp
    wyn = []
    with mp.get_context("spawn").Pool(a.procesy) as pool:
        for W in pool.imap_unordered(zadanie, zad):
            wyn.append(W); pd.concat(wyn).to_csv(a.wyniki, index=False)
    W = pd.concat(wyn); W.to_csv(a.wyniki, index=False)
    k = ["konfig", "wal_ev", "spr_ev", "spr_ev_p05", "spr_wr", "spr_tpd48", "spr_ev_wzrost", "spr_ev_spadek"]
    for rodz in ("boost", "drzewa", "polaczone"):
        w = W[(W.rodzina == rodz) & (W.tx == 10)]
        s = w.groupby("konfig")[["wal_ev", "spr_ev", "spr_ev_p05", "spr_wr", "spr_tpd48", "spr_ev_wzrost", "spr_ev_spadek"]].mean()
        print(f"\n=== {rodz.upper()}: srednio po etykietach, 10 tx/dzien — wybor wg WALIDACJI, raport SPRAWDZIANU ===")
        print(s.sort_values("wal_ev", ascending=False).head(10).round(3).to_string())
    for st in ("S", "L"):
        w = W[(W.strona == st) & (W.tx == 10)]
        print(f"\n=== strona {st}: najlepsza konfiguracja per etykieta (wybor na walidacji) ===")
        if not len(w):
            continue
        w2 = w.reset_index(drop=True)
        best = w2.loc[w2.groupby(["tp", "sl", "hz"]).wal_ev.idxmax()]
        print(best[["tp", "sl", "hz", "rodzina", "konfig", "wal_ev", "spr_ev", "spr_ev_p05", "spr_wr", "wr_los_spr", "spr_tpd48"]].round(3).to_string(index=False))
    print(f"\ncalosc {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

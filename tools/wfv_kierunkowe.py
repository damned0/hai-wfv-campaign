#!/usr/bin/env python3
"""WFV modeli KIERUNKOWYCH: osobny model LONG i osobny SHORT (ET / RF / HGB) — 2026-09-14.

Decyzja uzytkownika: zamiast jednego modelu 3-klasowego (NEUTRAL 93% -> uczciwe modele milcza)
dwa modele binarne na strone: "czy long 2.5/1.5 ATR w 48 h wygra" i to samo dla shorta.

Walk-forward (uczciwie): 12 okien po 45 dni konczacych sie na --koniec. Dla kazdego okna:
- trening WYLACZNIE na danych sprzed (poczatek okna - embargo 7 dni), probka do --wiersze;
- przecechowanie w oknie: model na wszystkich cechach -> waznosc -> 10 najlepszych bez dubli (|rho|<=0.5);
- prognoza na oknie; prog wejscia PRZYCZYNOWY: kwantyl prognoz z poprzednich 7 dni (bez przyszlosci),
  dobrany pod ~--tx wejsc dziennie na 48 symboli.
Sygnaly -> plik parquet -> prawdziwy backtester silnika (HAI_REGULA=plik: cel/stop ATR, czesciowe
zamkniecie, trailing, koszty) przez tools/test_regul_silnik.py, z limitem 10 pozycji i rezimem rynku.

--tylko-coin: bez cech wspolnych dla calego rynku (model ma wybierac coina, nie dzien).
Wariant z PRZECIEKIEM (koncept porownawczy): ten sam przebieg na zbiorze zbudowanym z
HAI_TRENING_JAK_PRODUKCJA=1 — tylko do pokazania, ile dawal przeciek; nie na produkcje.
"""
import argparse, os, sys, time, json, itertools, subprocess, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True); ap.add_argument("--katalog", required=True)
ap.add_argument("--modele", default="et,rf,hgb"); ap.add_argument("--etykieta", default="2.5,1.5,48")
ap.add_argument("--okna", type=int, default=12); ap.add_argument("--dni", type=int, default=45)
ap.add_argument("--koniec", default=""); ap.add_argument("--embargo", type=int, default=7)
ap.add_argument("--tx", type=float, default=10.0); ap.add_argument("--ile", type=int, default=10)
ap.add_argument("--wiersze", type=int, default=400_000); ap.add_argument("--procesy", type=int, default=4)
ap.add_argument("--tylko-coin", action="store_true"); ap.add_argument("--bez-silnika", action="store_true")
a = ap.parse_args()
NJ = int(os.environ.get("KIER_NJ", "6"))
RYNKOWE = ("fear_greed", "btc_trend_1h", "btc_trend_4h", "btc_trend_1d", "btc_rsi_4h", "btc_dominance_chg",
           "hour_sin", "hour_cos", "day_of_week", "x_weekend")
TP, SL, HZ = a.etykieta.split(","); TP, SL, HZ = float(TP), float(SL), int(HZ)


def zbuduj(typ):
    if typ == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200, max_features="sqrt", n_jobs=NJ, random_state=1)
    if typ == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200, max_features="sqrt",
                                      max_samples=0.5, n_jobs=NJ, random_state=1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, random_state=1)


def X_(df, cechy, typ):
    X = df[cechy].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return X.fillna(0) if typ in ("et", "rf") else X


def zadanie(args):
    nr, start, typ, strona, cechy, kor = args
    kol = f"r_{TP}_{SL}_{HZ}_{strona}"
    Z = pd.read_parquet(a.przygotowany, columns=cechy + ["symbol", "_t", kol])
    Z = Z[Z[kol].notna()]
    koniec_tr = start - pd.Timedelta(days=a.embargo)
    tr = Z[Z._t < koniec_tr]
    if len(tr) > a.wiersze:
        tr = tr.sample(a.wiersze, random_state=nr)
    y = (tr[kol] > 0).astype(int)
    m0 = zbuduj(typ); m0.fit(X_(tr, cechy, typ), y)
    if typ in ("et", "rf"):
        waz = np.asarray(m0.feature_importances_)
    else:
        from sklearn.inspection import permutation_importance
        n = min(15_000, len(tr))
        waz = permutation_importance(m0, X_(tr.iloc[:n], cechy, typ), y.iloc[:n], n_repeats=2, random_state=1,
                                     scoring="roc_auc", n_jobs=min(NJ, 6)).importances_mean
    wyb = []
    for j in np.argsort(-waz):
        if all(kor[cechy[j]][g] <= 0.5 for g in wyb):
            wyb.append(cechy[j])
        if len(wyb) == a.ile:
            break
    m = zbuduj(typ); m.fit(X_(tr, wyb, typ), y)
    koniec_ok = start + pd.Timedelta(days=a.dni)
    # KONTROLA MODELU: AUC na treningu (w probce) vs na oknie (poza probka) — duza roznica = przeuczenie
    from sklearn.metrics import roc_auc_score
    ok_ = Z[(Z._t >= start) & (Z._t < koniec_ok)]
    auc_tr = roc_auc_score(y, m.predict_proba(X_(tr, wyb, typ))[:, 1])
    auc_ok = roc_auc_score((ok_[kol] > 0).astype(int), m.predict_proba(X_(ok_, wyb, typ))[:, 1]) if ok_[kol].nunique() > 1 else np.nan
    import joblib
    os.makedirs(f"{a.katalog}/modele", exist_ok=True)
    joblib.dump({"model": m, "cechy": wyb, "okno": nr, "start": str(start), "etykieta": (TP, SL, HZ, strona)},
                f"{a.katalog}/modele/{typ}_{strona}_okno{nr:02d}.pkl")
    # prognozy od tygodnia przed oknem (do przyczynowego progu) do konca okna
    P = Z[(Z._t >= start - pd.Timedelta(days=7)) & (Z._t < koniec_ok)][["symbol", "_t"]].copy()
    P["p"] = m.predict_proba(X_(Z.loc[P.index], wyb, typ))[:, 1]
    # prog: kwantyl prognoz z poprzednich 168 h (wszystkie symbole) -> ~tx wejsc/dzien na 48 symboli
    n_sym = max(P.symbol.nunique(), 1)
    q = 1 - min(0.5, a.tx * n_sym / 48 / (24 * n_sym))
    godz = P.groupby("_t")["p"].apply(np.array).sort_index()
    t_ix = godz.index; prog = pd.Series(np.nan, index=t_ix)
    okno = []
    for i, t in enumerate(t_ix):
        # tylko przeszlosc: godziny (t-168h, t)
        while okno and okno[0][0] <= t - pd.Timedelta(hours=168):
            okno.pop(0)
        if okno:
            prog[t] = np.quantile(np.concatenate([x for _, x in okno]), q)
        okno.append((t, godz.iloc[i]))
    P["prog"] = P["_t"].map(prog)
    S = P[(P._t >= start) & (P.p >= P.prog)].copy()
    S["akcja"] = 1 if strona == "L" else -1
    S["okno"] = nr; S["model"] = typ; S["strona"] = strona
    print(f"  okno {nr} ({start.date()}) {typ} {strona}: {len(S)} sygnalow | AUC trening {auc_tr:.3f} / okno {auc_ok:.3f} | "
          f"wygranych w treningu {y.mean():.1%} | cechy: {', '.join(wyb[:5])}...", flush=True)
    return S[["symbol", "_t", "akcja", "p", "okno", "model", "strona"]], {"okno": nr, "model": typ, "strona": strona,
            "cechy": wyb, "waznosc": [float(waz[cechy.index(c)]) for c in wyb], "auc_trening": float(auc_tr),
            "auc_okno": float(auc_ok), "n_trening": int(len(tr)), "wygranych_trening": float(y.mean()),
            "koniec_treningu": str(tr._t.max()), "sygnalow": int(len(S))}


def main():
    t0 = time.time(); os.makedirs(a.katalog, exist_ok=True)
    import pyarrow.parquet as pq
    kol_all = pq.read_schema(a.przygotowany).names
    wyk = set(PZ.MAKRO_USUNIETE) | (set(RYNKOWE) if a.tylko_coin else set())
    import re
    wynik_kol = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
    kand = [k for k in kol_all if not wynik_kol.match(k) and not k.startswith(("_", "label_", "trade_", "cel_", "__"))
            and k not in ("timestamp", "symbol", "close") and k not in wyk]
    Zs = pd.read_parquet(a.przygotowany, columns=kand + ["_t"])
    koniec = pd.Timestamp(a.koniec) if a.koniec else Zs._t.max().normalize() - pd.Timedelta(days=3)
    starty = [koniec - pd.Timedelta(days=a.dni * k) for k in range(a.okna, 0, -1)]
    tr0 = Zs[Zs._t < starty[0] - pd.Timedelta(days=a.embargo)]
    cechy = [k for k in kand if pd.api.types.is_numeric_dtype(Zs[k]) and Zs[k].nunique() > 10 and tr0[k].notna().mean() > 0.5]
    kor = tr0[cechy].sample(min(200_000, len(tr0)), random_state=3).replace([np.inf, -np.inf], np.nan).rank().corr().abs().fillna(0).to_dict()
    del Zs, tr0
    print(f"WFV kierunkowe: {len(cechy)} cech{' (tylko coin)' if a.tylko_coin else ''}, okna {a.okna}x{a.dni} dni "
          f"od {starty[0].date()} do {koniec.date()}, etykieta {TP}/{SL} ATR {HZ} h, modele {a.modele}", flush=True)
    zad = [(nr, st, typ, s, cechy, kor) for nr, st in enumerate(starty, 1) for typ in a.modele.split(",") for s in ("L", "S")]
    import multiprocessing as mp
    syg, meta = [], []
    with mp.get_context("spawn").Pool(a.procesy) as pool:
        for S, mt in pool.imap_unordered(zadanie, zad):
            syg.append(S); meta.append(mt)
    Y = pd.concat(syg, ignore_index=True)
    Y["ts_ms"] = Y["_t"].astype("datetime64[ms]").astype("int64")
    Y.to_parquet(f"{a.katalog}/sygnaly_wszystkie.parquet", index=False)
    json.dump(meta, open(f"{a.katalog}/cechy_okien.json", "w"), indent=1)
    print(f"sygnaly: {len(Y)} ({time.time() - t0:.0f} s)", flush=True)
    if a.bez_silnika:
        return
    for typ in a.modele.split(","):
        S = Y[Y.model == typ]
        # konflikt long i short na tym samym symbolu i godzinie -> oba odpadaja
        k = S.groupby(["symbol", "ts_ms"]).akcja.transform("nunique")
        S = S[k == 1].drop_duplicates(["symbol", "ts_ms"])
        f = f"{a.katalog}/sygnaly_{typ}.parquet"; S.to_parquet(f, index=False)
        print(f"\n######## SILNIK: {typ} ({len(S)} sygnalow)", flush=True)
        for zakres, suf in (("wl48", ""), ("wszystkie", "_106")):
            print(f"--- symbole: {zakres}", flush=True)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_regul_silnik.py"),
                            "--regula", "plik", "--plik", f, "--tp", str(TP), "--sl", str(SL), "--od", str(starty[0].date()),
                            "--symbole", zakres, "--procesy", str(max(a.procesy, 4)),
                            "--wyniki", f"{a.katalog}/transakcje_{typ}{suf}.pkl"])
    # KONTROLA: czy trening NIGDY nie siega za poczatek okna minus embargo
    M = pd.DataFrame(meta)
    M["start_okna"] = [str(starty[int(o) - 1]) for o in M.okno]
    zle = (pd.to_datetime(M.koniec_treningu) >= pd.to_datetime(M.start_okna) - pd.Timedelta(days=a.embargo)).sum()
    print(f"\nKONTROLA PODZIALU: zadan z treningiem siegajacym w okno/embargo: {zle} (musi byc 0)")
    print("AUC trening vs okno (srednio):"); print(M.groupby(["model", "strona"])[["auc_trening", "auc_okno", "wygranych_trening", "sygnalow"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Szerokie poszukiwanie uczciwego sygnalu: horyzonty x geometrie x zestawy cech x modele.

DLACZEGO NIE SAMO WR (2026-09-13)
---------------------------------
WR da sie "kupic" geometria: przy celu 0.6xATR i stopie 1.5xATR losowe wejscia
trafiaja ~71%, a mimo to traca (jedna strata zjada 2.5 wygranej). Kazda kombinacja
jest wiec oceniana trzema liczbami: WR, PRZEWAGA nad WR samej geometrii (losowe
wejscia z tym samym celem/stopem) i wartosc oczekiwana PO PROWIZJACH.

UCZCIWOSC
---------
- cechy z obecnego ml_trainer (bez przecieku 4h/1d, sprawdzone testami)
- podzial KALENDARZOWY wspolny dla symboli (podziel): starsze 60% czasu = selekcja + trening,
  7 dni przerwy, reszta = ocena (per symbol przeciekal — patrz podziel())
- prog wejscia dobierany z ROZKLADU wynikow modelu (bez etykiet) pod zadana liczbe
  transakcji dziennie, jedna pozycja na symbol naraz (jak silnik na zywo)
- wynik transakcji z prawdziwej sciezki ceny: pierwsze dotkniecie celu albo stopu
  w horyzoncie (oba w tej samej swiecy -> liczony STOP, ostroznie), inaczej zamkniecie
  po horyzoncie; minus koszty (2 x (prowizja 0.06% + poslizg 0.05%) ceny)

Uzycie:
  python3 poszukiwania.py --zbior Z.parquet --etap A [--symbole BTC,ETH] [--wyniki out.csv]
  etap A: LGB na calej siatce; etap B: 6 typow modeli dla najlepszych z A (--z-etapu A.csv)
"""
import argparse, os, sys, time, logging, itertools, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
logging.disable(logging.WARNING)
ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/hai_common")
try:   # makro tradfi wyciete 2026-09-14 — stare zbiory na dysku wciaz maja te kolumny
    from hai_common.ml_trainer import MAKRO_USUNIETE
except Exception:
    MAKRO_USUNIETE = ('gold_chg', 'oil_wti_chg', 'sp500_chg', 'vix_chg', 'us10y_chg', 'dxy_chg')

GEOMETRIE = [(0.6, 1.5), (0.8, 1.5), (1.0, 1.5), (1.2, 1.5), (2.5, 1.5), (4.0, 1.0)]
HORYZONTY = [3, 6, 12, 24, 48]
KOSZT_CENY = 2 * (0.0006 + 0.0005) * 100      # % ceny na cala transakcje
CEL_TX_DZIEN = (10, 12, 15)                   # na 48 symboli
ZAPIS_MODELI = os.environ.get("POSZ_ZAPIS_MODELI", "")   # etap B: katalog na modele finalistow
RODZINY = {
    "momentum": ["rsi", "x_rsi_7", "x_rsi_21", "r_di_spread", "r_cci_20", "ema_mid_r", "ema_slow_r",
                 "ema_fast_r", "x_ema9_over_21", "x_ema21_over_55", "r_momentum_20", "momentum",
                 "price_position_bb", "macd_hist", "r_stoch_rsi"],
    "wolumen_flow": ["volume_ratio", "volume_zscore", "r_volume_zscore_50", "taker_buy_ratio", "cvd_z6",
                     "cvd_z24", "delta_pct_ma6", "x_obv_slope_24h", "e_volume_delta_imbalance",
                     "vwap_dev", "r_vwap_distance", "e_atr_pct_x_vol_zscore", "atr_pct", "bb_bandwidth_pct", "adx_14"],
    "struktura": ["sr_dist_pct", "sr_node_strength", "fib_dist_pct", "r_htf_trend_4h", "trend_1h", "trend_4h",
                  "trend_1d", "rsi_4h", "rsi_1d", "bars_cross", "e_tk_cross", "e_ichimoku_cloud_thickness",
                  "dist_above_liq", "dist_below_liq", "hour_cos"],
    "makro_deryw": ["funding_rate", "funding_change_24h", "oi_change_24h", "oi_zscore_30d", "oi_total_log",
                    "ls_ratio", "ls_ratio_chg_24h",   # dxy/vix/sp500/us10y wyciete 2026-09-14
                    "btc_dominance_chg", "btc_corr_24h", "btc_beta_72h", "rel_strength_btc"],
}


def wyniki_sciezki(c, h, l, atr, tp, sl, hz, strona):
    """Wynik transakcji w ATR dla KAZDEGO wiersza (wejscie na zamknieciu swiecy i)."""
    n = len(c); r = np.full(n, np.nan)
    ok = (np.arange(n) + hz < n) & (atr > 0)
    zrob = np.zeros(n, bool)
    for k in range(1, hz + 1):
        idx = np.minimum(np.arange(n) + k, n - 1)
        if strona == 1:
            up = (h[idx] - c) / atr; dn = (c - l[idx]) / atr
        else:
            up = (c - l[idx]) / atr; dn = (h[idx] - c) / atr
        stop = (~zrob) & (dn >= sl); cel = (~zrob) & (up >= tp) & ~stop
        r[stop] = -sl; r[cel] = tp
        zrob |= stop | cel
    koniec = np.minimum(np.arange(n) + hz, n - 1)
    mtm = strona * (c[koniec] - c) / atr
    r = np.where(zrob, r, mtm)
    r[~ok] = np.nan
    return r


def przygotuj(zbior, symbole):
    import hai_common.ml_trainer as mt
    if not os.path.exists(zbior):
        print("brak zbioru — buduje pelny, uczciwy (obecny ml_trainer)...", flush=True)
        Z = mt.build_dataset(); Z.to_parquet(zbior, index=False)
    Z = pd.read_parquet(zbior)
    if symbole: Z = Z[Z.symbol.isin(symbole)]
    Z["_t"] = pd.to_datetime(Z["timestamp"])
    czesci = []
    for s, g in Z.groupby("symbol"):
        o = pd.read_parquet(f"{ROOT}/data_warehouse/ohlcv/binance/1h/{s}.parquet")
        o["timestamp"] = pd.to_datetime(o["timestamp"]); o = o.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        c, h, l = (o[k].values.astype(float) for k in ("close", "high", "low"))
        atr = mt.calc_atr(h, l, c, mt.ATR_PERIOD)
        pos = pd.Series(np.arange(len(o)), index=o["timestamp"])
        idx = pos.reindex(g["_t"]).values
        g = g[~np.isnan(idx)].copy(); idx = idx[~np.isnan(idx)].astype(int)
        g["_koszt_atr"] = KOSZT_CENY / np.maximum(g["atr_pct"].values if "atr_pct" in g else 1.0, 0.05)
        for (tp, sl), hz in itertools.product(GEOMETRIE, HORYZONTY):
            for st, nm in ((1, "L"), (-1, "S")):
                g[f"r_{tp}_{sl}_{hz}_{nm}"] = wyniki_sciezki(c, h, l, atr, tp, sl, hz, st)[idx]
        czesci.append(g)
    Z = pd.concat(czesci, ignore_index=True)
    Z["_r"] = Z.groupby("symbol")["_t"].rank(pct=True)
    return Z


EMBARGO_DNI = 7


def podziel(Z):
    """Trening/test po KALENDARZU, wspolnie dla wszystkich symboli (2026-09-14).

    Bylo: starsze 60% / nowsze 40% osobno dla kazdego symbolu. Symbole maja rozna
    dlugosc historii, wiec trening mlodych coinow obejmowal daty bedace testem
    starych — a cechy wspolne dla rynku (makro, funding zbiorczy) pozwalaly
    zapamietac "tego dnia rynek spadal". Short 1.0/1.5 6h: per symbol WR 74.5%
    ev +0.22, kalendarzowo WR 61.6% ev -0.05 (tools/test_podzialu.py).
    Ciecie = 60. percentyl czasu calego zbioru (POSZ_CIECIE z main), potem
    EMBARGO_DNI przerwy, zeby etykiety 48h treningu nie siegaly w test.
    """
    D = pd.Timestamp(os.environ["POSZ_CIECIE"])
    return Z[Z._t < D], Z[Z._t >= D + pd.Timedelta(days=EMBARGO_DNI)]


def wybierz_transakcje(df, wynik, hz, tx_dzien, dni, n_sym):
    """Najwyzsze wyniki modelu -> transakcje; jedna pozycja na symbol przez hz godzin."""
    cel_n = tx_dzien * dni * n_sym / 48
    prog = np.quantile(wynik, max(0.0, 1 - 3 * cel_n / len(wynik)))   # zapas na odrzucone nakladki
    kand = df.assign(_w=wynik)[wynik >= prog].sort_values("_t")
    wziete = []
    for s, g in kand.groupby("symbol"):
        wolne = pd.Timestamp.min
        for t, w in zip(g["_t"], g.index):
            if t >= wolne:
                wziete.append(w); wolne = t + pd.Timedelta(hours=hz)
    wz = kand.loc[wziete].sort_values("_w", ascending=False).head(int(cel_n))
    return wz


def jeden(args):
    """Jeden eksperyment: etykieta (geometria, horyzont, strona) x zestaw cech x model."""
    (plik, tp, sl, hz, st, zestaw, cechy, model) = args
    Z = pd.read_parquet(plik, columns=list(set(cechy + ["symbol", "_t", "_r", "_koszt_atr",
                                                         f"r_{tp}_{sl}_{hz}_{st}"])))
    kol = f"r_{tp}_{sl}_{hz}_{st}"
    Z = Z[Z[kol].notna()]
    sel, spr = podziel(Z)
    if len(sel) > 1_500_000: sel = sel.sample(1_500_000, random_state=1)
    y = (sel[kol] > 0).astype(int)
    X, Xs = sel[cechy].astype(np.float32), spr[cechy].astype(np.float32)
    # ml_trainer:387 (strefy popytu/podazy) potrafi dac +-inf; RF/ET/HistGB sie na tym wysypuja
    X = X.replace([np.inf, -np.inf], np.nan); Xs = Xs.replace([np.inf, -np.inf], np.nan)
    if model == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
    elif model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=8, verbosity=0)
    elif model == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, thread_count=8)
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200, n_jobs=8, random_state=1)
        X, Xs = X.fillna(0), Xs.fillna(0)
    elif model == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        m = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200, n_jobs=8, random_state=1)
        X, Xs = X.fillna(0), Xs.fillna(0)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31)
    m.fit(X, y)
    p = m.predict_proba(Xs)[:, 1]
    dni = (spr._t.max() - spr._t.min()).days; n_sym = spr.symbol.nunique()
    baza_wr = (spr[kol] > 0).mean()
    baza_ev = (spr[kol] - spr._koszt_atr).mean()
    out = []
    for tx in CEL_TX_DZIEN:
        wz = wybierz_transakcje(spr, p, hz, tx, dni, n_sym)
        if len(wz) < 50: continue
        r = wz[kol].values; k = wz._koszt_atr.values
        # Pulapka dni (2026-09-13): cechy dzienne (makro) daja serie transakcji jednego dnia —
        # 8538 transakcji RF-macro to bylo 67 dni decyzji. Przedzial ufnosci liczymy wiec
        # bootstrapem po DNIACH, nie po transakcjach.
        dz = pd.Series(wz._t.dt.date.values)
        uniq = dz.unique(); rng = np.random.default_rng(7)
        grp = {d: np.where(dz.values == d)[0] for d in uniq}
        ev_b, wr_b = [], []
        for _ in range(300):
            ii = np.concatenate([grp[d] for d in rng.choice(uniq, len(uniq))])
            ev_b.append((r[ii] - k[ii]).mean()); wr_b.append((r[ii] > 0).mean())
        out.append(dict(tp=tp, sl=sl, hz=hz, strona=st, zestaw=zestaw, model=model, tx_dzien=tx,
                        n=len(wz), wr=(r > 0).mean(), wr_geometrii=baza_wr, przewaga_wr=(r > 0).mean() - baza_wr,
                        ev_atr=(r - k).mean(), ev_losowe=baza_ev,
                        r_brutto=r.mean(), koszt_atr=k.mean(), n_dni=len(uniq),
                        ev_p05_dni=np.percentile(ev_b, 5), wr_p05_dni=np.percentile(wr_b, 5),
                        ev_013=r.mean() - k.mean() * 0.13 / KOSZT_CENY,
                        ev_005=r.mean() - k.mean() * 0.05 / KOSZT_CENY, cechy="|".join(cechy)))
    if ZAPIS_MODELI:   # finalisci nie moga przepasc
        import joblib
        os.makedirs(ZAPIS_MODELI, exist_ok=True)
        joblib.dump({"model": m, "cechy": cechy, "etykieta": (tp, sl, hz, st), "zestaw": zestaw,
                     "typ": model, "wyniki": out},
                    f"{ZAPIS_MODELI}/{model}_{tp}_{sl}_{hz}_{st}_{zestaw}.pkl")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zbior", required=True); ap.add_argument("--etap", default="A")
    ap.add_argument("--symbole", default=""); ap.add_argument("--wyniki", default="poszukiwania_A.csv")
    ap.add_argument("--z-etapu", default=""); ap.add_argument("--procesy", type=int, default=24)
    ap.add_argument("--geometrie", default=""); ap.add_argument("--horyzonty", default="")
    ap.add_argument("--przygotowany", default="", help="gotowy *_przygotowany.parquet (np. z innego przebiegu)")
    a = ap.parse_args()
    global GEOMETRIE, HORYZONTY
    if a.geometrie: GEOMETRIE = [tuple(map(float, g.split("/"))) for g in a.geometrie.split(",")]
    if a.horyzonty: HORYZONTY = [int(x) for x in a.horyzonty.split(",")]
    t0 = time.time()
    przyg = a.przygotowany or a.wyniki.replace(".csv", "_przygotowany.parquet")
    if os.path.exists(przyg) and not a.symbole:
        Z = pd.read_parquet(przyg); print("uzywam gotowego przygotowanego zbioru", flush=True)
    else:
        Z = przygotuj(a.zbior, [s for s in a.symbole.split(",") if s])
        Z.to_parquet(przyg, index=False)
    print(f"przygotowane: {len(Z)} wierszy, {Z.symbol.nunique()} symboli, {time.time()-t0:.0f}s", flush=True)
    os.environ["POSZ_CIECIE"] = str(Z["_t"].quantile(0.6))   # dziedzicza procesy 'spawn'
    print(f"podzial kalendarzowy: trening < {os.environ['POSZ_CIECIE']}, test od +{EMBARGO_DNI} dni", flush=True)
    wszystkie = [k for k in Z.columns if not k.startswith(("_", "r_", "label_", "trade_", "cel_"))
                 and k not in ("timestamp", "symbol", "close") and k not in MAKRO_USUNIETE
                 and np.issubdtype(Z[k].dtype, np.number) and Z[k].nunique() > 2]
    from sklearn.metrics import roc_auc_score
    zadania = []
    if a.etap == "A":
        s0, _ = podziel(Z)
        sel = s0.sample(min(400_000, len(s0)), random_state=2)
        for (tp, sl), hz in itertools.product(GEOMETRIE, HORYZONTY):
            for st in ("L", "S"):
                kol = f"r_{tp}_{sl}_{hz}_{st}"; s_ = sel[sel[kol].notna()]; y = (s_[kol] > 0).astype(int)
                auc = {f: abs(roc_auc_score(y, s_[f].fillna(s_[f].median())) - 0.5) for f in wszystkie if s_[f].notna().mean() > 0.5}
                ranking = [f for f, _ in sorted(auc.items(), key=lambda x: -x[1])]
                top15 = ranking[:15]
                # top15 bywa w polowie klonami (x_rsi_7/rsi/x_rsi_21/r_stoch_rsi...). Zestaw
                # "zroznicowane": po kolei z rankingu, ale pomijamy ceche skorelowana > 0.7
                # (Spearman) z ktoras juz wybrana — 15 roznych zrodel informacji.
                pr = s_.sample(min(50_000, len(s_)), random_state=3)[ranking[:60]].rank()
                kor = pr.corr().abs(); zroz = []
                for f in ranking[:60]:
                    if all(not (kor.loc[f, g] > 0.7) for g in zroz): zroz.append(f)
                    if len(zroz) == 15: break
                zestawy = {"top15": top15, "wszystkie": wszystkie, "zroznicowane": zroz}
                zestawy.update({k: [f for f in v if f in Z.columns] for k, v in RODZINY.items()})
                for zn, cz in zestawy.items():
                    if len(cz) >= 5: zadania.append((przyg, tp, sl, hz, st, zn, cz, "lgb"))
    else:
        A = pd.read_csv(a.z_etapu)
        best = A.sort_values("ev_atr", ascending=False).drop_duplicates(["tp", "sl", "hz", "strona", "zestaw"]).head(12)
        for _, r in best.iterrows():
            for m in ("lgb", "xgb", "cat", "rf", "et", "hgb"):
                zadania.append((przyg, r.tp, r.sl, int(r.hz), r.strona, r.zestaw, r.cechy.split("|"), m))
    # Wznowienie (2026-09-13): pod ma limit 125 GB w kontenerze (free pokazuje 1 TB hosta);
    # przy 24 procesach OOM zabijal prace. Juz policzone kombinacje pomijamy.
    wczesniej = []
    if os.path.exists(a.wyniki):
        P = pd.read_csv(a.wyniki); wczesniej = P.to_dict("records")
        zrob = set(zip(P.tp, P.sl, P.hz, P.strona, P.zestaw, P.model))
        zadania = [z for z in zadania if (z[1], z[2], z[3], z[4], z[5], z[7]) not in zrob]
        print(f"wznowienie: {len(zrob)} kombinacji juz policzonych", flush=True)
    print(f"eksperymentow: {len(zadania)}", flush=True)
    # FIX 2026-09-13: pierwszy przebieg na podzie — jeden proces padl (LightGBM/OpenMP po
    # fork()), co zepsulo cala pule: 344 z 360 zadan skonczylo sie BrokenProcessPool.
    # Teraz 'spawn' + po awarii puli nowa pula dla zadan, ktore nie zdazyly sie policzyc.
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool
    wyn, zostalo, zrobione, proby = list(wczesniej), list(enumerate(zadania)), 0, {}
    while zostalo:
        with ProcessPoolExecutor(a.procesy, mp_context=mp.get_context("spawn")) as ex:
            fut = {ex.submit(jeden, z): (i, z) for i, z in zostalo}
            nowe = []
            for f in as_completed(fut):
                i, z = fut[f]
                try:
                    wyn += f.result(); zrobione += 1
                except BrokenProcessPool:
                    proby[i] = proby.get(i, 0) + 1
                    if proby[i] <= 2: nowe.append((i, z))
                    else: print(f"  pomijam zadanie {i} po 3 awariach: {z[1:7]}", flush=True)
                except Exception as e:
                    print(f"  blad zadania {i} {z[1:6]}: {e}", flush=True)
                if zrobione % 20 == 0:
                    pd.DataFrame(wyn).to_csv(a.wyniki, index=False)
                    print(f"  {zrobione}/{len(zadania)} ({time.time()-t0:.0f}s)", flush=True)
        if nowe: print(f"  awaria puli — ponawiam {len(nowe)} zadan w nowej puli", flush=True)
        zostalo = nowe
    W = pd.DataFrame(wyn); W.to_csv(a.wyniki, index=False)
    print("\n=== NAJLEPSZE wg wartosci oczekiwanej po kosztach (ATR na transakcje) ===")
    kol = ["tp", "sl", "hz", "strona", "zestaw", "model", "tx_dzien", "n", "wr", "wr_geometrii", "przewaga_wr", "ev_atr", "ev_013", "ev_005", "n_dni", "ev_p05_dni", "wr_p05_dni"]
    print(W.sort_values("ev_atr", ascending=False)[kol].head(25).round(3).to_string(index=False))
    print("\n=== WR >= 70% — najlepsze wg przewagi nad geometria ===")
    print(W[W.wr >= 0.70].sort_values("przewaga_wr", ascending=False)[kol].head(15).round(3).to_string(index=False))
    for c_, nm in (("ev_013", "0.13% (taker, maly poslizg)"), ("ev_005", "0.05% (zlecenia z limitem)")):
        print(f"\n=== NAJLEPSZE przy kosztach {nm} ===")
        print(W.sort_values(c_, ascending=False)[kol].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

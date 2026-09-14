#!/usr/bin/env python3
"""Boostingi CAT / XGB / LGB osobno + "gwaranci" w cechach (2026-09-14).

Pytanie uzytkownika: ktore cechy sprawiaja, ze pozycja W OGOLE WCHODZI, i ktore sprawiaja,
ze ja DOWOZIMY. "Jezeli cecha nie grala, to pozycja nie wchodzila."

Dla kazdej etykiety (wieksze ruchy: 2.5/1.5, 4/1, 6/1.5 ATR; 24/48/72 h; L i S) i kazdego modelu:
1. przecechowanie od zera z calego magazynu (85 cech): model na wszystkich cechach (trening),
   waznosc = srednie |SHAP| na probce treningu; zachlannie po waznosci, pomijajac ceche
   skorelowana |rho| > 0.5 (Spearman, trening) z juz wybrana — do 15 cech;
2. model na wybranych cechach, ocena na TESCIE (podzial kalendarzowy + 7 dni, poszukiwania.podziel):
   top transakcje pod 10 i 15 tx/dzien (48 symboli), jedna pozycja na symbol przez horyzont,
   koszty 0.22%, WR obok WR losowych wejsc, bootstrap po dniach, kwartaly;
3. gwaranci — SHAP w przestrzeni marginesu (log-odds) dla KAZDEJ wybranej transakcji:
   prog = najnizszy margines wsrod wybranych; cecha f jest DECYDUJACA dla wejscia i, gdy
   margines_i - wklad_f,i < prog (bez niej pozycja by nie weszla).
   Per cecha: udzial wejsc, w ktorych byla decydujaca ("gwarant wejscia"), oraz WR / wynik
   transakcji z nia decydujaca vs bez ("gwarant dowiezienia").
"""
import argparse, os, re, sys, time, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True, help="*_przygotowany.parquet (zbudowany, jesli brak)")
ap.add_argument("--zbior", default="", help="surowy zbior ml_trainer (budowany, jesli brak) — potrzebny tylko bez --przygotowany")
ap.add_argument("--tylko", default="", help="jedna etykieta 'tp,sl,hz,strona' bez puli procesow (GitHub: 1 zadanie = 1 maszyna)")
ap.add_argument("--wyniki", required=True); ap.add_argument("--procesy", type=int, default=18)
ap.add_argument("--rho", type=float, default=0.5); ap.add_argument("--ile", type=int, default=15)
a = ap.parse_args()
PZ.GEOMETRIE = [(2.5, 1.5), (4.0, 1.0), (6.0, 1.5)]
PZ.HORYZONTY = [24, 48, 72]
WYNIK_KOL = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
NJ = int(os.environ.get("GWAR_NJ", "14"))
MODELE_DIR = a.wyniki.replace(".csv", "_modele")


def zbuduj(typ):
    if typ == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=NJ)
    if typ == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=50, n_jobs=NJ, verbosity=0)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, thread_count=NJ)


def shap(m, typ, X):
    """-> (wklady n x f, margines n) w przestrzeni log-odds, natywnie dla kazdego boostingu."""
    if typ == "lgb":
        c = m.predict(X, pred_contrib=True)
    elif typ == "xgb":
        import xgboost as xgb
        c = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    else:
        from catboost import Pool
        c = m.get_feature_importance(Pool(X), type="ShapValues")
    c = np.asarray(c)
    return c[:, :-1], c.sum(axis=1)          # ostatnia kolumna = wartosc bazowa


def zadanie(args):
    tp, sl, hz, st, cechy, kor = args
    kol = f"r_{tp}_{sl}_{hz}_{st}"
    Z = pd.read_parquet(a.przygotowany, columns=cechy + ["symbol", "_t", "_koszt_atr", kol])
    Z = Z[Z[kol].notna()]
    # Rezim rynku: BTC wyzej (1) / nizej (0) niz 7 dni temu, na zamknieciu swiecy decyzji (bez przyszlosci)
    b = pd.read_parquet(f"{PZ.ROOT}/data_warehouse/ohlcv/binance/1h/BTC.parquet")
    b["_t"] = pd.to_datetime(b.timestamp); b = b.drop_duplicates("_t").set_index("_t").sort_index()
    rez = (b.close / b.close.shift(168) - 1 > 0).astype(int)
    rez.index = rez.index.astype("datetime64[ns]")            # magazyn ma [ms], zbior [ns] — bez tego reindex = same NaN
    Z["_rez"] = rez.reindex(pd.DatetimeIndex(Z["_t"].values).astype("datetime64[ns]")).fillna(-1).astype(int).values
    if (Z["_rez"] == -1).mean() > 0.05:
        raise RuntimeError(f"rezim rynku: {(Z['_rez'] == -1).mean():.0%} wierszy bez dopasowania do BTC")
    for f in cechy:
        Z[f] = Z[f].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    sel, spr = PZ.podziel(Z)
    if len(sel) > 1_500_000:
        sel = sel.sample(1_500_000, random_state=1)
    if len(sel) > 1_000_000:
        sel = sel.sample(1_000_000, random_state=2)      # 27 vCPU na podzie — 1 mln wierszy wystarcza
    y = (sel[kol] > 0).astype(int); baza = (spr[kol] > 0).mean()
    # waga "zrownowazona": laczna waga wzrostow = waga spadkow (model nie moze nauczyc sie samego kierunku rynku)
    cz = sel["_rez"].value_counts()
    w_zr = sel["_rez"].map(lambda r: len(sel) / (2 * cz.get(r, 1)) if r in (0, 1) else 1.0).values
    baza_rez = {r: (spr.loc[spr._rez == r, kol] > 0).mean() for r in (0, 1)}
    prob = sel.sample(min(50_000, len(sel)), random_state=5)
    wyniki, gwaranci = [], []
    import joblib
    os.makedirs(MODELE_DIR, exist_ok=True)
    for typ in ("cat", "xgb", "lgb"):
        # 1. przecechowanie: model na wszystkich cechach -> waznosc SHAP -> zachlannie bez dubli
        m0 = zbuduj(typ); m0.fit(sel[cechy], y)
        waz = np.abs(shap(m0, typ, prob[cechy])[0]).mean(axis=0)
        wyb = []
        for j in np.argsort(-waz):
            if all(kor[cechy[j]][g] <= a.rho for g in wyb):
                wyb.append(cechy[j])
            if len(wyb) == a.ile:
                break
        # 2. model na wybranych cechach: zwykly i zrownowazony po rezimach rynku
        dni = (spr._t.max() - spr._t.min()).days; n_sym = spr.symbol.nunique()
        for wariant, wagi in ((typ, None), (typ + "_zr", w_zr)):
          m = zbuduj(typ); m.fit(sel[wyb], y, sample_weight=wagi)
          p = m.predict_proba(spr[wyb])[:, 1]
          joblib.dump({"model": m, "cechy": wyb, "etykieta": (tp, sl, hz, st), "typ": typ, "wariant": wariant},
                      f"{MODELE_DIR}/{wariant}_{tp}_{sl}_{hz}_{st}.pkl")
          for tx in (10, 15):
              wz = PZ.wybierz_transakcje(spr, p, hz, tx, dni, n_sym)
              if len(wz) < 50:
                  continue
              r = wz[kol].values; ev = r - wz._koszt_atr.values
              dz = wz._t.dt.date.values; uniq = np.unique(dz); rng = np.random.default_rng(7)
              grp = {d: np.where(dz == d)[0] for d in uniq}
              bs = [ev[np.concatenate([grp[d] for d in rng.choice(uniq, len(uniq))])].mean() for _ in range(300)]
              kw = pd.Series(ev).groupby(wz._t.dt.to_period("Q").values).mean()
              rz = {}
              for rr, nm in ((1, "wzrost"), (0, "spadek")):
                  mm = wz._rez.values == rr
                  rz.update({f"n_{nm}": int(mm.sum()), f"wr_{nm}": (r[mm] > 0).mean() if mm.any() else np.nan,
                             f"wr_geom_{nm}": baza_rez[rr], f"ev_{nm}": ev[mm].mean() if mm.any() else np.nan})
              wyniki.append(dict(tp=tp, sl=sl, hz=hz, strona=st, model=wariant, tx_dzien=tx, **rz, n=len(wz), n_dni=len(uniq),
                                 wr=(r > 0).mean(), wr_geometrii=baza, przewaga_wr=(r > 0).mean() - baza,
                                 ev=ev.mean(), ev_p05_dni=np.percentile(bs, 5), kw_plus=int((kw > 0).sum()),
                                 kw_n=len(kw), cechy="|".join(wyb)))
              if tx != 10:
                  continue
              # 3. gwaranci: SHAP dla kazdej wybranej transakcji
              wk, marg = shap(m, typ, wz[wyb])
              prog = marg.min()
              decyd = (marg[:, None] - wk) < prog                   # bez tej cechy pozycja by nie weszla
              wygr = r > 0
              for j, f in enumerate(wyb):
                  d = decyd[:, j]
                  gwaranci.append(dict(tp=tp, sl=sl, hz=hz, strona=st, model=wariant, cecha=f,
                                       udzial_decydujaca=d.mean(), n_decydujaca=int(d.sum()),
                                       wr_gdy_decydujaca=wygr[d].mean() if d.any() else np.nan,
                                       wr_gdy_nie=wygr[~d].mean() if (~d).any() else np.nan,
                                       ev_gdy_decydujaca=ev[d].mean() if d.any() else np.nan,
                                       ev_gdy_nie=ev[~d].mean() if (~d).any() else np.nan,
                                       sr_wklad=wk[:, j].mean(), wr_wszystkich=wygr.mean(), ev_wszystkich=ev.mean()))
    print(f"  {st} {tp}/{sl} {hz}h gotowe", flush=True)
    return wyniki, gwaranci


def main():
    t0 = time.time()
    if not os.path.exists(a.przygotowany):
        Zp = PZ.przygotuj(a.zbior, []); Zp.to_parquet(a.przygotowany, index=False); del Zp
        print(f"zbior przygotowany w {time.time() - t0:.0f} s", flush=True)
    import pyarrow.parquet as pq
    kolumny = [k for k in pq.read_schema(a.przygotowany).names if not WYNIK_KOL.match(k)
               and not k.startswith(("label_", "trade_", "cel_")) and k not in ("timestamp", "close")]
    Z = pd.read_parquet(a.przygotowany, columns=kolumny)
    # Ciecie na sztywno (POSZ_CIECIE_STALE), gdy dane koncza sie w innym dniu niz na innej maszynie —
    # inaczej 60. percentyl czasu przesuwa podzial i wyniki GH/poda nie sa porownywalne.
    os.environ["POSZ_CIECIE"] = os.environ.get("POSZ_CIECIE_STALE") or str(Z["_t"].quantile(0.6))
    sel, _ = PZ.podziel(Z)
    cechy = [k for k in Z.columns if not k.startswith("_") and k != "symbol" and k not in PZ.MAKRO_USUNIETE
             and pd.api.types.is_numeric_dtype(Z[k]) and Z[k].nunique() > 10 and sel[k].notna().mean() > 0.5]
    prob = sel[cechy].sample(min(200_000, len(sel)), random_state=3).replace([np.inf, -np.inf], np.nan)
    kor = prob.rank().corr().abs().fillna(0).to_dict()
    print(f"magazyn: {len(cechy)} cech, zbior {len(Z)} wierszy; ciecie {os.environ['POSZ_CIECIE']}", flush=True)
    del Z, sel, prob
    zad = [(tp, sl, hz, st, cechy, kor) for (tp, sl), hz, st in itertools.product(PZ.GEOMETRIE, PZ.HORYZONTY, "LS")]
    if a.tylko:
        t = a.tylko.split(","); cel = (float(t[0]), float(t[1]), int(t[2]), t[3])
        zad = [z for z in zad if z[:4] == cel]
        assert zad, f"nieznana etykieta {a.tylko}"
    import multiprocessing as mp
    W, G = [], []
    if a.tylko:
        w, g = zadanie(zad[0]); W += w; G += g
    else:
        with mp.get_context("spawn").Pool(a.procesy) as pool:
            for w, g in pool.imap_unordered(zadanie, zad):
                W += w; G += g
                pd.DataFrame(W).to_csv(a.wyniki, index=False)
                pd.DataFrame(G).to_csv(a.wyniki.replace(".csv", "_gwaranci.csv"), index=False)
    pd.DataFrame(W).to_csv(a.wyniki, index=False)
    pd.DataFrame(G).to_csv(a.wyniki.replace(".csv", "_gwaranci.csv"), index=False)
    if a.tylko:
        print(f"zadanie {a.tylko}: {len(W)} wynikow, {len(G)} wierszy gwarantow, {time.time() - t0:.0f} s")
        return
    W, G = pd.DataFrame(W), pd.DataFrame(G)
    k = ["tp", "sl", "hz", "strona", "model", "tx_dzien", "n", "wr", "wr_geometrii", "ev", "ev_p05_dni", "kw_plus", "kw_n",
         "wr_wzrost", "wr_geom_wzrost", "ev_wzrost", "wr_spadek", "wr_geom_spadek", "ev_spadek"]
    print("\n=== NAJLEPSZE wg wyniku po kosztach ===")
    print(W.sort_values("ev", ascending=False)[k].head(15).round(3).to_string(index=False))
    print("\n=== ODPORNE (dolna granica z dni > 0) ===")
    print(W[W.ev_p05_dni > 0].sort_values("ev_p05_dni", ascending=False)[k].head(15).round(3).to_string(index=False))
    print("\n=== srednio per model ===")
    print(W.groupby("model")[["ev", "przewaga_wr", "ev_p05_dni", "ev_wzrost", "ev_spadek"]].mean().round(3).to_string())
    print("\n=== SHORTY w rynku ROSNACYM (czy model wybiera coiny, a nie tylko kierunek rynku) ===")
    sh = W[(W.strona == "S") & (W.n_wzrost >= 30)].copy(); sh["przewaga_wzrost"] = sh.wr_wzrost - sh.wr_geom_wzrost
    print(sh.sort_values("ev_wzrost", ascending=False)[k].head(10).round(3).to_string(index=False))
    print("\n=== GWARANCI: cechy najczesciej decydujace o wejsciu (srednio po etykietach i modelach) ===")
    g = G.groupby("cecha").agg(ile_razy=("cecha", "size"), udzial=("udzial_decydujaca", "mean"),
                               wr_decyd=("wr_gdy_decydujaca", "mean"), wr_nie=("wr_gdy_nie", "mean"),
                               ev_decyd=("ev_gdy_decydujaca", "mean"), ev_nie=("ev_gdy_nie", "mean"))
    g["roznica_ev"] = g.ev_decyd - g.ev_nie
    print(g.sort_values("udzial", ascending=False).head(20).round(3).to_string())
    print("\n=== GWARANCI DOWIEZIENIA: gdy cecha decyduje, wynik lepszy niz bez niej (min. 6 wystapien) ===")
    print(g[g.ile_razy >= 6].sort_values("roznica_ev", ascending=False).head(15).round(3).to_string())
    print(f"\ncalosc {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

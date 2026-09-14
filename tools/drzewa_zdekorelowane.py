#!/usr/bin/env python3
"""Drzewa na ZDEKORELOWANYCH cechach: ensemble 20 cech + trio ET/RF/HGB po 10 roznych (2026-09-14).

Pomysl uzytkownika: zamiast modeli na tych samych, zdublowanych cechach (RSI 1h/4h/1d,
trend 1h/4h/1d...) — maksymalnie rozne cechy, a trzy drzewa kazde na innym garniturze.
Cele: wieksze ruchy (2.5/1.5, 4/1, 6/1.5 ATR; 24-72 h), bo tam siedziala przewaga regul.

Uczciwosc (jak tools/poszukiwania.py, z tych samych funkcji):
- podzial KALENDARZOWY wspolny dla symboli, 7 dni przerwy (poszukiwania.podziel);
- dekorelacja i ranking cech liczone TYLKO na treningu;
- wynik z prawdziwej sciezki ceny, koszty 0.22% ceny, jedna pozycja na symbol przez horyzont;
- WR zawsze obok WR losowych wejsc z ta sama geometria; bootstrap po DNIACH.

Dekorelacja: Spearman na probce treningu, klastrowanie hierarchiczne (1-|rho|, srednie
wiazanie), ciecie przy |rho| = 0.5; z kazdego klastra cecha o najwyzszej trafnosci
jednowymiarowej (srednie |AUC-0.5| po etykietach). 30 najlepszych reprezentantow:
ensemble = 20 pierwszych; trio = reprezentanci rozdani po kolei (0,3,6.. / 1,4,7.. / 2,5,8..),
zeby kazde drzewo dostalo tyle samo silnych cech.
"""
import argparse, os, re, sys, time, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--zbior", required=True); ap.add_argument("--przygotowany", required=True)
ap.add_argument("--wyniki", required=True); ap.add_argument("--procesy", type=int, default=6)
ap.add_argument("--rho", type=float, default=0.5, help="prog |rho| laczenia cech w klaster")
ap.add_argument("--tylko", default="", help="jedna etykieta 'tp,sl,hz,strona' bez puli (GitHub: 1 zadanie = 1 maszyna)")
a = ap.parse_args()
PZ.GEOMETRIE = [(2.5, 1.5), (4.0, 1.0), (6.0, 1.5)]
PZ.HORYZONTY = [24, 48, 72]
WYNIK_KOL = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
MODELE_DIR = a.wyniki.replace(".csv", "_modele")


def model(typ, nj=int(os.environ.get("DRZEWA_NJ", "14"))):
    if typ == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=200, max_features=0.6,
                                    n_jobs=nj, random_state=1)
    if typ == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=200, max_features=0.6,
                                      n_jobs=nj, random_state=1)
    if typ == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=300, random_state=1)
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=300,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=nj)


def ocen(spr, wynik, kol, hz, tx):
    dni = (spr._t.max() - spr._t.min()).days; n_sym = spr.symbol.nunique()
    wz = PZ.wybierz_transakcje(spr, wynik, hz, tx, dni, n_sym)
    if len(wz) < 50:
        return None
    r = wz[kol].values; k = wz._koszt_atr.values; ev = r - k
    dz = wz._t.dt.date.values; uniq = np.unique(dz); rng = np.random.default_rng(7)
    grp = {d: np.where(dz == d)[0] for d in uniq}
    b = [ev[np.concatenate([grp[d] for d in rng.choice(uniq, len(uniq))])].mean() for _ in range(300)]
    kw = pd.Series(ev).groupby(wz._t.dt.to_period("Q").values).mean()
    return dict(n=len(wz), n_dni=len(uniq), wr=(r > 0).mean(), ev=ev.mean(), ev_p05_dni=np.percentile(b, 5),
                kw_plus=int((kw > 0).sum()), kw_n=len(kw))


def zadanie(args):
    tp, sl, hz, st, zestawy = args
    kol = f"r_{tp}_{sl}_{hz}_{st}"
    wszystkie = sorted(set(itertools.chain.from_iterable(zestawy.values())))
    Z = pd.read_parquet(a.przygotowany, columns=wszystkie + ["symbol", "_t", "_koszt_atr", kol])
    Z = Z[Z[kol].notna()]
    sel, spr = PZ.podziel(Z)
    if len(sel) > 1_500_000:
        sel = sel.sample(1_500_000, random_state=1)
    y = (sel[kol] > 0).astype(int)
    baza = (spr[kol] > 0).mean()
    przew, out = {}, []
    import joblib
    os.makedirs(MODELE_DIR, exist_ok=True)
    for nazwa, (typ, cechy) in {"ET_10a": ("et", zestawy["trio_a"]), "RF_10b": ("rf", zestawy["trio_b"]),
                                "HGB_10c": ("hgb", zestawy["trio_c"]), "LGB_20": ("lgb", zestawy["ens20"])}.items():
        X = sel[cechy].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xs = spr[cechy].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        if typ in ("et", "rf"):
            X, Xs = X.fillna(0), Xs.fillna(0)
        m = model(typ); m.fit(X, y)
        przew[nazwa] = m.predict_proba(Xs)[:, 1]
        joblib.dump({"model": m, "cechy": cechy, "etykieta": (tp, sl, hz, st), "typ": typ},
                    f"{MODELE_DIR}/{nazwa}_{tp}_{sl}_{hz}_{st}.pkl")
    # ensemble trzech drzew: srednia RANG (modele maja rozne skale prawdopodobienstw)
    rangi = {k: pd.Series(v).rank(pct=True).values for k, v in przew.items()}
    przew["TRIO_srednia"] = (rangi["ET_10a"] + rangi["RF_10b"] + rangi["HGB_10c"]) / 3
    # glosowanie: ile drzew ma dany wiersz w swoim gornym 3% (potem sortujemy po sredniej randze)
    glosy = sum((rangi[k] >= 0.97).astype(int) for k in ("ET_10a", "RF_10b", "HGB_10c"))
    przew["TRIO_2z3"] = np.where(glosy >= 2, przew["TRIO_srednia"], 0.0)
    for nazwa, p in przew.items():
        for tx in (10, 15):
            o = ocen(spr, p, kol, hz, tx)
            if o:
                out.append(dict(tp=tp, sl=sl, hz=hz, strona=st, model=nazwa, tx_dzien=tx, wr_geometrii=baza,
                                przewaga_wr=o["wr"] - baza, **o))
    print(f"  {st} {tp}/{sl} {hz}h gotowe", flush=True)
    return out


def main():
    t0 = time.time()
    if os.path.exists(a.przygotowany):
        Z = pd.read_parquet(a.przygotowany)
    else:
        Z = PZ.przygotuj(a.zbior, []); Z.to_parquet(a.przygotowany, index=False)
    print(f"zbior: {len(Z)} wierszy, {Z.symbol.nunique()} symboli, {time.time() - t0:.0f} s", flush=True)
    # ciecie na sztywno (POSZ_CIECIE_STALE), gdy dane koncza sie w innym dniu niz na innej maszynie
    os.environ["POSZ_CIECIE"] = os.environ.get("POSZ_CIECIE_STALE") or str(Z["_t"].quantile(0.6))
    sel, _ = PZ.podziel(Z)
    cechy = [k for k in Z.columns if not WYNIK_KOL.match(k) and not k.startswith(("_", "label_", "trade_", "cel_"))
             and k not in ("timestamp", "symbol", "close") and k not in PZ.MAKRO_USUNIETE
             and pd.api.types.is_numeric_dtype(Z[k])
             and Z[k].nunique() > 10 and sel[k].notna().mean() > 0.5]
    prob = sel.sample(min(200_000, len(sel)), random_state=3)
    # trafnosc jednowymiarowa na TRENINGU: srednie |AUC-0.5| po etykietach duzych ruchow
    from sklearn.metrics import roc_auc_score
    etyk = [f"r_{tp}_{sl}_{hz}_{st}" for (tp, sl), hz, st in itertools.product(PZ.GEOMETRIE, PZ.HORYZONTY, "LS")]
    traf = {}
    for f in cechy:
        x = prob[f].replace([np.inf, -np.inf], np.nan); x = x.fillna(x.median()); w = []
        for e in etyk:
            m = prob[e].notna()
            if m.sum() > 1000:
                w.append(abs(roc_auc_score((prob.loc[m, e] > 0).astype(int), x[m]) - 0.5))
        traf[f] = float(np.mean(w)) if w else 0.0
    kor = prob[cechy].rank().corr().abs().fillna(0).values
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    odl = np.clip(1 - kor, 0, None); np.fill_diagonal(odl, 0)
    kl = fcluster(linkage(squareform(odl, checks=False), "average"), t=1 - a.rho, criterion="distance")
    rep = {}
    for f, k in zip(cechy, kl):
        if k not in rep or traf[f] > traf[rep[k]]:
            rep[k] = f
    reps = sorted(rep.values(), key=lambda f: -traf[f])[:30]
    ix = [cechy.index(f) for f in reps]; sub = kor[np.ix_(ix, ix)].copy(); np.fill_diagonal(sub, 0)
    zestawy = {"ens20": reps[:20], "trio_a": reps[0::3], "trio_b": reps[1::3], "trio_c": reps[2::3]}
    print(f"{len(cechy)} cech -> {len(set(kl))} klastrow przy |rho|>{a.rho}; wybrane 30, max |rho| miedzy nimi "
          f"{sub.max():.2f}, srednio {sub[np.triu_indices(len(ix), 1)].mean():.2f}", flush=True)
    for k, v in zestawy.items():
        print(f"  {k}: " + ", ".join(f"{f}({traf[f]:.3f})" for f in v), flush=True)
    json.dump({"zestawy": zestawy, "trafnosc": {f: traf[f] for f in reps}, "klastry": int(len(set(kl)))},
              open(a.wyniki.replace(".csv", "_cechy.json"), "w"), indent=1)
    del Z, sel, prob
    zad = [(tp, sl, hz, st, zestawy) for (tp, sl), hz, st in itertools.product(PZ.GEOMETRIE, PZ.HORYZONTY, "LS")]
    if a.tylko:
        t = a.tylko.split(","); cel = (float(t[0]), float(t[1]), int(t[2]), t[3])
        zad = [z for z in zad if z[:4] == cel]; assert zad, f"nieznana etykieta {a.tylko}"
        W = pd.DataFrame(zadanie(zad[0])); W.to_csv(a.wyniki, index=False)
        print(f"zadanie {a.tylko}: {len(W)} wynikow, {time.time() - t0:.0f} s"); return
    import multiprocessing as mp
    wyn = []
    with mp.get_context("spawn").Pool(a.procesy) as pool:
        for w in pool.imap_unordered(zadanie, zad):
            wyn += w; pd.DataFrame(wyn).to_csv(a.wyniki, index=False)
    W = pd.DataFrame(wyn); W.to_csv(a.wyniki, index=False)
    k = ["tp", "sl", "hz", "strona", "model", "tx_dzien", "n", "wr", "wr_geometrii", "przewaga_wr", "ev", "ev_p05_dni", "kw_plus", "kw_n"]
    print("\n=== NAJLEPSZE wg wyniku po kosztach (ATR/tx) ===")
    print(W.sort_values("ev", ascending=False)[k].head(20).round(3).to_string(index=False))
    print("\n=== ODPORNE: dolna granica z dni > 0 ===")
    print(W[W.ev_p05_dni > 0].sort_values("ev_p05_dni", ascending=False)[k].head(20).round(3).to_string(index=False))
    print("\n=== srednio per model ===")
    print(W.groupby("model")[["ev", "przewaga_wr", "ev_p05_dni"]].mean().round(3).sort_values("ev", ascending=False).to_string())
    print(f"\ncalosc {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

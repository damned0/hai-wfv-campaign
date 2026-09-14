#!/usr/bin/env python3
"""Bramki na skrajnosciach cech: wchodzimy TYLKO gdy cecha "gra" (2026-09-13).

Dla kazdej cechy, strony, geometrii (cel/stop w ATR) i horyzontu: jaka jest trafnosc
i wynik po kosztach transakcji otwieranych wylacznie wtedy, gdy cecha jest w swojej
skrajnosci (dolne/gorne 5% i 10%). Progi wyznaczane na STARSZYCH 60% danych, ocena
na nowszych 40%. Potem pary skrajnosci (warunek A I warunek B).

Kontrola pulapek:
- przewaga nad WR samej geometrii (losowe wejscia z tym samym celem/stopem)
- jedna pozycja na symbol przez horyzont (jak silnik), inaczej jedna swieca = 6 "transakcji"
- liczba DNI z sygnalem i dolna granica (5. percentyl) wyniku z bootstrapu po dniach —
  cechy dzienne (makro) daja serie w jednym dniu; RF-macro: 8538 transakcji = 67 dni
Wejscie: *_przygotowany.parquet z tools/poszukiwania.py (cechy + r_<cel>_<stop>_<hz>_<L|S>).
"""
import sys, os, argparse, itertools, numpy as np, pandas as pd

KOSZT_CENY = 0.22   # % ceny na cala transakcje (jak w poszukiwania.py)


def dedup(df, hz):
    """Jedna pozycja na symbol: po wejsciu nastepne mozliwe dopiero po hz godzinach."""
    if not len(df): return df
    d = df.sort_values(["symbol", "_t"])
    sy = d["symbol"].values; t = d["_t"].values.astype("datetime64[ns]").astype(np.int64)
    krok = np.int64(hz) * 3_600_000_000_000
    keep = np.zeros(len(d), bool); ost_s = None; wolne = np.int64(-2**62)
    for i in range(len(d)):
        if sy[i] != ost_s: ost_s = sy[i]; wolne = np.int64(-2**62)
        if t[i] >= wolne: keep[i] = True; wolne = t[i] + krok
    return d[keep]


def ocen(df, kol, hz, dni, n_sym, koszt_mnoz=1.0, boot=True):
    d = dedup(df, hz)
    if len(d) < 30: return None
    r = d[kol].values; k = d["_koszt_atr"].values * koszt_mnoz
    ev = r - k
    out = dict(n=len(d), tx_dzien=len(d) / dni * 48 / n_sym, wr=(r > 0).mean(), ev=ev.mean())
    dz = d["_t"].dt.date.values; uniq = np.unique(dz); out["n_dni"] = len(uniq)
    if boot and len(uniq) >= 10:
        rng = np.random.default_rng(3); grp = {u: np.where(dz == u)[0] for u in uniq}
        b = [ev[np.concatenate([grp[u] for u in rng.choice(uniq, len(uniq))])].mean() for _ in range(200)]
        out["ev_p05_dni"] = float(np.percentile(b, 5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zbior", required=True); ap.add_argument("--wyniki", default="ogony.csv")
    ap.add_argument("--geometrie", default="0.6/1.5,0.8/1.5,1.0/1.5,1.2/1.5,2.5/1.5,4.0/1.0")
    ap.add_argument("--horyzonty", default="3,6,12,24")
    ap.add_argument("--pary", type=int, default=25, help="ile najlepszych pojedynczych warunkow laczyc w pary")
    ap.add_argument("--procesy", type=int, default=8, help="pod: limit 125 GB w kontenerze — nie 24")
    a = ap.parse_args()
    Z = pd.read_parquet(a.zbior)
    from poszukiwania import MAKRO_USUNIETE
    cechy = [k for k in Z.columns if not k.startswith(("_", "r_", "label_", "trade_", "cel_"))
             and k not in ("timestamp", "symbol", "close") and k not in MAKRO_USUNIETE and pd.api.types.is_numeric_dtype(Z[k]) and Z[k].nunique() > 10]
    sel, spr = Z[Z._r <= 0.6], Z[Z._r > 0.6]
    dni_s = (sel._t.max() - sel._t.min()).days; dni_p = (spr._t.max() - spr._t.min()).days; ns = Z.symbol.nunique()
    print(f"{len(cechy)} cech | selekcja {dni_s} dni, sprawdzian {dni_p} dni, {ns} symboli", flush=True)
    global G
    G = dict(Z=Z, sel=sel, spr=spr, cechy=cechy, dni_p=dni_p, ns=ns, pary=a.pary,
             Q={f: sel[f].quantile([0.05, 0.10, 0.90, 0.95]).values for f in cechy})
    zad = [(g, hz, st) for g, hz in itertools.product(a.geometrie.split(","), map(int, a.horyzonty.split(",")))
           for st in ("L", "S")]
    import multiprocessing as mp
    wiersze = []
    with mp.get_context("fork").Pool(min(a.procesy, len(zad))) as pool:
        for w in pool.imap_unordered(licz_kombinacje, zad):
            wiersze += w
            pd.DataFrame(wiersze).to_csv(a.wyniki, index=False)
    W = pd.DataFrame(wiersze); W.to_csv(a.wyniki, index=False)
    raport(W)


G = {}


def licz_kombinacje(zadanie):
    g, hz, st = zadanie
    sel, spr, cechy, dni_p, ns = G["sel"], G["spr"], G["cechy"], G["dni_p"], G["ns"]
    tp, sl = g.split("/"); wiersze = []
    if True:
        if True:
            kol = f"r_{float(tp)}_{float(sl)}_{hz}_{st}"
            if kol not in sel.columns: return []
            baza_s = (sel[kol].dropna() > 0).mean(); baza_p = (spr[kol].dropna() > 0).mean()
            pojedyncze = []
            for f in cechy:
                q = G["Q"][f]
                for nazwa, maska_s, maska_p, war in (
                    ("<=q05", sel[f] <= q[0], spr[f] <= q[0], (f, "<=", q[0])),
                    ("<=q10", sel[f] <= q[1], spr[f] <= q[1], (f, "<=", q[1])),
                    (">=q90", sel[f] >= q[2], spr[f] >= q[2], (f, ">=", q[2])),
                    (">=q95", sel[f] >= q[3], spr[f] >= q[3], (f, ">=", q[3]))):
                    s_ = sel[maska_s & sel[kol].notna()]
                    if len(s_) < 300: continue
                    wr_s = (s_[kol] > 0).mean()
                    if wr_s - baza_s < 0.02: continue          # w selekcji musi byc choc lekka przewaga
                    ev_s = (s_[kol] - s_._koszt_atr).mean()
                    pojedyncze.append((ev_s, wr_s, war, maska_p))
            # sprawdzian dla najlepszych pojedynczych + pary
            pojedyncze.sort(key=lambda x: -x[0])
            top = pojedyncze[:G["pary"]]
            kandydaci = [((w,), m, ev_s, wr_s) for ev_s, wr_s, w, m in top]
            for (e1, w1s, c1, m1), (e2, w2s, c2, m2) in itertools.combinations(top, 2):
                if c1[0] == c2[0]: continue
                kandydaci.append(((c1, c2), m1 & m2, None, None))
            for war, maska_p, ev_s, wr_s in kandydaci:
                p_ = spr[maska_p & spr[kol].notna()]
                o = ocen(p_, kol, hz, dni_p, ns)
                if not o: continue
                wiersze.append(dict(tp=float(tp), sl=float(sl), hz=hz, strona=st,
                                    warunek=" I ".join(f"{c[0]} {c[1]} {c[2]:.4g}" for c in war),
                                    liczba_warunkow=len(war), wr_selekcja=wr_s, ev_selekcja=ev_s,
                                    wr_geometrii=baza_p, przewaga_wr=o["wr"] - baza_p, **o))
            print(f"  {tp}/{sl} {hz}h {st}: {len(pojedyncze)} warunkow z przewaga w selekcji", flush=True)
    return wiersze


def raport(W):
    k = ["tp", "sl", "hz", "strona", "warunek", "n", "tx_dzien", "n_dni", "wr", "wr_geometrii", "przewaga_wr", "ev", "ev_p05_dni"]
    ok = W[(W.n_dni >= 30)]
    print("\n=== NAJLEPSZE BRAMKI na sprawdzianie wg wyniku po kosztach (min. 30 dni z sygnalem) ===")
    print(ok.sort_values("ev", ascending=False)[k].head(25).round(3).to_string(index=False))
    print("\n=== WR >= 70% i wynik > 0, dolna granica z dni > 0 (najbardziej odporne) ===")
    print(ok[(ok.wr >= 0.70) & (ok.ev > 0) & (ok.ev_p05_dni > 0)].sort_values("ev_p05_dni", ascending=False)[k].head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

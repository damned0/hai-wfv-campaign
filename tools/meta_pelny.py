#!/usr/bin/env python3
"""Meta-etykieta z PELNYM retreningiem bazy w kazdym oknie (2026-09-16).

Po co jeszcze raz: meta-filtr byl juz mierzony dwa razy (tools/meta_filtr.py na hybrydzie 15.09 — 4/8 okien ponad
losowe; tools/meta_ranga1.py na wyborze rangi 1 LONG 16.09 — 1/6 okien ponad losowe). Oba razy meta uczyla sie na
prognozach bazy, ktora te dane WIDZIALA w treningu. Zarzut jest uzasadniony: model jest na swoim treningu zbyt
pewny, wiec rozklad prognoz, na ktorym uczy sie meta, nie jest tym, ktory zobaczy na tescie.

Tu kazde okno dostaje DWA treningi bazy (na typ modelu):
  BAZA-A: dane < S - embargo - dni_meta      -> prognozuje [S - embargo - dni_meta, S - embargo)  (poza proba!)
                                                to jest material do nauki META
  BAZA-B: dane < S - embargo                 -> prognozuje okno testowe [S, S + dni)
  META uczona na sygnalach BAZY-A, stosowana do sygnalow BAZY-B. Nigdy w tyl, nigdy na swoim treningu.

Sygnal: hybryda = srednia percentyli przyczynowych 6 modeli, potem wybor PRZEKROJOWY top-1 coina w godzinie
(tam siedzi alfa — patrz long_wyjscia.py). Wynik transakcji: wejscie po zamknieciu swiecy sygnalu, wyjscie po
--trzymanie h, koszt --koszt.
Cechy meta (bez rynkowych, inaczej meta zgaduje dzien): percentyle 6 modeli, ich odchylenie/min/maks/rozstep,
przewaga nad ranga 2, ATR%, godzina (sin/cos).
KONTROLA: losowy filtr przepuszczajacy tyle samo transakcji, 1000 losowan -> percentyl 95. Bez tego kazdy filtr
wyglada dobrze, bo rozklad zwrotow ma ciezki ogon.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

TOOLS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, TOOLS)

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True); ap.add_argument("--katalog", required=True)
ap.add_argument("--etykieta", default="1.5,1.5,6,L"); ap.add_argument("--okno", type=int, required=True)
ap.add_argument("--okna", type=int, default=12); ap.add_argument("--dni", type=int, default=45)
ap.add_argument("--embargo", type=int, default=7); ap.add_argument("--dni-meta", type=int, default=60)
ap.add_argument("--ile", type=int, default=10); ap.add_argument("--wiersze", type=int, default=600_000)
ap.add_argument("--trzymanie", type=int, default=24); ap.add_argument("--koszt", type=float, default=0.22)
ap.add_argument("--koniec", required=True)
A = ap.parse_args()


def wh_modul():
    """helpery z wfv_hybrydy (zbuduj/Xp/waznosc/przyczynowy_pct) — jedno zrodlo definicji modeli"""
    stary = sys.argv
    sys.argv = ["wfv_hybrydy.py", "--przygotowany", A.przygotowany, "--katalog", A.katalog,
                "--etykieta", A.etykieta, "--koniec", A.koniec, "--ile", str(A.ile), "--wiersze", str(A.wiersze)]
    try:
        import importlib, wfv_hybrydy
        importlib.reload(wfv_hybrydy)
        return wfv_hybrydy
    finally:
        sys.argv = stary


WH = wh_modul()
TYPY = WH.TYPY
KOL = WH.KOL
os.makedirs(A.katalog, exist_ok=True)

import poszukiwania as PZ
import pyarrow.parquet as pq
import re
wk = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
kol = pq.read_schema(A.przygotowany).names
CECHY = [k for k in kol if not wk.match(k) and not k.startswith(("_", "label_", "trade_", "cel_", "__"))
         and k not in ("timestamp", "symbol", "close") and k not in PZ.MAKRO_USUNIETE and k not in WH.RYNKOWE]
Z = pd.read_parquet(A.przygotowany, columns=CECHY + ["symbol", "_t", KOL])
Z = Z[~Z.symbol.isin(PZ.MARTWE_COINY)].sort_values("_t").reset_index(drop=True)
kor = Z[CECHY].sample(min(80_000, len(Z)), random_state=0).corr(method="spearman").abs().fillna(0)

koniec = pd.Timestamp(A.koniec)
START = koniec - pd.Timedelta(days=A.dni * (A.okna - A.okno + 1))
print(f"okno {A.okno}: test [{START.date()} .. {(START + pd.Timedelta(days=A.dni)).date()}), "
      f"wierszy {len(Z)}, cech {len(CECHY)}", flush=True)


def trenuj_i_prognozuj(do_treningu, od_prog, do_prog, tag):
    """6 modeli uczonych na _t < do_treningu; prognoza na [od_prog, do_prog) -> percentyle przyczynowe"""
    tr = Z[(Z._t < do_treningu) & Z[KOL].notna()]
    if len(tr) > A.wiersze:
        tr = tr.sample(A.wiersze, random_state=A.okno)
    if len(tr) < 20_000:
        return None
    y = (tr[KOL] > 0).astype(int)
    maska = (Z._t >= od_prog) & (Z._t < do_prog)
    P = Z.loc[maska, ["symbol", "_t"]].copy()
    s0 = tr.sample(min(250_000, len(tr)), random_state=2)
    for t in TYPY:
        m0 = WH.zbuduj(t); m0.fit(WH.Xp(s0, CECHY, t), (s0[KOL] > 0).astype(int))
        w = WH.waznosc(m0, t, WH.Xp(s0, CECHY, t), (s0[KOL] > 0).astype(int))
        wyb = []
        for j in np.argsort(-w):
            if all(kor[CECHY[j]][g] <= 0.5 for g in wyb):
                wyb.append(CECHY[j])
            if len(wyb) == A.ile:
                break
        m = WH.zbuduj(t); m.fit(WH.Xp(tr, wyb, t), y)
        p = m.predict_proba(WH.Xp(Z.loc[maska], wyb, t))[:, 1]
        P[t] = WH.przyczynowy_pct(P._t.values, p)
        print(f"  [{tag}] {t}: gotowe ({len(tr)} wierszy treningu)", flush=True)
    return P.dropna(subset=list(TYPY))


def sygnaly(P):
    """hybryda = srednia 6 percentyli; wybor przekrojowy top-1 w godzinie + cechy meta"""
    P = P.copy()
    P["hyb"] = P[list(TYPY)].mean(axis=1)
    P["m_std"] = P[list(TYPY)].std(axis=1); P["m_min"] = P[list(TYPY)].min(axis=1)
    P["m_max"] = P[list(TYPY)].max(axis=1); P["m_rozstep"] = P.m_max - P.m_min
    P["r"] = P.groupby("_t").hyb.rank(ascending=False, method="first")
    drugi = P[P.r == 2].set_index("_t").hyb
    S = P[P.r == 1].copy()
    S["przewaga_nad_2"] = S.hyb.values - S._t.map(drugi).values
    S["godz_sin"] = np.sin(2 * np.pi * S._t.dt.hour / 24); S["godz_cos"] = np.cos(2 * np.pi * S._t.dt.hour / 24)
    return S


def _ceny():
    """zamkniecia 1h z magazynu — przygotowany zbior NIE ma kolumny close (blad 1. przebiegu, 16.09)"""
    import glob as _g
    out = {}
    for p in _g.glob(os.path.join(os.environ.get("HAI_ROOT", "/root/ProjektHAI"),
                                  "data_warehouse/ohlcv/binance/1h/*.parquet")):
        s = os.path.basename(p)[:-8]
        o = pd.read_parquet(p, columns=["timestamp", "close"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None).astype("datetime64[ns]")
        o = o.drop_duplicates("timestamp").sort_values("timestamp")
        out[s] = o.set_index("timestamp").close
    return out


CENY = _ceny()
print(f"ceny z magazynu: {len(CENY)} symboli", flush=True)


def wynik(S):
    """zwrot transakcji: wejscie po zamknieciu swiecy sygnalu, wyjscie po --trzymanie h"""
    out = []
    for sym, g in S.groupby("symbol"):
        c = CENY.get(sym)
        if c is None: continue
        idx = c.index.get_indexer(pd.DatetimeIndex(g._t.values))
        v = c.values
        for t, i in zip(g._t.values, idx):
            if i < 0 or i + A.trzymanie >= len(v): continue
            out.append((t, sym, (v[i + A.trzymanie] / v[i] - 1) * 100 - A.koszt))
    W = pd.DataFrame(out, columns=["_t", "symbol", "netto"])
    if not len(W): return W.assign(**{c: [] for c in S.columns if c not in W.columns})
    return S.merge(W, on=["_t", "symbol"], how="inner")


t0 = time.time()
GRAN = START - pd.Timedelta(days=A.embargo)
PA = trenuj_i_prognozuj(GRAN - pd.Timedelta(days=A.dni_meta), GRAN - pd.Timedelta(days=A.dni_meta), GRAN, "BAZA-A")
PB = trenuj_i_prognozuj(GRAN, START, START + pd.Timedelta(days=A.dni), "BAZA-B")
if PA is None or PB is None:
    print("za malo danych treningu — okno pominiete"); sys.exit(0)
MA = wynik(sygnaly(PA)); MB = wynik(sygnaly(PB))
print(f"meta-material {len(MA)} transakcji, test {len(MB)} transakcji ({time.time() - t0:.0f} s)", flush=True)

CM = list(TYPY) + ["hyb", "m_std", "m_min", "m_max", "m_rozstep", "przewaga_nad_2", "godz_sin", "godz_cos"]
raport = {"okno": A.okno, "n_meta": len(MA), "n_test": len(MB), "baza_netto": float(MB.netto.mean()),
          "baza_wr": float((MB.netto > 0).mean())}
if len(MA) >= 200 and len(MB) >= 30:
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                       subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(MA[CM], (MA.netto > 0).astype(int))
    MB = MB.assign(p=m.predict_proba(MB[CM])[:, 1])
    rng = np.random.default_rng(0)
    for prog in (0.45, 0.50, 0.55):
        prz = MB[MB.p >= prog]
        if len(prz) < 10: continue
        los = np.array([MB.netto.values[rng.choice(len(MB), len(prz), replace=False)].mean() for _ in range(1000)])
        raport[f"prog_{prog}"] = {"n": len(prz), "netto": float(prz.netto.mean()),
                                  "wr": float((prz.netto > 0).mean()),
                                  "odrzucone": float(MB[MB.p < prog].netto.mean()) if (MB.p < prog).any() else None,
                                  "los95": float(np.percentile(los, 95)),
                                  "ponad_los": bool(prz.netto.mean() > np.percentile(los, 95))}
        print(f"  prog {prog}: n={len(prz)} netto {prz.netto.mean():+.4f} WR {(prz.netto > 0).mean():.3f} "
              f"| losowe95 {np.percentile(los, 95):+.4f} | ponad: {prz.netto.mean() > np.percentile(los, 95)}", flush=True)
json.dump(raport, open(f"{A.katalog}/meta_okno_{A.okno:02d}.json", "w"), indent=1)
MB.to_parquet(f"{A.katalog}/test_okno_{A.okno:02d}.parquet", index=False)
print(f"KONIEC okno {A.okno} ({time.time() - t0:.0f} s)")

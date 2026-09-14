#!/usr/bin/env python3
"""Test przecieku fundingu (2026-09-14).

Funding z Coinalyze (interval daily) ma znacznik POCZATKU doby i wartosc `c` z KONCA doby — ten sam blad,
ktory dla OI naprawiono 21.08 (+1 dzien), a dla fundingu nie. Do tego jednostki: Coinalyze w procentach
(0.01 = 0.01%), Binance fapi jako ulamek (0.0001) — w magazynie pomieszane miedzy coinami i w czasie.

Tu: odtwarzamy funding_rate / funding_change_24h dwoma sposobami z plikow magazynu:
  przeciek  — jak ml_trainer (ostatni wiersz <= godzina, jednostki jak w pliku),
  uczciwy   — wiersz dobowy widoczny dopiero od D+1 00:00, jednostki w ulamku (dobowe / 100).
Sprawdzamy (1) czy "przeciek" odtwarza wartosci zbioru przygotowanego (czy rozumiemy potok),
(2) regule funding-short (progi q05 z danych przed cieciem) na obu wersjach vs losowy short,
(3) korelacje cechy z wynikiem transakcji short.
"""
import argparse, os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser(); ap.add_argument("--przygotowany", required=True)
ap.add_argument("--wh", default=os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse"))
ap.add_argument("--ciecie", default="2024-11-07"); a = ap.parse_args()
RK = ["r_2.5_1.5_48_S", "r_2.5_1.5_24_S", "r_6.0_1.5_48_S"]


def szeregi(s):
    p = f"{a.wh}/derivatives/funding_rates/{s}.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p); d["timestamp"] = pd.to_datetime(d.timestamp).astype("datetime64[ns]")
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    krok = d.timestamp.diff().dt.total_seconds().div(3600).bfill()
    dob = (krok >= 20).values
    u = d.copy(); u.loc[dob, "timestamp"] = u.loc[dob, "timestamp"] + pd.Timedelta(days=1)
    u.loc[dob, "funding_rate"] = u.loc[dob, "funding_rate"] / 100.0
    u = u.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return d, u, dob.mean()


def probkuj(d, t):
    tt = d.timestamp.values; v = d.funding_rate.values
    i = np.searchsorted(tt, t, side="right") - 1; j = np.searchsorted(tt, t - np.timedelta64(24, "h"), side="right") - 1
    fr = np.where(i >= 0, v[np.clip(i, 0, None)], 0.0); f24 = np.where(j >= 0, v[np.clip(j, 0, None)], 0.0)
    return fr, fr - f24


Z = pd.read_parquet(a.przygotowany, columns=["symbol", "_t", "_koszt_atr", "funding_rate", "funding_change_24h"] + RK)
Z = Z[~Z.symbol.isin(PZ.MARTWE_COINY)].copy(); Z["_t"] = Z._t.astype("datetime64[ns]")
czesci, udz = [], {}
for s, g in Z.groupby("symbol"):
    r = szeregi(s)
    if r is None: continue
    d, u, udz[s] = r
    t = g._t.values
    g = g.copy()
    g["fr_p"], g["fch_p"] = probkuj(d, t)
    g["fr_u"], g["fch_u"] = probkuj(u, t)
    czesci.append(g)
Z = pd.concat(czesci)
print(f"{len(Z)} wierszy, {Z.symbol.nunique()} symboli; udzial wierszy dobowych (Coinalyze) w plikach: "
      f"srednio {np.mean(list(udz.values())):.0%}")
zg = np.isclose(Z.fr_p, Z.funding_rate, rtol=1e-6, atol=1e-12).mean()
zg2 = np.isclose(Z.fch_p, Z.funding_change_24h, rtol=1e-6, atol=1e-12).mean()
print(f"(1) odtworzenie potoku: funding_rate zgodny w {zg:.1%} wierszy, funding_change_24h w {zg2:.1%}")

C = pd.Timestamp(a.ciecie); TR, TE = Z[Z._t < C - pd.Timedelta(days=7)], Z[Z._t >= C]
print(f"\n(2) regula funding-short: progi q05 z treningu (< {C.date()} - 7 dni), test od {C.date()} "
      f"({TE._t.dt.normalize().nunique()} dni)")
for rk in RK:
    ev_los = (TE[rk] - TE._koszt_atr).mean(); wr_los = (TE[rk] > 0).mean()
    print(f"  {rk}: losowy short ev {ev_los:+.3f} ATR, WR {wr_los:.1%}")
    for w, fr, fch in (("przeciek", "fr_p", "fch_p"), ("uczciwy", "fr_u", "fch_u")):
        q1, q2 = TR[fch].quantile(0.05), TR[fr].quantile(0.05)
        m = (TE[fch] <= q1) & (TE[fr] <= q2); S = TE[m]
        ev = S[rk] - S._koszt_atr
        dni = ev.groupby(S._t.dt.normalize()).mean()
        bs = [dni.sample(len(dni), replace=True, random_state=k).mean() for k in range(300)] if len(dni) > 5 else [np.nan]
        # godziny doby: przeciek najsilniejszy wczesnie (wiecej przyszlosci w wartosci z konca doby)
        wcz = S[S._t.dt.hour < 8]; poz = S[S._t.dt.hour >= 16]
        print(f"    {w:9s} sygnalow {len(S):6d} ({len(S)/max(1,TE._t.dt.normalize().nunique()):.0f}/dzien) "
              f"WR {(S[rk] > 0).mean():.1%} ev {ev.mean():+.3f} (p05 {np.nanpercentile(bs, 5):+.3f}) | "
              f"godz. 0-7: ev {(wcz[rk] - wcz._koszt_atr).mean():+.3f} (n {len(wcz)}), 16-23: {(poz[rk] - poz._koszt_atr).mean():+.3f} (n {len(poz)})")

print("\n(3) Spearman cechy z wynikiem shorta r_2.5_1.5_48_S na tescie:")
for c in ("fr_p", "fr_u", "fch_p", "fch_u"):
    print(f"  {c}: {TE[c].corr(TE['r_2.5_1.5_48_S'], method='spearman'):+.4f}")

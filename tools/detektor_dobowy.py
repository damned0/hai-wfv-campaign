#!/usr/bin/env python3
"""Detektor przecieku z dobowym znacznikiem (2026-09-14).

Zrodlo dobowe ze znacznikiem 00:00 i wartoscia z konca doby niesie o 00:00 ~24 h przyszlosci, o 23:00 ~1 h.
Podpis przecieku: korelacja cechy z wynikiem transakcji WYRAZNIE silniejsza we wczesnych godzinach doby niz
w poznych. Cecha uczciwa ma podobna korelacje o kazdej porze. Liczone na tescie (od ciecia), dla shorta i longa.
"""
import argparse, os, sys, re, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser(); ap.add_argument("--przygotowany", required=True)
ap.add_argument("--ciecie", default="2024-11-07"); a = ap.parse_args()
kol = pq.read_schema(a.przygotowany).names
cechy = [k for k in kol if not re.match(r"^r_\d", k) and not k.startswith(("_", "label_", "__"))
         and k not in ("timestamp", "symbol") and k not in PZ.MAKRO_USUNIETE]
RK = ["r_2.5_1.5_48_S", "r_2.5_1.5_48_L"]
Z = pd.read_parquet(a.przygotowany, columns=["symbol", "_t"] + cechy + RK)
Z = Z[(~Z.symbol.isin(PZ.MARTWE_COINY)) & (Z._t >= pd.Timestamp(a.ciecie))]
h = Z._t.dt.hour.values; W, P = h < 6, h >= 18
Z = Z.sample(min(len(Z), 1_500_000), random_state=3) if len(Z) > 1_500_000 else Z
h = Z._t.dt.hour.values; W, P = h < 6, h >= 18
wyn = []
for c in cechy:
    x = pd.to_numeric(Z[c], errors="coerce")
    if x.nunique() < 3: continue
    r = {"cecha": c}
    for rk in RK:
        s = rk[-1]
        r[f"wcz_{s}"] = x[W].corr(Z[rk][W], method="spearman")
        r[f"poz_{s}"] = x[P].corr(Z[rk][P], method="spearman")
    r["roznica"] = max(abs(r["wcz_S"]) - abs(r["poz_S"]), abs(r["wcz_L"]) - abs(r["poz_L"]))
    wyn.append(r)
T = pd.DataFrame(wyn).sort_values("roznica", ascending=False)
pd.set_option("display.width", 200)
print(f"test od {a.ciecie}: {len(Z)} wierszy; godziny 0-5 (wcz) vs 18-23 (poz); Spearman z wynikiem 2.5/1.5 ATR 48 h")
print("PODEJRZANE (|wcz| - |poz| > 0.015):")
print(T[T.roznica > 0.015].round(4).to_string(index=False))
print("\nnajczystsze 10 na koncu listy dla porownania:")
print(T.tail(10).round(4).to_string(index=False))
T.to_csv(os.path.join(os.path.dirname(a.przygotowany), "detektor_dobowy.csv"), index=False)

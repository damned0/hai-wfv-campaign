#!/usr/bin/env python3
"""Naprawa cech fundingu w zbiorze przygotowanym (2026-09-14) — BADANIA, nie produkcja.

Przeciek (tools/test_fundingu.py): funding dobowy z Coinalyze ma znacznik 00:00 i wartosc z konca doby,
w procentach; Binance fapi (8 h) w ulamku. Naprawa:
  - wiersz dobowy widoczny dopiero od D+1 00:00 (jak OI od 21.08),
  - jednostki: dobowe / 100 -> ulamek, jak Binance,
  - przeliczenie 4 cech wzorami ml_trainer: funding_rate, funding_change_24h (fr - fr sprzed 24 h),
    e_rsi_x_funding = rsi * funding_rate, funding_x_oizscore = funding_rate * oi_zscore_30d.
Wynik: nowy plik (--wyjscie); oryginal nietkniety. Kontrola: detektor godzin doby na nowych kolumnach.
"""
import argparse, os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(); ap.add_argument("--przygotowany", required=True); ap.add_argument("--wyjscie", required=True)
ap.add_argument("--wh", default=os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")); a = ap.parse_args()
assert os.path.abspath(a.wyjscie) != os.path.abspath(a.przygotowany)


def uczciwy(s):
    p = f"{a.wh}/derivatives/funding_rates/{s}.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p); d["timestamp"] = pd.to_datetime(d.timestamp).astype("datetime64[ns]")
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    dob = (d.timestamp.diff().dt.total_seconds().div(3600).bfill() >= 20).values
    d.loc[dob, "timestamp"] = d.loc[dob, "timestamp"] + pd.Timedelta(days=1)
    d.loc[dob, "funding_rate"] = d.loc[dob, "funding_rate"] / 100.0
    return d.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


Z = pq.read_table(a.przygotowany).to_pandas()
Z = Z[[c for c in Z.columns if not c.startswith("__")]]
t_all = Z._t.astype("datetime64[ns]").values
fr = np.zeros(len(Z)); fch = np.zeros(len(Z)); brak = []
for s, idx in Z.groupby("symbol").indices.items():
    d = uczciwy(s)
    if d is None: brak.append(s); continue
    tt, v, t = d.timestamp.values, d.funding_rate.values, t_all[idx]
    i = np.searchsorted(tt, t, side="right") - 1; j = np.searchsorted(tt, t - np.timedelta64(24, "h"), side="right") - 1
    f0 = np.where(i >= 0, v[np.clip(i, 0, None)], 0.0); f24 = np.where(j >= 0, v[np.clip(j, 0, None)], 0.0)
    fr[idx] = f0; fch[idx] = f0 - f24
stare = Z[["funding_rate", "funding_change_24h"]].abs().median()
Z["funding_rate"] = fr; Z["funding_change_24h"] = fch
Z["e_rsi_x_funding"] = Z.rsi * fr
Z["funding_x_oizscore"] = fr * Z.oi_zscore_30d
Z.to_parquet(a.wyjscie, index=False)
print(f"naprawiono {len(Z)} wierszy, {Z.symbol.nunique()} symboli (bez pliku fundingu: {brak})")
print(f"mediana |funding_rate| przed {stare.funding_rate:.6f} -> po {np.median(np.abs(fr)):.6f}; "
      f"|funding_change_24h| {stare.funding_change_24h:.6f} -> {np.median(np.abs(fch)):.6f}")
T = Z[Z._t >= pd.Timestamp("2024-11-07")]; h = T._t.dt.hour.values
for c in ("funding_rate", "funding_change_24h", "e_rsi_x_funding", "funding_x_oizscore"):
    w = T[c][h < 6].corr(T["r_2.5_1.5_48_S"][h < 6], method="spearman")
    p = T[c][h >= 18].corr(T["r_2.5_1.5_48_S"][h >= 18], method="spearman")
    print(f"  kontrola {c:20s}: godz. 0-5 {w:+.4f}  18-23 {p:+.4f}  (przed naprawa ~ -0.10 vs 0.00)")

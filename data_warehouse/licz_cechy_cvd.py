#!/usr/bin/env python3
"""Cechy z order flow (CVD) — znormalizowane per symbol.

PO CO
-----
Magazyn ma orderflow/binance/*.parquet z of_cvd, of_delta, of_delta_pct —
dane ZYWE, godzinowe, 128 symboli. Ale ml_trainer NIGDY po nie nie siegal:
`cvd_x_adx` jest przypisane ZEREM na sztywno (ml_trainer.py:1176), a
`of_cvd_chg_24h` w ogole nie istnieje w cache. Flaga HAI_LEJEK_CVD dodaje
obie do list cech modeli — czyli mozna ja wlaczyc i dostac model karmiony
stalym zerem oraz nieistniejaca kolumna.

DLACZEGO NORMALIZACJA PER SYMBOL (kluczowe)
-------------------------------------------
`of_cvd` to wartosc SKUMULOWANA w jednostkach bezwzglednych — CVD BTC liczy
sie w miliardach, altcoina w tysiacach. Zmierzone 2026-08-24:

    of_cvd_chg_24h surowe:  BTC +0.527  |  cale uniwersum +0.017
    cvd_z24 (z-score/symbol):           |  cale uniwersum +0.351

Czyli surowa roznica dziala tylko przy pomiarze na JEDNYM symbolu. Zestawienie
104 symboli miesza nieporownywalne skale i sygnal znika. Po normalizacji wraca
— dwudziestokrotnie.

To jest pulapka pomiarowa warta zapamietania: test na pojedynczym symbolu
pokazal sile 0.527, a prawdziwa wartosc to 0.351.

CECHY
-----
  cvd_z6 / cvd_z24    zmiana CVD w oknie 6h/24h, z-score rolling 200h per symbol
  delta_pct_ma6       srednia of_delta_pct z 6h — juz procentowe, wiec
                      przenosi sie miedzy symbolami bez normalizacji (0.295)

ODRZUCONE: cvd_rel6 (CVD/srednia |CVD|) — 0.070. Skumulowany CVD przechodzi
przez zero, wiec dzielenie jest niestabilne.

BEZ PRZECIEKU: wszystkie okna wsteczne (diff/rolling na zamknietych swiecach).

Uruchomienie:
    python3 data_warehouse/licz_cechy_cvd.py
    python3 data_warehouse/licz_cechy_cvd.py --podglad --symbol BTC
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

OF_DIR = Path("/root/ProjektHAI/data_warehouse/orderflow/binance")
OUT = Path("/root/ProjektHAI/data_warehouse/cechy_cvd")
OKNO_Z = 200


def policz(of: pd.DataFrame) -> pd.DataFrame:
    of = of.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    d6, d24 = of["of_cvd"].diff(6), of["of_cvd"].diff(24)

    def zscore(x):
        m = x.rolling(OKNO_Z, min_periods=50).mean()
        s = x.rolling(OKNO_Z, min_periods=50).std().replace(0, np.nan)
        return (x - m) / s

    out = pd.DataFrame({
        "timestamp": of["timestamp"],
        "cvd_z6": zscore(d6),
        "cvd_z24": zscore(d24),
        "delta_pct_ma6": of["of_delta_pct"].rolling(6, min_periods=3).mean(),
    })
    # float32 + zaokraglenie: z-score nie potrzebuje 15 cyfr znaczacych, a
    # float64 dawal 103 MB na 128 symboli — parquet nie kompresuje losowych
    # mantys. Te pliki jada do repo GH przy kazdej kampanii, wiec rozmiar liczy sie
    # podwojnie. Po zmianie ~4x mniej, wartosci identyczne co do 4 miejsca.
    out = out.replace([np.inf, -np.inf], np.nan)
    for k in ("cvd_z6", "cvd_z24", "delta_pct_ma6"):
        out[k] = out[k].round(4).astype("float32")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--podglad", action="store_true")
    ap.add_argument("--symbol")
    a = ap.parse_args()

    pliki = sorted(OF_DIR.glob("*.parquet"))
    if a.symbol:
        pliki = [p for p in pliki if p.stem == a.symbol]
    OUT.mkdir(parents=True, exist_ok=True)
    ok = pominiete = 0
    for p in pliki:
        try:
            of = pd.read_parquet(p)
        except Exception:
            pominiete += 1
            continue
        if len(of) < 300 or "of_cvd" not in of.columns:
            pominiete += 1
            continue
        of["timestamp"] = pd.to_datetime(of["timestamp"])
        d = policz(of)
        if a.podglad:
            print(f"=== {p.stem} ({len(d):,} wierszy) ===")
            print(d.tail(4).to_string(index=False))
            for k in d.columns:
                if k == "timestamp":
                    continue
                print(f"  {k:16} pokrycie {d[k].notna().mean()*100:5.1f}%  "
                      f"sr {d[k].mean():+.4f}  std {d[k].std():.4f}")
            return
        d.to_parquet(OUT / f"{p.stem}.parquet", compression="snappy", index=False)
        ok += 1
    print(f"  zapisano {ok} symboli do {OUT} (pominieto {pominiete})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cechy sprzezenia alta z BTC — korelacja, beta, sila relatywna, dekorelacja.

PO CO
-----
Magazyn ma cechy opisujace SAMEGO BTC (btc_trend_1h/4h/1d, btc_rsi_4h,
btc_dominance_chg, x_btc_leadlag), ale ANI JEDNEJ opisujacej ZWIAZEK danego
alta z BTC. To realna luka: model widzi, ze BTC rosnie, i widzi, ze alt rosnie,
ale nie wie, czy alt idzie ZA BTC, czy WLASNA droga.

Wazne rozroznienie wzgledem kampanii 2026-08-18/19: tamta sprawdzila 54
NIEUZYWANE cechy magazynu na 5 rdzeniach i nie dala przewagi — ale to byla
eksploracja WYCZERPANEGO zbioru. Te cechy sa NOWE, liczone od zera z danych,
ktorych nikt dotad tak nie zestawil.

CECHY (liczone per symbol, na swiecach 1h)
------------------------------------------
  btc_corr_24h        korelacja zwrotow alt vs BTC, okno 24h
  btc_corr_72h        to samo, okno 72h — wolniejsze tlo
  btc_corr_change     corr_24h - corr_72h; dodatnie = sprzezenie ROSNIE
  btc_beta_72h        beta: cov(alt,btc)/var(btc) — jak mocno alt reaguje
  rel_strength_btc    zwrot alta 24h minus zwrot BTC 24h (punkty procentowe)
  corr_breakdown      1 gdy korelacja spadla ponizej 0.3 ORAZ alt idzie w
                      przeciwna strone niz BTC — krotka dekorelacja czesto
                      poprzedza wiekszy ruch coin-specific

BEZ PRZECIEKU: wszystkie okna sa WSTECZNE (rolling po zamknietych swiecach),
zaden nie siega do przyszlosci. Zwroty liczone jako pct_change() na close,
czyli wartosc w wierszu t opisuje ruch, ktory ZAKONCZYL sie w t.

Uruchomienie:
    python3 data_warehouse/licz_korelacje_btc.py            # wszystkie symbole
    python3 data_warehouse/licz_korelacje_btc.py --symbol FIL --podglad
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WH = Path("/root/ProjektHAI/data_warehouse/ohlcv/binance/1h")
OUT = Path("/root/ProjektHAI/data_warehouse/korelacje_btc")

PROG_DEKORELACJI = 0.30


def wczytaj(sym):
    p = WH / f"{sym}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["timestamp", "close"])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def policz(sym, btc):
    d = wczytaj(sym)
    if d is None or len(d) < 200:
        return None
    m = d.merge(btc, on="timestamp", how="inner", suffixes=("", "_btc"))
    if len(m) < 200:
        return None

    r = m["close"].pct_change()
    rb = m["close_btc"].pct_change()

    out = pd.DataFrame({"timestamp": m["timestamp"]})
    out["btc_corr_24h"] = r.rolling(24, min_periods=12).corr(rb)
    out["btc_corr_72h"] = r.rolling(72, min_periods=36).corr(rb)
    out["btc_corr_change"] = out["btc_corr_24h"] - out["btc_corr_72h"]

    # beta = cov(alt,btc)/var(btc); var=0 (BTC bez ruchu) -> beta niezdefiniowana
    kow = r.rolling(72, min_periods=36).cov(rb)
    war = rb.rolling(72, min_periods=36).var()
    out["btc_beta_72h"] = np.where(war > 1e-12, kow / war, np.nan)

    # sila relatywna: o ile alt pobil BTC przez ostatnie 24h, w punktach proc.
    out["rel_strength_btc"] = (m["close"].pct_change(24) - m["close_btc"].pct_change(24)) * 100

    # dekorelacja: slabe sprzezenie ORAZ przeciwne kierunki 24h
    kier_alt = np.sign(m["close"].pct_change(24))
    kier_btc = np.sign(m["close_btc"].pct_change(24))
    out["corr_breakdown"] = (
        (out["btc_corr_24h"] < PROG_DEKORELACJI) & (kier_alt != kier_btc) & (kier_alt != 0)
    ).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--podglad", action="store_true", help="nie zapisuj, pokaz probke")
    a = ap.parse_args()

    btc = wczytaj("BTC")
    if btc is None:
        raise SystemExit("brak BTC.parquet — bez niego nie da sie liczyc korelacji")
    btc = btc.rename(columns={"close": "close_btc"})

    symbole = [a.symbol] if a.symbol else sorted(
        p.stem for p in WH.glob("*.parquet") if p.stem != "BTC")

    OUT.mkdir(parents=True, exist_ok=True)
    ok = pominiete = 0
    for s in symbole:
        w = policz(s, btc)
        if w is None:
            pominiete += 1
            continue
        if a.podglad:
            print(f"\n=== {s} ({len(w)} wierszy) ===")
            print(w.tail(5).to_string(index=False))
            kryte = {c: f"{w[c].notna().mean()*100:.0f}%" for c in w.columns if c != "timestamp"}
            print(f"  pokrycie (nie-NaN): {kryte}")
            print(f"  dekorelacja aktywna w {w['corr_breakdown'].mean()*100:.1f}% swiec")
        else:
            w.to_parquet(OUT / f"{s}.parquet", compression="snappy", index=False)
        ok += 1

    if not a.podglad:
        print(f"  zapisano {ok} symboli do {OUT} (pominieto {pominiete})")


if __name__ == "__main__":
    main()

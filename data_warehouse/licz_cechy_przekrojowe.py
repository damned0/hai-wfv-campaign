#!/usr/bin/env python3
"""Cechy zdarzeniowe i przekrojowe — alternatywa dla rolling korelacji z BTC.

PO CO
-----
Kampania 2026-08-22 pokazala, ze rolling korelacja z BTC (btc_corr_24h itd.)
NIE wnosi wartosci w zadnej z trzech rol: jako cecha modelu, jako detektor
rezimu ani jako modyfikator rozmiaru pozycji. Diagnoza z raporty/korebtc.txt
jest trafna: korelacja jest OPOZNIONA i SZUMNA, a modele i tak widza BTC przez
btc_trend_1h/4h/1d i btc_rsi_4h.

Ten skrypt liczy cechy INNEJ KLASY — zdarzenia i pozycje w przekroju rynku,
a nie usrednione statystyki.

CECHY
-----
  resid_ret_4h / resid_ret_24h
      Zwrot coina PO ODJECIU tego, co tlumaczy ruch BTC:
          r_coin - beta * r_btc
      To momentum "wlasne" coina. Beta liczona rollingiem 72h. Rozni sie od
      rel_strength_btc (proste r_coin - r_btc) tym, ze uwzglednia, JAK MOCNO
      dany coin zwykle reaguje na BTC — alt o becie 2.0 rosnacy tyle co BTC
      faktycznie jest SLABY, a rel_strength tego nie widzi.

  rank_mom_24h
      Percentyl momentum 24h na tle CALEGO uniwersum w tej samej godzinie.
      Modele widza wartosc bezwzgledna zwrotu, ale nie wiedza, czy +3% to
      duzo czy malo NA TLE RYNKU tego dnia. To jest informacja, ktorej
      dotad nie mialy w zadnej postaci.

  rank_vol_chg
      Percentyl zmiany wolumenu na tle uniwersum — czy przeplyw idzie
      wlasnie tutaj, czy wszedzie po rowno.

  event_dekorelacji
      ZDARZENIE, nie poziom: coin rosnie (>1%) w oknie 6h, podczas gdy BTC
      stoi lub spada (<=0), a wolumen coina jest powyzej mediany uniwersum.
      Czyli "oderwal sie TERAZ i ma wlasny przeplyw" — w odroznieniu od
      "srednio jest slabo skorelowany", ktore okazalo sie bezuzyteczne.

BEZ PRZECIEKU: wszystkie okna wsteczne (rolling/pct_change na zamknietych
swiecach), percentyle liczone W OBREBIE tej samej godziny — zaden nie siega
do przyszlosci. To istotne po trzech przeciekach znalezionych 15-21.08.

Uruchomienie:
    python3 data_warehouse/licz_cechy_przekrojowe.py
    python3 data_warehouse/licz_cechy_przekrojowe.py --podglad --symbol FIL
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WH = Path("/root/ProjektHAI/data_warehouse/ohlcv/binance/1h")
OUT = Path("/root/ProjektHAI/data_warehouse/cechy_przekrojowe")
OKNO_BETA = 72


def wczytaj_wszystko():
    """Wspolna ramka: timestamp x symbol -> close, volume. Potrzebna do percentyli."""
    zamk, wol = {}, {}
    for p in sorted(WH.glob("*.parquet")):
        try:
            d = pd.read_parquet(p, columns=["timestamp", "close", "volume"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.dropna().drop_duplicates("timestamp").set_index("timestamp").sort_index()
        zamk[p.stem] = d["close"]
        wol[p.stem] = d["volume"]
    return pd.DataFrame(zamk), pd.DataFrame(wol)


def policz(C, V):
    """C, V: DataFrame timestamp x symbol. Zwraca dict symbol -> DataFrame cech."""
    R = C.pct_change()
    if "BTC" not in C.columns:
        raise SystemExit("brak BTC w magazynie — bez niego nie ma odniesienia")
    rb = R["BTC"]

    # beta kazdego coina do BTC, rolling 72h.
    # WYDAJNOSC: DataFrame.rolling().cov(Series) liczy pary kolumna-po-kolumnie
    # i na 106 symbolach x 46k godzin nie konczy sie w rozsadnym czasie
    # (zmierzone: przerwane po 900s). Rozpisujemy kowariancje jawnie:
    #   cov(r,b) = E[r*b] - E[r]*E[b]
    # co sprowadza sie do trzech wektorowych rolling().mean() na calej ramce.
    m_r = R.rolling(OKNO_BETA, min_periods=36).mean()
    m_b = rb.rolling(OKNO_BETA, min_periods=36).mean()
    m_rb = R.mul(rb, axis=0).rolling(OKNO_BETA, min_periods=36).mean()
    kow = m_rb.sub(m_r.mul(m_b, axis=0))
    war = rb.rolling(OKNO_BETA, min_periods=36).var()
    beta = kow.div(war.replace(0, np.nan), axis=0)

    mom24 = C.pct_change(24) * 100
    mom4 = C.pct_change(4) * 100
    mom6 = C.pct_change(6) * 100
    vchg = V.pct_change(24).replace([np.inf, -np.inf], np.nan)

    # PERCENTYLE w obrebie tej samej godziny (axis=1) — pozycja na tle rynku
    rank_mom = mom24.rank(axis=1, pct=True)
    rank_vol = vchg.rank(axis=1, pct=True)

    # residualne momentum: ile ruchu NIE tlumaczy BTC
    btc4, btc24 = C["BTC"].pct_change(4) * 100, C["BTC"].pct_change(24) * 100
    resid4 = mom4.sub(beta.mul(btc4, axis=0))
    resid24 = mom24.sub(beta.mul(btc24, axis=0))

    # zdarzenie dekorelacji: coin rosnie, BTC nie, wolumen powyzej mediany rynku.
    # UWAGA: `DataFrame & Series` pandas probuje wyrownac po KOLUMNACH ramki
    # (symbole) wzgledem indeksu serii (znaczniki czasu) — daje to zly wynik
    # i zabija wydajnosc. Rozgłaszamy jawnie przez numpy (n,1).
    btc6 = (C["BTC"].pct_change(6) * 100).to_numpy()[:, None]
    event = pd.DataFrame(
        (mom6.to_numpy() > 1.0) & (btc6 <= 0.0) & (rank_vol.to_numpy() >= 0.5),
        index=C.index, columns=C.columns).astype(float)

    out = {}
    for s in C.columns:
        if s == "BTC":
            continue
        d = pd.DataFrame({
            "timestamp": C.index,
            "resid_ret_4h": resid4[s].values,
            "resid_ret_24h": resid24[s].values,
            "rank_mom_24h": rank_mom[s].values,
            "rank_vol_chg": rank_vol[s].values,
            "event_dekorelacji": event[s].values,
        }).replace([np.inf, -np.inf], np.nan)
        out[s] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--podglad", action="store_true")
    ap.add_argument("--symbol")
    a = ap.parse_args()

    C, V = wczytaj_wszystko()
    print(f"  uniwersum: {C.shape[1]} symboli x {C.shape[0]:,} godzin")
    wynik = policz(C, V)

    if a.podglad:
        s = a.symbol or next(iter(wynik))
        d = wynik[s]
        print(f"\n=== {s} ===")
        print(d.tail(4).to_string(index=False))
        for k in d.columns:
            if k == "timestamp":
                continue
            print(f"  {k:20} pokrycie {d[k].notna().mean()*100:5.1f}%  "
                  f"sr {d[k].mean():+.4f}  min {d[k].min():+.3f}  max {d[k].max():+.3f}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for s, d in wynik.items():
        d.to_parquet(OUT / f"{s}.parquet", compression="snappy", index=False)
    print(f"  zapisano {len(wynik)} symboli do {OUT}")


if __name__ == "__main__":
    main()

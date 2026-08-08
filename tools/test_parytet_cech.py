#!/usr/bin/env python3
"""Test parytetu cech: TRENING vs BACKTEST vs LIVE.

DLACZEGO ISTNIEJE
-----------------
Przez cala sesje 2026-08-05..08 lapalismy ciagle ten sam blad w osmiu roznych
miejscach: brak danych zamienial sie po cichu w wartosc domyslna, a przebieg
wygladal zdrowo do konca.

  feat_src.get(f, np.zeros(n))    # backtester -> 0.0
  kwargs.get('highs_1h')          # engine     -> None
  [f for f in _fl if f in df]     # feature_mix -> wyrzuca kolumne
  "funding_rate": np.zeros(n)     # backtester -> zero na sztywno

Skutki: 20h kampanii na zlych cechach, 99 pozycji otwartych na wyzerowanym
wejsciu, caly nurt rptr/SMC "zmierzony" jako szum, choc nigdy nie zostal
policzony. Zaden z tych bledow nie rzucil wyjatku.

CO ROBI
-------
Bierze jedna swieca i liczy wektor cech TRZEMA sciezkami, ktore w produkcji
musza dawac to samo:

  1. TRENING   ml_trainer.build_features_for_symbol   (dataset -> modele)
  2. BACKTEST  Backtester._build_feat_src             (symulacja WFV)
  3. LIVE      features.build_features_live           (handel na zywo)

i zglasza kazda cecha, ktora:
  - istnieje w jednej sciezce, a w innej nie (BRAK),
  - jest zerem w jednej, a liczba w innej (ZERO — najgrozniejszy przypadek,
    bo skaler robi z zera skrajna wartosc OOD i model dostaje smiec),
  - rozni sie wartoscia ponad tolerancje (ROZJAZD).

Kod wyjscia != 0 gdy cokolwiek znalazl — nadaje sie do preflightu kampanii.

UZYCIE
------
    python3 tools/test_parytet_cech.py [--symbol AVAX] [--swiec 400]
    python3 tools/test_parytet_cech.py --tylko-krytyczne   # tylko ZERO/BRAK
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hai_common"))

import numpy as np
import pandas as pd

TOLERANCJA = 1e-4      # wzgledna; formuly moga sie roznic o blad numeryczny
BLISKO_ZERA = 1e-12


class _Shim:
    """Minimalny odpowiednik strategii dla build_features_live.
    features.py wola: calculate_rsi, calculate_ema, calculate_atr(ceny, n),
    calculate_roc, detect_trend."""

    def calculate_rsi(self, p, n=14):
        s = pd.Series(p, dtype=float); d = s.diff()
        au = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        ad = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return float(100 - 100/(1 + au.iloc[-1]/ad.iloc[-1])) if ad.iloc[-1] else 100.0

    def calculate_ema(self, p, n):
        return float(pd.Series(p, dtype=float).ewm(span=n, adjust=False).mean().iloc[-1])

    def calculate_atr(self, prices, n=14):
        return float(pd.Series(prices, dtype=float).diff().abs().rolling(n).mean().iloc[-1])

    def calculate_roc(self, p, n=10):
        p = list(map(float, p))
        return float((p[-1]/p[-1-n] - 1) * 100) if len(p) > n and p[-1-n] else 0.0

    def detect_trend(self, p):
        e9, e21 = self.calculate_ema(p, 9), self.calculate_ema(p, 21)
        return 1 if e9 > e21 else (-1 if e9 < e21 else 0)


def wczytaj(symbol, swiec):
    wh = ROOT / "data_warehouse" / "ohlcv" / "binance"
    d1h = pd.read_parquet(wh / "1h" / f"{symbol}.parquet").tail(swiec).reset_index(drop=True)
    d4h = pd.read_parquet(wh / "4h" / f"{symbol}.parquet").tail(swiec // 4 + 60).reset_index(drop=True)
    d1d = pd.read_parquet(wh / "1d" / f"{symbol}.parquet").tail(swiec // 24 + 60).reset_index(drop=True)
    for d in (d1h, d4h, d1d):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    return d1h, d4h, d1d


def sciezka_trening(d1h, d4h, d1d, symbol):
    """UWAGA: makro / fear&greed / btc_context / ls_ratio NIE sa liczone z OHLCV —
    wchodza przez load_symbol_data(). Podanie samych swiec dawaloby dla nich zera
    i test krzyczalby na wlasny artefakt zamiast na prawdziwy blad (zlapane przy
    pierwszym uruchomieniu 2026-08-08). Bierzemy pelny loader i tylko PODMIENIAMY
    swiece na te, ktorych uzywaja pozostale sciezki."""
    from hai_common import ml_trainer as mt
    data = mt.load_symbol_data(symbol) or {}
    data = {**data, "1h": d1h, "4h": d4h, "1d": d1d}
    df = mt.build_features_for_symbol(data, symbol)
    if df is None or df.empty:
        return {}, None
    # UWAGA: df NIE konczy sie na ostatniej swiecy. build_features_for_symbol
    # obcina rozgrzewke z przodu i lookahead etykiet z tylu (400 swiec -> 272,
    # ostatnia starsza o 2 doby). Zwracamy WLASNY znacznik czasu, zeby pozostale
    # sciezki wzialy DOKLADNIE ten sam moment — inaczej test porownuje rozne
    # swiece i zglasza rozne wartosci jako "blad" (zlapane 2026-08-08).
    ost = df.iloc[-1]
    return {k: float(v) for k, v in ost.items()
            if isinstance(v, (int, float, np.floating, np.integer))
            and not str(k).startswith("label")}, pd.Timestamp(ost["timestamp"])


def sciezka_backtest(d1h, d4h, d1d, symbol, cel_ts=None):
    """Wola PRAWDZIWA metode backtestera (_build_feat_src), nie kopie logiki."""
    from hai_common.backtester import Backtester

    def swiece(d):
        return [{"timestamp": int(pd.Timestamp(t).value // 10**6), "open": float(o),
                 "high": float(h), "low": float(l), "close": float(c), "volume": float(v)}
                for t, o, h, l, c, v in zip(d["timestamp"], d["open"], d["high"],
                                            d["low"], d["close"], d["volume"])]
    c1h, c4h, c1d = swiece(d1h), swiece(d4h), swiece(d1d)
    bt = Backtester()
    ind = bt._precompute_indicators(c1h, c4h, c1d)
    n = len(c1h)
    neural = {"nbeats_pred_return_4h": np.zeros(n),
              "transformer_pred_return_4h": np.zeros(n),
              "taker_buy_ratio": np.full(n, 0.5)}
    # Derywaty PRAWDZIWA sciezka backtestu (_load_deriv_arrays), nie wlasne zera.
    # Bez tego test zglaszal funding_*/oi_*/ls_* jako "backtest=0" — a to byl
    # jego wlasny slownik, nie kod produkcyjny (falszywy alarm 2026-08-08).
    _tdf = pd.DataFrame({"ts_ms": np.array([c["timestamp"] for c in c1h], dtype=np.int64)})
    neural.update(bt._load_deriv_arrays(symbol, _tdf, n))
    fs = bt._build_feat_src(ind, neural, c1h, symbol, n)
    # indeks TEJ SAMEJ swiecy co trening (nie ostatniej — patrz sciezka_trening)
    idx = n - 1
    if cel_ts is not None:
        cel_ms = int(pd.Timestamp(cel_ts).value // 10**6)
        dop = [i for i, c in enumerate(c1h) if c["timestamp"] == cel_ms]
        if not dop:
            raise SystemExit(f"backtest: brak swiecy {cel_ts} w oknie")
        idx = dop[0]
    out = {}
    for k, v in fs.items():
        a = np.asarray(v)
        if a.ndim == 1 and len(a) == n:
            out[k] = float(a[idx])
    return out


def sciezka_live(d1h, d4h, d1d, symbol, cel_ts=None):
    """Derywaty podajemy z latest_deriv_live() — DOKLADNIE tak, jak robi to
    engine._compute_features w produkcji. Bez tego build_features_live bierze
    domyslne zera i test zglasza wlasny artefakt jako blad (zlapane 2026-08-08:
    6 falszywych alarmow na funding_*/oi_*)."""
    from hai_common.features import build_features_live, latest_deriv_live
    if cel_ts is not None:
        # live liczy cechy dla OSTATNIEJ podanej swiecy — zeby trafic w ten sam
        # moment co trening, obcinamy historie do celu (a nie bierzemy indeksu).
        d1h = d1h[d1h["timestamp"] <= pd.Timestamp(cel_ts)].reset_index(drop=True)
        d4h = d4h[d4h["timestamp"] <= pd.Timestamp(cel_ts)].reset_index(drop=True)
        d1d = d1d[d1d["timestamp"] <= pd.Timestamp(cel_ts)].reset_index(drop=True)
    d = latest_deriv_live(symbol)
    f = build_features_live(
        _Shim(), d1h["close"].tolist(), d4h["close"].tolist(), d1d["close"].tolist(),
        d1h["volume"].tolist(), highs_1h=d1h["high"].tolist(),
        lows_1h=d1h["low"].tolist(), symbol=symbol,
        timestamps_1h=d1h["timestamp"].tolist(),
        funding_rate=d["funding_rate"], funding_change_24h=d["funding_change_24h"],
        oi_total_log=d["oi_total_log"], oi_change_24h=d["oi_change_24h"],
        oi_zscore_30d=d["oi_zscore_30d"], taker_buy_ratio=d["taker_buy_ratio"],
        ls_ratio=d["ls_ratio"], ls_ratio_chg_24h=d["ls_ratio_chg_24h"],
        # timestamp MUSI byc podany — bez niego live bierze biezacy czas i cechy
        # kalendarzowe (x_weekend, hour_*, day_of_week) opisuja DZIS, a nie
        # porownywana swieca (falszywy alarm na x_weekend, 2026-08-08).
        timestamp=pd.Timestamp(cel_ts).tz_localize("UTC") if cel_ts is not None else None)
    if not f:
        return {}
    return {k: float(v) for k, v in f.items() if isinstance(v, (int, float, np.floating, np.integer))}


def porownaj(tren, back, live, tylko_krytyczne=False):
    wszystkie = sorted(set(tren) | set(back) | set(live))
    problemy = []
    for k in wszystkie:
        wart = {"trening": tren.get(k), "backtest": back.get(k), "live": live.get(k)}
        obecne = {p: v for p, v in wart.items() if v is not None}
        if len(obecne) < 2:
            continue  # cecha znana tylko jednej sciezce — nie ma czego porownywac
        brak = [p for p, v in wart.items() if v is None]
        zera = [p for p, v in obecne.items() if abs(v) < BLISKO_ZERA]
        niezera = [p for p, v in obecne.items() if abs(v) >= BLISKO_ZERA]

        if zera and niezera:
            problemy.append(("ZERO", k, wart,
                             f"{'+'.join(zera)}=0, a {'+'.join(niezera)} ma wartosc"))
            continue
        if brak and len(obecne) >= 2:
            problemy.append(("BRAK", k, wart, f"nie ma w: {'+'.join(brak)}"))
            continue
        if tylko_krytyczne:
            continue
        vals = list(obecne.values())
        skala = max(abs(v) for v in vals) or 1.0
        if (max(vals) - min(vals)) / skala > TOLERANCJA:
            problemy.append(("ROZJAZD", k, wart, f"rozrzut {max(vals)-min(vals):.6g}"))
    return problemy, len(wszystkie)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AVAX")
    ap.add_argument("--swiec", type=int, default=400)
    ap.add_argument("--tylko-krytyczne", action="store_true",
                    help="zglaszaj tylko ZERO i BRAK (pomijaj rozjazdy wartosci)")
    a = ap.parse_args()

    print(f"=== PARYTET CECH: {a.symbol}, {a.swiec} swiec ===\n")
    d1h, d4h, d1d = wczytaj(a.symbol, a.swiec)
    print(f"  ostatnia swieca: {d1h['timestamp'].iloc[-1]}")

    tren, cel_ts = sciezka_trening(d1h, d4h, d1d, a.symbol)
    print(f"  moment porownania: {cel_ts}  (trening obcina lookahead etykiet)")
    back = sciezka_backtest(d1h, d4h, d1d, a.symbol, cel_ts)
    live = sciezka_live(d1h, d4h, d1d, a.symbol, cel_ts)
    print(f"  trening : {len(tren):3d} cech")
    print(f"  backtest: {len(back):3d} cech")
    print(f"  live    : {len(live):3d} cech\n")

    problemy, ile = porownaj(tren, back, live, a.tylko_krytyczne)
    kryt = [p for p in problemy if p[0] in ("ZERO", "BRAK")]

    if not problemy:
        print(f"  OK — {ile} cech, zadnych rozbieznosci miedzy sciezkami")
        return 0

    for typ in ("ZERO", "BRAK", "ROZJAZD"):
        grupa = [p for p in problemy if p[0] == typ]
        if not grupa:
            continue
        print(f"  --- {typ} ({len(grupa)}) ---")
        for _, k, w, opis in grupa:
            def fmt(v):
                return "   brak" if v is None else f"{v:9.4f}"
            print(f"    {k:30s} tren={fmt(w['trening'])} back={fmt(w['backtest'])} "
                  f"live={fmt(w['live'])}  | {opis}")
        print()

    print(f"  RAZEM: {len(problemy)} rozbieznosci na {ile} cech "
          f"(w tym {len(kryt)} krytycznych: ZERO/BRAK)")
    return 1 if kryt else 0


if __name__ == "__main__":
    sys.exit(main())

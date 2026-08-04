#!/usr/bin/env python3
# ===========================================
# Lekki builder of_cvd/of_delta z 1-min kline BULK DUMPS (taker_buy_volume)
# ===========================================
# Gate of_cvd potrzebuje tylko CVD (skumulowana delta agresora). Klines maja
# taker_buy_base_volume -> delta = 2*taker_buy - volume. Dumpy 1m ~1MB/dzien
# (vs aggTrades ~100MB) -> zbieramy dla CALEJ puli tanio. Percentyl gate jest
# per-symbol (self-normalized), wiec of_cvd z klines jest spojne z gate.
#
# Uzycie: python3 build_of_cvd_klines.py --symbol ETH --days 180
#   (BTC juz ma bogatszy of_* z aggTrades - domyslnie pomijamy chyba ze --force)
import io, sys, zipfile, urllib.request, urllib.error, argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

import os
WH = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse"))
BASE = "https://data.binance.vision/data/futures/um/daily/klines"

# mapowanie stem -> symbol Binance (z ohlcv sa stemy jak BTC, ETH, 1000PEPE...)
def binance_sym(stem):
    return f"{stem}USDT"

def fetch_day(sym, d):
    url = f"{BASE}/{sym}/1m/{sym}-1m-{d.isoformat()}.zip"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    # kolumny: open_time,o,h,l,c,volume,close_time,qv,count,taker_buy_vol,taker_buy_qv,ignore
    df = df.iloc[:, [0, 5, 9]]
    df.columns = ["ot", "vol", "tbv"]
    # nowsze dumpy maja NAGLOWEK -> odrzuc wiersze nienumeryczne
    df["ot"] = pd.to_numeric(df["ot"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df["tbv"] = pd.to_numeric(df["tbv"], errors="coerce")
    df = df.dropna()
    return df if len(df) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stem = a.symbol
    out = WH / "orderflow" / "binance" / f"{stem}.parquet"
    if stem == "BTC" and not a.force:
        print(f"{stem}: pomijam (ma bogatszy of_* z aggTrades; --force by nadpisac)")
        return
    sym = binance_sym(stem)
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=a.days)
    parts = []
    d = start
    got = miss = 0
    while d <= end:
        x = fetch_day(sym, d)
        if x is not None and len(x):
            parts.append(x); got += 1
        else:
            miss += 1
        d += dt.timedelta(days=1)
    if not parts:
        print(f"{stem}: 0 dumpow ({sym}) - pomijam"); return
    m = pd.concat(parts, ignore_index=True)
    # ot moze byc w ms lub us; normalizuj
    ot = m["ot"].astype("int64")
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    m["ts"] = pd.to_datetime(ot, unit=unit)
    m["delta"] = 2 * m["tbv"] - m["vol"]
    m = m.sort_values("ts")
    # agreguj do godziny: suma delty i wolumenu
    m["hour"] = m["ts"].dt.floor("h")
    g = m.groupby("hour").agg(of_delta=("delta", "sum"),
                              _vol=("vol", "sum")).reset_index()
    g = g.rename(columns={"hour": "timestamp"})
    g["of_cvd"] = g["of_delta"].cumsum()
    g["of_delta_pct"] = np.where(g["_vol"] > 0, g["of_delta"] / g["_vol"], 0.0)
    g = g[["timestamp", "of_cvd", "of_delta", "of_delta_pct"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out, index=False)
    print(f"{stem}: {len(g)} godzin ({got}d OK/{miss} brak) "
          f"{str(g.timestamp.min())[:10]}->{str(g.timestamp.max())[:10]} of_cvd μ={g.of_cvd.mean():.0f}")

if __name__ == "__main__":
    main()

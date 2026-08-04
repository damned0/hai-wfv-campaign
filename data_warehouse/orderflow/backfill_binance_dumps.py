#!/usr/bin/env python3
# ===========================================
# gen.Flow — backfill order-flow z bulk dumps Binance (data.binance.vision)
# ===========================================
# Darmowe dzienne dumpy aggTrades (futures USD-M): ~18MB/dzien BTC, ~1.5M
# tradow. Duzo szybsze niz paginacja REST (~godzina na 90 dni vs ~11h REST).
# Kazdy dzien -> CSV -> znormalizowane trady -> cechy order-flow (build_flow_
# features) -> append do warehouse (dedup, CVD przeliczane globalnie).
# Uzupelnia historie pod pierwszy trening gen.Flow (neural sekwencyjny).
#
# Uzycie:
#   python3 backfill_binance_dumps.py --days 90            # ostatnie 90 dni
#   python3 backfill_binance_dumps.py --days 90 --symbol BTC
# URL: https://data.binance.vision/data/futures/um/daily/aggTrades/{SYM}/...
# Wynik: dopisuje do orderflow/binance/{SYM}.parquet (spojne z kolektorem live).
# ===========================================
import sys
import io
import zipfile
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_flow_features import build_flow_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("of.backfill")

OUT_ROOT = Path(__file__).resolve().parent
BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
_BINANCE = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def day_trades(sym_binance: str, day: str):
    """Pobierz+rozpakuj dzienny dump -> lista znormalizowanych tradow."""
    url = f"{BASE}/{sym_binance}/{sym_binance}-aggTrades-{day}.zip"
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return None  # dzien niedostepny (za stary / jeszcze nie ma)
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = zf.namelist()[0]
        df = pd.read_csv(zf.open(csv_name),
                         usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
        # is_buyer_maker True -> agresor SPRZEDAL (side=-1)
        df["side"] = df["is_buyer_maker"].map(lambda m: -1 if (m is True or m == "true") else 1)
        return [{"ts": int(t), "price": float(p), "qty": float(q), "side": int(s)}
                for t, p, q, s in zip(df["transact_time"], df["price"],
                                      df["quantity"], df["side"])]
    except Exception as e:
        log.warning(f"{day}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--symbol", default="BTC")
    args = ap.parse_args()

    sym = args.symbol
    binsym = _BINANCE.get(sym, f"{sym}USDT")
    out_dir = OUT_ROOT / "binance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sym}.parquet"

    today = datetime.now(timezone.utc).date()
    frames = []
    ok = miss = 0
    for i in range(args.days, 0, -1):
        day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        t0 = time.time()
        trades = day_trades(binsym, day)
        if not trades:
            miss += 1
            continue
        fc = build_flow_candles(trades)  # 24 swiece/dzien
        frames.append(fc)
        ok += 1
        log.info(f"  {day}: {len(trades)} tradow -> {len(fc)} swiec ({time.time()-t0:.0f}s)")

    if not frames:
        log.error("zero dni pobranych"); sys.exit(1)

    fresh = pd.concat(frames, ignore_index=True)
    if out.exists():
        old = pd.read_parquet(out)
        merged = pd.concat([old, fresh], ignore_index=True)
    else:
        merged = fresh
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["of_cvd"] = merged["of_delta"].cumsum().round(4)  # CVD globalne
    merged.to_parquet(out, compression="snappy", index=False)
    log.info(f"=== BACKFILL DONE | dni ok={ok} miss={miss} | total {len(merged)} swiec "
             f"({str(merged['timestamp'].min())[:10]} -> {str(merged['timestamp'].max())[:10]}) ===")


if __name__ == "__main__":
    main()

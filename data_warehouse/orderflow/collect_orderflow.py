#!/usr/bin/env python3
# ===========================================
# gen.Flow — kolektor order-flow (inkrementalny, append-delta)
# ===========================================
# Pobiera trady od ostatniej swiecy w magazynie do teraz, liczy cechy
# order-flow (build_flow_features) i DOPISUJE do parquet (dedup po timestamp).
# Wzorzec jak ls_ratio_collector (append-delta, nie skip-if-exists).
# AKTYWNE ZRODLO: ORDERFLOW_PROVIDER (domyslnie binance, darmowe). Symbole:
# ORDERFLOW_SYMBOLS (domyslnie BTC — reżim BTC-only). mmt/inne: gotowe w
# providers.py, wystarczy ORDERFLOW_PROVIDER=mmt + MMT_API_KEY.
#
# Uzycie:
#   python3 collect_orderflow.py                 # inkrement (od ostatniej swiecy)
#   python3 collect_orderflow.py --backfill 48   # seed: ostatnie 48h
# Cron: co 1h (order-flow jest ciezki — nie czesciej). Wynik:
#   data_warehouse/orderflow/{provider}/{SYM}.parquet
# ===========================================
import os
import sys
import time
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from providers import get_provider
from build_flow_features import build_flow_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("orderflow.collect")

TF_MS = 3600_000
OUT_ROOT = Path(__file__).resolve().parent


def collect_symbol(prov, sym: str, backfill_h: int = 0) -> str:
    out_dir = OUT_ROOT / prov.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sym}.parquet"
    now_ms = int(time.time() * 1000)
    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame()

    if backfill_h > 0 or existing.empty:
        start = now_ms - (backfill_h or 48) * TF_MS
    else:
        last = pd.to_datetime(existing["timestamp"]).max()
        start = int(last.timestamp() * 1000) + 1  # od nastepnej swiecy
    # nie liczymy biezacej (niepelnej) swiecy — do poczatku biezacej godziny
    end = (now_ms // TF_MS) * TF_MS
    if start >= end:
        log.info(f"{prov.name}:{sym} aktualne (brak nowych pelnych swiec)")
        return "up-to-date"

    log.info(f"{prov.name}:{sym} pobieram trady {pd.to_datetime(start,unit='ms')} -> {pd.to_datetime(end,unit='ms')}")
    trades = prov.fetch_trades(sym, start, end)
    if not trades:
        log.warning(f"{prov.name}:{sym} zero tradow w oknie")
        return "no-data"
    fresh = build_flow_candles(trades, TF_MS)
    if fresh.empty:
        return "no-candles"

    if not existing.empty:
        merged = pd.concat([existing, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        # of_cvd musi byc skumulowane GLOBALNIE, nie per-batch -> przelicz
        merged["of_cvd"] = merged["of_delta"].cumsum().round(4)
    else:
        merged = fresh
    merged.to_parquet(out, compression="snappy", index=False)
    log.info(f"{prov.name}:{sym} +{len(fresh)} swiec (total {len(merged)}) -> {out.name}")
    return f"+{len(fresh)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0, help="seed ostatnie N godzin")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    prov = get_provider(args.provider)
    if not prov.active:
        log.error(f"dostawca {prov.name} NIEAKTYWNY (brak klucza?) — przerywam")
        sys.exit(1)
    syms = os.environ.get("ORDERFLOW_SYMBOLS", "BTC").split(",")
    log.info(f"=== gen.Flow collect | provider={prov.name} | symbole={syms} ===")
    for s in syms:
        try:
            collect_symbol(prov, s.strip(), args.backfill)
        except Exception as e:
            log.warning(f"{s}: blad {e}")
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()

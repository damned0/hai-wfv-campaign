"""
HAI_EPV Engine ver.10 Final — routes/screening.py
Created by Hauzer | Coded & produced by Claude Sonnet 5

Endpoint screening - serwuje wyniki screeningu (score 0/1/2/3 + PnL per okres)
dla podgladu mysli modelu na calym rynku. Czyta najnowszy screening_*.txt.
"""
from fastapi import APIRouter
from pathlib import Path
import re
import glob
import os
import asyncio
from datetime import datetime

router = APIRouter()

BASE = Path(__file__).resolve().parent.parent


def _parse_screening_file(path):
    """Parsuje plik screening -> {SYM: {score, d90, d180, d365}}."""
    out = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out

    # Linie per-symbol: "AAVE  +33  +47  +42  3/3" lub "1INCH  NA  NA  NA  0/3"
    # Format: SYM  90d  180d  365d  X/3
    pat = re.compile(
        r'^([A-Z0-9]+)\s+'
        r'([+\-]?\d+|NA)\s+'
        r'([+\-]?\d+|NA)\s+'
        r'([+\-]?\d+|NA)\s+'
        r'(\d)/3\s*$'
    )
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        sym, d90, d180, d365, score = m.groups()

        def num(x):
            return None if x == 'NA' else int(x)

        out[sym] = {
            'score': int(score),       # 0/1/2/3
            'd90':   num(d90),
            'd180':  num(d180),
            'd365':  num(d365),
        }
    return out


def _latest_screening_file():
    """Najnowszy screening_*.txt w katalogu instancji."""
    files = glob.glob(str(BASE / "screening_*.txt"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@router.post("/ai/screening/run")
async def run_screening():
    """Uruchamia re-screening (90d/180d/365d) w tle i zapisuje plik screening_EP_*.txt."""
    async def _run():
        from ..backtester import backtester
        from ..state import state
        try:
            per_sym = await backtester.run_screening()
            lines = []
            for sym in sorted(per_sym):
                data = per_sym[sym]
                def fmt(v):
                    if v is None:
                        return "NA"
                    return (f"+{int(v)}" if v > 0 else str(int(v)))
                d90s  = fmt(data.get("d90"))
                d180s = fmt(data.get("d180"))
                d365s = fmt(data.get("d365"))
                lines.append(f"{sym}  {d90s}  {d180s}  {d365s}  {data['score']}/3")
            ts   = datetime.now().strftime("%Y%m%d_%H%M")
            path = BASE / f"screening_EP_{ts}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            state.add_log("ai", "INFO", event="SCREENING",
                          message=f"Re-screen done: {len(per_sym)} symbols → {path.name}")
        except Exception as e:
            from ..state import state
            state.add_log("ai", "ERROR", event="SCREENING", message=f"Re-screen error: {e}")

    asyncio.create_task(_run())
    return {"status": "ok", "message": "Re-screening uruchomiony w tle (~5-10 min)"}


@router.get("/api/screening")
async def get_screening():
    """Zwraca {updated, file, symbols: {SYM: {score, d90, d180, d365}}}."""
    f = _latest_screening_file()
    if not f:
        return {"updated": None, "file": None, "symbols": {}}
    symbols = _parse_screening_file(f)
    try:
        mtime = os.path.getmtime(f)
    except Exception:
        mtime = None
    return {
        "updated": mtime,
        "file": os.path.basename(f),
        "count": len(symbols),
        "symbols": symbols,
    }

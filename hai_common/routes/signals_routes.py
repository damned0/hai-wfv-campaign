"""
HAI_EPV Engine ver.10 Final — routes/signals_routes.py
Created by Hauzer | Coded & produced by Claude Sonnet 5

Signals routes — model signals and strategy signals for all warehouse symbols.

GET /api/signals/model      -> raw ensemble predictions (no strategy filters)
GET /api/signals/strategy   -> strategy-filtered signals (BB, ADX, session, regime)
GET /api/signals/screened   -> combined model+strategy for screened symbols (score ≥ 2)
POST /api/signals/refresh   -> clear cache and trigger rescan
POST /api/signals/screened/refresh -> trigger screened rescan

Scan runs in background; endpoints return immediately from cache.
Cache TTL: 5 minutes (screened: 15 minutes).
"""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)

# Shared warehouse (train data source, 105 symbols)
WH_BASE = Path("/root/ProjektHAI/data_warehouse/ohlcv/binance")
EP_BASE  = Path(__file__).resolve().parent.parent   # HAI_EPV root
BARS = 300
CACHE_TTL          = 300   # 5 minutes for model/strategy
CACHE_TTL_SCREENED = 900   # 15 minutes for screened (slower scan)

_cache: Dict = {}
_scanning: Dict[str, bool] = {}  # prevents concurrent scans per type


def _read_sym(symbol: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Returns (closes_1h, vols_1h, closes_4h, closes_1d)."""
    def _load(tf: str):
        p = WH_BASE / tf / f"{symbol}.parquet"
        if not p.exists():
            return [], []
        try:
            df = pd.read_parquet(p, columns=["close", "volume"]).tail(BARS)
            return df["close"].dropna().tolist(), df["volume"].dropna().tolist()
        except Exception:
            return [], []

    c1h, v1h = _load("1h")
    c4h, _ = _load("4h")
    c1d, _ = _load("1d")
    return c1h, v1h, c4h, c1d


def _list_symbols() -> List[str]:
    d = WH_BASE / "1h"
    if not d.exists():
        return []
    return sorted(f.stem for f in d.glob("*.parquet"))


def _sync_scan_model() -> Dict:
    """Raw ensemble predictions, no strategy filters."""
    try:
        from ..ensemble import ensemble
        from ..features import build_features_live, build_feature_sequence_live
        from ..strategies.ai_strategy import AIStrategy

        if not ensemble.active:
            ensemble.load_models()
        strategy = AIStrategy("neutral")

        # Check if any sequence models are loaded
        has_seq_models = any(
            hasattr(m, "model_type") and m.model_type in ("lstm", "tcn")
            for m in ensemble.models.values()
        )

        symbols = _list_symbols()
        results, skipped = [], 0

        for sym in symbols:
            try:
                c1h, v1h, c4h, c1d = _read_sym(sym)
                if len(c1h) < 50 or len(v1h) < 30:
                    skipped += 1
                    continue

                feat = build_features_live(
                    strategy=strategy,
                    prices_1h=c1h, prices_4h=c4h, prices_1d=c1d,
                    volumes_1h=v1h,
                )
                if not feat:
                    skipped += 1
                    continue

                seq = None
                if has_seq_models:
                    seq = build_feature_sequence_live(
                        strategy=strategy,
                        prices_1h=c1h, prices_4h=c4h, prices_1d=c1d,
                        volumes_1h=v1h,
                    )

                res = ensemble.predict(feat, seq=seq)
                action = res.get("action", "NEUTRAL")
                conf = res.get("confidence", 0.0)
                if action == "NEUTRAL":
                    continue

                cur = float(c1h[-1])
                atr = feat.get("atr_pct", 2.0) / 100.0 * cur

                if action == "LONG":
                    entry = _fmt(cur - 0.3 * atr)
                    tp    = _fmt(cur + 3.0 * atr)
                    sl    = _fmt(cur - 1.0 * atr)
                else:
                    entry = _fmt(cur + 0.3 * atr)
                    tp    = _fmt(cur - 3.0 * atr)
                    sl    = _fmt(cur + 1.0 * atr)
                rr = round(abs(tp - entry) / abs(sl - entry), 2) if abs(sl - entry) > 0 else 0.0

                results.append({
                    "symbol":     sym,
                    "action":     action,
                    "confidence": round(conf, 3),
                    "price":      _fmt(cur),
                    "entry":      entry,
                    "tp":         tp,
                    "sl":         sl,
                    "rr":         rr,
                    "rsi":        round(feat.get("rsi", 50.0), 1),
                    "bb_pos":     round(feat.get("price_position_bb", 0.5), 3),
                    "adx":        round(feat.get("adx_14", 0.0), 1),
                    "momentum":   round(feat.get("momentum", 0.0), 2),
                })
            except Exception as e:
                logger.debug(f"model scan {sym}: {e}")
                skipped += 1

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(symbols),
            "signals": len(results),
            "skipped": skipped,
            "results": results,
        }
    except Exception as e:
        logger.error(f"_sync_scan_model: {e}", exc_info=True)
        return {"error": str(e), "results": []}


def _fmt(price: float) -> float:
    if price >= 1000: return round(price, 2)
    if price >= 1:    return round(price, 4)
    if price >= 0.01: return round(price, 5)
    return round(price, 7)


def _sync_scan_strategy() -> Dict:
    """Strategy-filtered signals (BB extreme + ADX + session + regime)."""
    try:
        from ..strategies.ai_strategy import AIStrategy

        strategy = AIStrategy("neutral")
        symbols = _list_symbols()
        results, skipped = [], 0

        for sym in symbols:
            try:
                c1h, v1h, c4h, c1d = _read_sym(sym)
                if len(c1h) < strategy.min_history:
                    skipped += 1
                    continue

                score, action = strategy.score_symbol(
                    f"{sym}/USDT:USDT", c1h, c4h, c1d, v1h
                )
                if action in ("LONG", "SHORT"):
                    cur = float(c1h[-1])
                    closes_arr = np.array(c1h[-15:], dtype=np.float64)
                    diffs = np.abs(np.diff(closes_arr))
                    atr = float(diffs.mean()) if len(diffs) > 0 else cur * 0.02

                    if action == "LONG":
                        entry = _fmt(cur - 0.3 * atr)
                        tp    = _fmt(cur + 3.0 * atr)
                        sl    = _fmt(cur - 1.0 * atr)
                    else:
                        entry = _fmt(cur + 0.3 * atr)
                        tp    = _fmt(cur - 3.0 * atr)
                        sl    = _fmt(cur + 1.0 * atr)

                    rr = round(abs(tp - entry) / abs(sl - entry), 2) if abs(sl - entry) > 0 else 0.0

                    results.append({
                        "symbol":     sym,
                        "action":     action,
                        "score":      round(score, 1),
                        "confidence": round(score / 100, 3),
                        "price":      _fmt(cur),
                        "entry":      entry,
                        "tp":         tp,
                        "sl":         sl,
                        "rr":         rr,
                    })
            except Exception as e:
                logger.debug(f"strategy scan {sym}: {e}")
                skipped += 1

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(symbols),
            "signals": len(results),
            "skipped": skipped,
            "results": results,
        }
    except Exception as e:
        logger.error(f"_sync_scan_strategy: {e}", exc_info=True)
        return {"error": str(e), "results": []}


def _read_screened_symbols(min_score: int = 2) -> Dict[str, int]:
    """Parse latest screening_EP_*.txt, return {symbol: score} for score >= min_score."""
    files = sorted(EP_BASE.glob("screening_EP_*.txt"))
    if not files:
        return {}
    latest = files[-1]
    result = {}
    for line in latest.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sym = parts[0]
        score_str = parts[-1]
        if "/" in score_str:
            try:
                score = int(score_str.split("/")[0])
                if score >= min_score:
                    result[sym] = score
            except ValueError:
                pass
    return result


def _sync_scan_screened(min_score: int = 2) -> Dict:
    """Combined model + strategy scan for screened symbols (score >= min_score)."""
    screened = _read_screened_symbols(min_score)
    if not screened:
        return {"error": "Brak pliku screeningu lub brak symboli z wymaganym score", "results": []}

    try:
        from ..ensemble import ensemble
        from ..features import build_features_live, build_feature_sequence_live
        from ..strategies.ai_strategy import AIStrategy

        if not ensemble.active:
            ensemble.load_models()
        strategy = AIStrategy("neutral")
        has_seq = any(
            hasattr(m, "model_type") and m.model_type in ("lstm", "tcn")
            for m in ensemble.models.values()
        )
    except Exception as e:
        return {"error": f"Init error: {e}", "results": []}

    combined = []
    for sym, sc_score in sorted(screened.items()):
        try:
            c1h, v1h, c4h, c1d = _read_sym(sym)
            if len(c1h) < max(50, strategy.min_history) or len(v1h) < 30:
                combined.append({"symbol": sym, "screening_score": sc_score, "model": None, "strategy": None})
                continue

            cur = float(c1h[-1])

            # ── Model prediction ──
            m_result = None
            try:
                feat = build_features_live(
                    strategy=strategy,
                    prices_1h=c1h, prices_4h=c4h, prices_1d=c1d, volumes_1h=v1h,
                )
                if feat:
                    seq = build_feature_sequence_live(
                        strategy=strategy,
                        prices_1h=c1h, prices_4h=c4h, prices_1d=c1d, volumes_1h=v1h,
                    ) if has_seq else None
                    res    = ensemble.predict(feat, seq=seq)
                    action = res.get("action", "NEUTRAL")
                    conf   = res.get("confidence", 0.0)
                    atr    = feat.get("atr_pct", 2.0) / 100.0 * cur
                    if action == "LONG":
                        entry, tp, sl = _fmt(cur - 0.3*atr), _fmt(cur + 3.0*atr), _fmt(cur - 1.0*atr)
                    elif action == "SHORT":
                        entry, tp, sl = _fmt(cur + 0.3*atr), _fmt(cur - 3.0*atr), _fmt(cur + 1.0*atr)
                    else:
                        entry = tp = sl = None
                    rr = round(abs(tp - entry) / abs(sl - entry), 2) if entry is not None and abs(sl - entry) > 0 else 0.0
                    m_result = {
                        "action":     action,
                        "confidence": round(conf, 3),
                        "price":      _fmt(cur),
                        "entry":      entry,
                        "tp":         tp,
                        "sl":         sl,
                        "rr":         rr,
                        "rsi":        round(feat.get("rsi", 50.0), 1),
                        "bb_pos":     round(feat.get("price_position_bb", 0.5), 3),
                        "adx":        round(feat.get("adx_14", 0.0), 1),
                    }
            except Exception as e:
                logger.debug(f"screened model {sym}: {e}")

            # ── Strategy signal ──
            s_result = None
            try:
                score, action = strategy.score_symbol(f"{sym}/USDT:USDT", c1h, c4h, c1d, v1h)
                if action in ("LONG", "SHORT"):
                    closes_arr = np.array(c1h[-15:], dtype=np.float64)
                    diffs = np.abs(np.diff(closes_arr))
                    atr   = float(diffs.mean()) if len(diffs) > 0 else cur * 0.02
                    if action == "LONG":
                        entry, tp, sl = _fmt(cur - 0.3*atr), _fmt(cur + 3.0*atr), _fmt(cur - 1.0*atr)
                    else:
                        entry, tp, sl = _fmt(cur + 0.3*atr), _fmt(cur - 3.0*atr), _fmt(cur + 1.0*atr)
                    rr = round(abs(tp - entry) / abs(sl - entry), 2) if abs(sl - entry) > 0 else 0.0
                    s_result = {
                        "action": action,
                        "score":  round(score, 1),
                        "price":  _fmt(cur),
                        "entry":  entry,
                        "tp":     tp,
                        "sl":     sl,
                        "rr":     rr,
                    }
            except Exception as e:
                logger.debug(f"screened strategy {sym}: {e}")

            # ── S/R levels ──
            sr_result = None
            try:
                sr_result = strategy.calculate_sr_levels(c1h, c4h)
            except Exception as e:
                logger.debug(f"screened sr {sym}: {e}")

            combined.append({
                "symbol":          sym,
                "screening_score": sc_score,
                "model":           m_result,
                "strategy":        s_result,
                "sr":              sr_result,
            })
        except Exception as e:
            logger.debug(f"screened {sym}: {e}")
            combined.append({"symbol": sym, "screening_score": sc_score, "model": None, "strategy": None})

    # Sort: aligned both > any both > model only > strategy only > none; then conf desc
    def _key(x):
        m, s = x.get("model"), x.get("strategy")
        has_m = m is not None and m.get("action", "NEUTRAL") != "NEUTRAL"
        has_s = s is not None
        aligned = has_m and has_s and m.get("action") == s.get("action")
        conf    = m.get("confidence", 0.0) if has_m else 0.0
        pri     = 3 if aligned else 2 if (has_m and has_s) else 1 if (has_m or has_s) else 0
        return (-pri, -x.get("screening_score", 0), -conf)

    combined.sort(key=_key)

    m_cnt    = sum(1 for x in combined if x.get("model") and x["model"].get("action","NEUTRAL") != "NEUTRAL")
    s_cnt    = sum(1 for x in combined if x.get("strategy"))
    both_cnt = sum(1 for x in combined
                   if x.get("model") and x["model"].get("action","NEUTRAL") != "NEUTRAL"
                   and x.get("strategy")
                   and x["model"].get("action") == x["strategy"].get("action"))

    return {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "screened_symbols":  len(screened),
        "model_signals":     m_cnt,
        "strategy_signals":  s_cnt,
        "both_aligned":      both_cnt,
        "results":           combined,
    }


async def _trigger_scan(scan_type: str):
    """Launch scan in thread if not already running, update cache on completion."""
    if _scanning.get(scan_type):
        return
    _scanning[scan_type] = True
    try:
        if scan_type == "model":
            fn = _sync_scan_model
        elif scan_type == "strategy":
            fn = _sync_scan_strategy
        else:
            fn = _sync_scan_screened
        data = await asyncio.to_thread(fn)
        _cache[scan_type] = {"ts": datetime.now(timezone.utc).timestamp(), "data": data}
        if scan_type == "screened":
            logger.info(f"signals/screened: {data.get('model_signals',0)} model, "
                        f"{data.get('strategy_signals',0)} strategy, "
                        f"{data.get('both_aligned',0)} aligned "
                        f"from {data.get('screened_symbols',0)} screened")
        else:
            logger.info(f"signals/{scan_type}: {data.get('signals', 0)} signals from "
                        f"{data.get('total_symbols', 0)} symbols")
    except Exception as e:
        logger.error(f"_trigger_scan {scan_type}: {e}")
    finally:
        _scanning[scan_type] = False


def _is_stale(scan_type: str) -> bool:
    cached = _cache.get(scan_type)
    if not cached:
        return True
    ttl = CACHE_TTL_SCREENED if scan_type == "screened" else CACHE_TTL
    return (datetime.now(timezone.utc).timestamp() - cached["ts"]) >= ttl


@router.get("/api/signals/model")
async def get_model_signals():
    """Surowe predykcje ensemble dla wszystkich symboli w warehouse."""
    if _is_stale("model"):
        asyncio.create_task(_trigger_scan("model"))
    cached = _cache.get("model")
    if cached:
        return cached["data"]
    return {
        "status": "scanning",
        "message": "Pierwsza analiza w toku (~30s), odśwież za chwilę",
        "results": [],
        "signals": 0,
    }


@router.get("/api/signals/strategy")
async def get_strategy_signals():
    """Sygnały po pełnych filtrach AIStrategy: BB extreme, ADX, sesja, reżim."""
    if _is_stale("strategy"):
        asyncio.create_task(_trigger_scan("strategy"))
    cached = _cache.get("strategy")
    if cached:
        return cached["data"]
    return {
        "status": "scanning",
        "message": "Pierwsza analiza w toku (~30s), odśwież za chwilę",
        "results": [],
        "signals": 0,
    }


@router.post("/api/signals/refresh")
async def refresh_signals():
    """Czyści cache i natychmiast wyzwala nowy skan obu typów."""
    _cache.pop("model", None)
    _cache.pop("strategy", None)
    asyncio.create_task(_trigger_scan("model"))
    asyncio.create_task(_trigger_scan("strategy"))
    return {"status": "scan triggered"}


@router.get("/api/signals/screened")
async def get_screened_signals():
    """Predykcje model + strategia dla screened symboli (score ≥ 2)."""
    cached = _cache.get("screened")
    if cached and not _is_stale("screened"):
        return cached["data"]
    if not _scanning.get("screened"):
        asyncio.create_task(_trigger_scan("screened"))
    if cached:
        return cached["data"]
    return {
        "status":   "scanning",
        "message":  "Analiza screened symboli w toku (~2-3 min), odśwież za chwilę",
        "results":  [],
        "screened_symbols": 0,
        "model_signals": 0,
        "strategy_signals": 0,
        "both_aligned": 0,
    }


@router.post("/api/signals/screened/refresh")
async def refresh_screened_signals():
    """Czyści cache screened i wyzwala nowy skan."""
    _cache.pop("screened", None)
    asyncio.create_task(_trigger_scan("screened"))
    return {"status": "screened scan triggered"}


@router.get("/api/signals/sr/{symbol}")
async def get_sr_levels(symbol: str):
    """Wsparcia i opory techniczne dla symbolu z warehouse."""
    symbol = symbol.upper().replace("-", "").replace("/USDT", "").replace(":USDT", "")
    try:
        df_1h = pd.read_parquet(WH_BASE / "1h" / f"{symbol}.parquet").tail(BARS)
        df_4h = pd.read_parquet(WH_BASE / "4h" / f"{symbol}.parquet").tail(100) if (WH_BASE / "4h" / f"{symbol}.parquet").exists() else None
        prices_1h = df_1h["close"].tolist()
        prices_4h = df_4h["close"].tolist() if df_4h is not None else []
        if len(prices_1h) < 20:
            return {"error": "Za mało danych"}
        from ..strategies.ai_strategy import AIStrategy
        strategy = AIStrategy.__new__(AIStrategy)
        sr = strategy.calculate_sr_levels(prices_1h, prices_4h)
        sr["symbol"] = symbol
        sr["timestamp"] = datetime.now(timezone.utc).isoformat()
        return sr
    except Exception as e:
        logger.error(f"SR levels error for {symbol}: {e}")
        return {"error": str(e)}

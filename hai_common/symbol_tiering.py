"""Tiering symboli na podstawie wyników WFV i backtestów.

Tiers:
  CORE: symbole z potwierdzoną przewagą (SEI, ENJ, STRK, ONE, ...)
  STANDARD: reszta uniwersum (domyślna waga 1.0)
  OBSERVE: nowe/niskowolumenowe symbole (mniejsza waga)
  BLOCKED: symbole chronicznie stratne

Konfiguracja przez .env:
  SYMBOL_TIERING=on|off (domyślnie off)
  SYMBOL_CORE=SEI,ENJ,STRK,ONE,RUNE,LDO,APT,ONDO
  SYMBOL_BLOCKED=...
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Domyślne tiering na podstawie raportu NewHorizonts
_DEFAULT_CORE_SYMBOLS = [
    "SEI/USDT:USDT", "ENJ/USDT:USDT", "STRK/USDT:USDT", "ONE/USDT:USDT",
    "RUNE/USDT:USDT", "LDO/USDT:USDT", "APT/USDT:USDT", "ONDO/USDT:USDT",
    "ASTER/USDT:USDT", "XPL/USDT:USDT",
]

_DEFAULT_STANDARD_SYMBOLS = [
    "DYDX/USDT:USDT", "SAND/USDT:USDT", "DOT/USDT:USDT", "CELR/USDT:USDT",
    "IMX/USDT:USDT", "WIF/USDT:USDT", "CRV/USDT:USDT", "MANA/USDT:USDT",
    "QNT/USDT:USDT", "FARTCOIN/USDT:USDT",
]

_DEFAULT_BLOCKED_SYMBOLS: List[str] = []


class SymbolTiering:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.core: List[str] = []
        self.standard: List[str] = []
        self.observe: List[str] = []
        self.blocked: List[str] = []
        self._symbol_map: Dict[str, str] = {}
        self._load_from_env()

    def _load_from_env(self):
        self.enabled = os.getenv("SYMBOL_TIERING", "off").lower() == "on"
        if not self.enabled:
            return
        raw = os.getenv("SYMBOL_CORE", "")
        self.core = [s.strip() for s in raw.split(",") if s.strip()] if raw else _DEFAULT_CORE_SYMBOLS
        raw = os.getenv("SYMBOL_BLOCKED", "")
        self.blocked = [s.strip() for s in raw.split(",") if s.strip()] if raw else _DEFAULT_BLOCKED_SYMBOLS
        for sym in self.core:
            self._symbol_map[sym] = "core"
        for sym in self.blocked:
            self._symbol_map[sym] = "blocked"

    def tier(self, symbol: str) -> str:
        if not self.enabled:
            return "standard"
        return self._symbol_map.get(symbol, "standard")

    def weight(self, symbol: str) -> float:
        t = self.tier(symbol)
        return {"core": 1.5, "standard": 1.0, "observe": 0.5, "blocked": 0.0}.get(t, 1.0)

    def filter_symbols(self, symbols: List[str]) -> List[str]:
        if not self.enabled:
            return symbols
        return [s for s in symbols if self.tier(s) != "blocked"]

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "core": self.core,
            "blocked": self.blocked,
        }


symbol_tiering = SymbolTiering()


def analyze_wfv_by_symbol(wfv_results_dir: Path) -> dict:
    """Analizuje wyniki WFV per symbol dla automatycznego tieringu."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "windows": 0})

    for f in wfv_results_dir.glob("wfv_*.json"):
        try:
            data = json.loads(f.read_text())
            for window in data.get("windows", []):
                for trade in window.get("trades", []):
                    sym = trade.get("symbol", "unknown")
                    stats[sym]["trades"] += 1
                    stats[sym]["wins"] += 1 if trade.get("pnl", 0) > 0 else 0
                    stats[sym]["pnl"] += trade.get("pnl", 0)
                    stats[sym]["windows"] += 1
        except Exception as e:
            logger.warning(f"Błąd analizy {f.name}: {e}")

    result = {}
    for sym, s in stats.items():
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        result[sym] = {
            "trades": s["trades"],
            "wr": round(wr, 1),
            "pnl": round(s["pnl"], 2),
            "windows": s["windows"],
            "score": round(s["pnl"] * wr / 100, 2),
        }
    return dict(sorted(result.items(), key=lambda x: x[1]["score"], reverse=True))

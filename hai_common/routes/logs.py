# ===========================================
# HAI_EPV Engine ver.10 Final — routes/logs.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: /logs/system, /logs/trading, /logs/ai — odczyt logow z state
# (in-memory + plikowe) dla widokow dashboardu.
# ===========================================
from fastapi import APIRouter, Request
from ..state import state
from ..app import templates

router = APIRouter()


def _filter_by_level(logs, level):
    """Filtr po poziomie. level=None lub 'ALL' -> bez filtra.
    Dopasowanie case-insensitive po polu 'level'."""
    if not logs or not level or level.upper() == "ALL":
        return logs
    lv = level.upper()
    return [l for l in logs if str(l.get("level", "")).upper() == lv]



@router.get("/logs/system")
async def logs_system(limit: int = 50, level: str = None):
    return _filter_by_level(state.get_logs("system", limit) or [], level)


@router.get("/logs/trading")
async def logs_trading(limit: int = 50, level: str = None):
    return _filter_by_level(state.get_logs("trading", limit) or [], level)


@router.get("/logs/ai")
async def logs_ai(limit: int = 50, level: str = None):
    return _filter_by_level(state.get_logs("ai", limit) or [], level)

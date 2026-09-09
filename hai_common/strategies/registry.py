# ===========================================
# HAI_EPV Engine ver.10 Final — strategies/registry.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: register_strategy()/get_strategy() - rejestr strategii po nazwie.
# W praktyce tylko "ai_strategy" jest wolane (engine/backtester/routes/ai) -
# pozostale zarejestrowane strategie (bollinger/momentum/rsi_divergence itp.)
# byly martwe i przeniesione do archiwum poza HAIs (audyt 2026-07-05).
# ===========================================
import logging
from typing import Dict, List, Optional, Type, Any
from .base import BaseStrategy

logger = logging.getLogger(__name__)

_strategy_registry: Dict[str, Type[BaseStrategy]] = {}
_active_strategies: Dict[str, BaseStrategy] = {}
_default_strategy: str = "ai_strategy"

def register_strategy(name: str, strategy_class: Type[BaseStrategy]):
    _strategy_registry[name] = strategy_class
    logger.info(f"? Strategia: {name}")

def get_strategy(name: str = None) -> Optional[BaseStrategy]:
    if name is None:
        from ..config import config
        name = config.DEFAULT_STRATEGY
    if name in _active_strategies:
        return _active_strategies[name]
    if name in _strategy_registry:
        strategy = _strategy_registry[name]()
        _active_strategies[name] = strategy
        return strategy
    return None

def get_all_strategies() -> List[Dict[str, Any]]:
    return [{"name": name, "active": name in _active_strategies} for name in _strategy_registry]

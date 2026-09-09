# ===========================================
# HAI_EPV Engine ver.10 Final — core/events.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: EventBus (pub/sub wewnetrzny), Events (staly rejestr nazw
# zdarzen systemowych/tradingowych).
# ===========================================
from datetime import timezone

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Any, Awaitable
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventBus:
    """
    System nerwowy HyperAI.
    Moduły komunikują się tylko przez EventBus.
    Zero bezpośrednich importów między modułami.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[Dict] = []
        self._max_history = 200

    def subscribe(self, event_name: str, callback: Callable, async_mode: bool = False):
        """Subskrybuje callback na zdarzenie"""
        if async_mode:
            self._async_subscribers[event_name].append(callback)
        else:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable):
        """Usuwa subskrypcję"""
        if callback in self._subscribers.get(event_name, []):
            self._subscribers[event_name].remove(callback)
        if callback in self._async_subscribers.get(event_name, []):
            self._async_subscribers[event_name].remove(callback)

    def emit(self, event_name: str, data: Dict[str, Any] = None):
        """Emituje zdarzenie - wywołuje wszystkie callbacki"""
        data = data or {}
        data["_event"] = event_name
        data["_timestamp"] = datetime.now(timezone.utc).isoformat()

        self._add_to_history(event_name, data)

        for callback in self._subscribers.get(event_name, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"EventBus error [{event_name}]: {e}")

        for callback in self._async_subscribers.get(event_name, []):
            try:
                asyncio.create_task(self._safe_async_call(callback, data))
            except Exception as e:
                logger.error(f"EventBus async error [{event_name}]: {e}")

    async def _safe_async_call(self, callback: Callable, data: Dict) -> Awaitable:
        """Bezpiecznie wywołuje async callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"EventBus callback error: {e}")

    def _add_to_history(self, event_name: str, data: Dict):
        """Dodaje zdarzenie do historii"""
        self._history.append({
            "event": event_name,
            "timestamp": data.get("_timestamp"),
            "data": {k: v for k, v in data.items() if not k.startswith("_")}
        })
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, limit: int = 50, event_name: str = None) -> List[Dict]:
        """Pobiera historię zdarzeń"""
        history = self._history
        if event_name:
            history = [h for h in history if h["event"] == event_name]
        return history[-limit:]


class Events:
    """Nazwy standardowych zdarzeń w HyperAI"""

    # Ceny
    PRICE_UPDATE = "price_update"
    MULTI_PRICE_UPDATE = "multi_price_update"

    # Sygnały z TOP 100
    SIGNAL_LONG = "signal_long"
    SIGNAL_SHORT = "signal_short"
    SIGNAL_CLOSE = "signal_close"
    SIGNAL_NEUTRAL = "signal_neutral"
    TOP5_UPDATE = "top5_update"

    # Trading
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    ORDER_FILLED = "order_filled"
    ORDER_FAILED = "order_failed"

    # System
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    MODE_CHANGED = "mode_changed"

    # AI
    AI_TRAINING_STARTED = "ai_training_started"
    AI_TRAINING_COMPLETED = "ai_training_completed"
    AI_SIGNAL = "ai_signal"

    # Risk
    RISK_BLOCKED = "risk_blocked"
    RISK_WARNING = "risk_warning"
    PANIC_MODE = "panic_mode"
    PANIC_MODE_CLEARED = "panic_mode_cleared"


bus = EventBus()
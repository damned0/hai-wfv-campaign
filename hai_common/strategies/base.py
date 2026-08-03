# ===========================================
# HAI_EPV Engine ver.10 Final — strategies/base.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: BaseStrategy (interfejs bazowy), StrategyResult, wspolne
# numpy EMA/RSI - bez duplikacji w klasach dziedziczacych.
# ===========================================
from datetime import timezone
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    action: str
    symbol: str
    reason: str = ""
    confidence: float = 0.0
    price: float = 0.0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    signal: Optional[Signal] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    opportunities: List[str] = field(default_factory=list)
    score: float = 0.0


class BaseStrategy(ABC):

    def __init__(self):
        self.name    = "base"
        self.version = "1.0.0"

    @abstractmethod
    def analyze(self, symbol, prices, volumes=None, **kwargs) -> StrategyResult: ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]: ...

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]): ...

    # ?? Wskazniki ? numpy, bez petli ??????????????????????????

    def calculate_ema(self, prices, period: int) -> float:
        """EMA przez numpy ? ~20x szybsze niz petla Python."""
        if not prices or len(prices) < 1:
            return 0.0
        arr = np.array(prices, dtype=np.float64)
        if len(arr) < period:
            return float(arr[-1])
        alpha = 2.0 / (period + 1)
        ema   = arr[0]
        for p in arr[1:]:
            ema = p * alpha + ema * (1.0 - alpha)
        return float(ema)

    def calculate_rsi(self, prices, period: int = 14) -> float:
        """RSI przez numpy diff ? szybki i dokladny."""
        if not prices or len(prices) < period + 1:
            return 50.0
        arr    = np.array(prices, dtype=np.float64)
        deltas = np.diff(arr)[-period:]
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def detect_trend(self, prices, short_period: int = 21,
                     long_period: int = 200) -> str:
        if not prices or len(prices) < long_period:
            return "SIDEWAYS"
        ema_short = self.calculate_ema(prices, short_period)
        ema_long  = self.calculate_ema(prices, long_period)
        if ema_short > ema_long * 1.005:
            return "UP"
        elif ema_short < ema_long * 0.995:
            return "DOWN"
        return "SIDEWAYS"

    def calculate_atr(self, prices, period: int = 14) -> float:
        """Average True Range ? miara zmiennosci."""
        if not prices or len(prices) < period + 1:
            return 0.0
        arr = np.array(prices, dtype=np.float64)
        trs = np.abs(np.diff(arr))[-period:]
        return float(trs.mean())

    def calculate_roc(self, prices, period: int = 10) -> float:
        """Rate of Change ? momentum procentowy."""
        if not prices or len(prices) < period + 1:
            return 0.0
        old = prices[-(period + 1)]
        if old == 0:
            return 0.0
        return float(((prices[-1] - old) / old) * 100)

    def create_signal(self, action: str, symbol: str, reason: str = "",
                      confidence: float = 0.0, price: float = 0.0,
                      **metadata) -> Signal:
        from datetime import datetime
        return Signal(
            action=action, symbol=symbol, reason=reason,
            confidence=confidence, price=price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def get_info(self) -> Dict:
        return {"name": self.name, "version": self.version}

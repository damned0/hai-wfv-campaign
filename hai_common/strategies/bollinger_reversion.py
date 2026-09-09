import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class Signal:
    """Signal object — wszystkie atrybuty które engine.py potrzebuje"""
    def __init__(self, symbol, action, entry, tp, sl, reason, confidence):
        self.symbol = symbol
        self.action = action  # "LONG" lub "SHORT"
        self.entry = entry
        self.price = entry  # engine szuka .price
        self.tp = tp
        self.sl = sl
        self.reason = reason
        self.confidence = confidence

class StrategyResult:
    def __init__(self, signal=None, indicators=None, score=0.0, opportunities=None):
        self.signal = signal
        self.indicators = indicators or {}
        self.score = score
        self.opportunities = opportunities or []

class BollingerReversionStrategy:
    """Band_reversion strategy — dotknięcie BB + świeca reaktywna"""
    def __init__(self, mode: str = "neutral", us_session: bool = False, 
                 no_monday: bool = False, min_quote_volume: float = 0,
                 adx_max: float = 999.0, tp_mode: str = "opposite"):
        self.name = "bollinger_reversion"
        self.top_n = 5
        self.min_history = 100
        self.bb_period = 20
        self.bb_std = 2.0
        self.atr_period = 14
        self.tp_mode = tp_mode
        self.sl_base_pct = 0.02
        self.use_us_session = us_session
        self.no_monday = no_monday
        self.min_quote_volume = min_quote_volume
        self.adx_max = adx_max
        self.mode = mode
        self._set_mode_params()

    def _set_mode_params(self):
        if self.mode == "aggressive":
            self.sl_base_pct = 0.015
        elif self.mode == "passive":
            self.sl_base_pct = 0.025
        else:
            self.sl_base_pct = 0.02

    def _atr_at(self, candles: List[Dict], period: int = 14) -> float:
        """ATR z candle'ów"""
        if len(candles) < period + 1:
            return 0.0
        try:
            trs = []
            for i in range(len(candles) - period, len(candles)):
                h = candles[i].get("high", 0)
                l = candles[i].get("low", 0)
                c = candles[i].get("close", 0)
                pc = candles[i-1].get("close", 0) if i > 0 else c
                tr = max(h - l, abs(h - pc), abs(l - pc))
                trs.append(tr)
            return float(np.mean(trs)) if trs else 0.0
        except:
            return 0.0

    def analyze(self, symbol: str, prices: List, volumes=None, **kwargs) -> StrategyResult:
        """Główna analiza — zwraca StrategyResult z sygnałem"""
        if not prices or len(prices) < self.min_history:
            return StrategyResult()

        in_position = kwargs.get("in_position", False)
        if in_position:
            return StrategyResult()

        # Konwersja do candle'ów (jeśli engine podał dict'y)
        candles = None
        if prices and isinstance(prices[0], dict):
            candles = prices
        else:
            # Fallback: stwórz candle'y z closes
            candles = [{"close": p, "high": p, "low": p, "open": p, "volume": 0} for p in prices]

        if len(candles) < self.bb_period:
            return StrategyResult()
        
        # Bollinger Bands
        closes = [c.get("close", 0) for c in candles[-self.bb_period:]]
        window = np.asarray(closes)
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            return StrategyResult()
        
        lower = mean - self.bb_std * std
        upper = mean + self.bb_std * std

        # Bieżąca i poprzednia świeca
        cur_candle = candles[-1]
        prev_candle = candles[-2] if len(candles) > 1 else candles[-1]

        cur_close = cur_candle.get("close", 0)
        cur_high = cur_candle.get("high", 0)
        cur_low = cur_candle.get("low", 0)
        prev_close = prev_candle.get("close", 0)

        # Sygnały: dotknięcie BB + świeca reaktywna
        long_signal = (cur_low <= lower) and (cur_close > prev_close)
        short_signal = (cur_high >= upper) and (cur_close < prev_close)

        if not (long_signal or short_signal):
            return StrategyResult()

        # Określ stronę
        side = "LONG" if long_signal else "SHORT"
        entry = cur_close

        # TP/SL — ATR-based
        atr = self._atr_at(candles, self.atr_period)
        if atr <= 0:
            atr = abs(entry * 0.02)

        if self.tp_mode == "opposite":
            tp = upper if side == "LONG" else lower
        else:  # mid
            tp = mean

        sl = entry - 1.5 * atr if side == "LONG" else entry + 1.5 * atr

        # Sanity
        if (side == "LONG" and tp <= entry) or (side == "SHORT" and tp >= entry):
            return StrategyResult()

        # Stwórz signal
        signal = Signal(
            symbol=symbol,
            action=side,
            entry=entry,
            tp=tp,
            sl=sl,
            reason=f"BB Reversion {side}",
            confidence=0.7
        )

        indicators = {
            "bb_lower": lower,
            "bb_upper": upper,
            "bb_mean": mean,
            "atr": atr,
        }

        return StrategyResult(signal=signal, indicators=indicators, score=70.0)

    def score_symbol(self, symbol: str, prices_1h: List, prices_4h: List,
                     prices_1d: List, volumes_1h: List = None) -> Tuple[float, str]:
        """TOP5 scoring — zwraca (score, action)"""
        if not prices_1h or len(prices_1h) < self.min_history:
            return 0.0, "NEUTRAL"

        # Konwersja
        candles = None
        if isinstance(prices_1h[0], dict):
            candles = prices_1h
        else:
            candles = [{"close": p, "high": p, "low": p, "open": p, "volume": 0} for p in prices_1h]

        if len(candles) < self.bb_period:
            return 0.0, "NEUTRAL"
        
        # BB
        closes = [c.get("close", 0) for c in candles[-self.bb_period:]]
        window = np.asarray(closes)
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.0, "NEUTRAL"
        
        lower = mean - self.bb_std * std
        upper = mean + self.bb_std * std

        cur_candle = candles[-1]
        prev_candle = candles[-2] if len(candles) > 1 else candles[-1]

        cur_close = cur_candle.get("close", 0)
        cur_high = cur_candle.get("high", 0)
        cur_low = cur_candle.get("low", 0)
        prev_close = prev_candle.get("close", 0)

        long_signal = (cur_low <= lower) and (cur_close > prev_close)
        short_signal = (cur_high >= upper) and (cur_close < prev_close)

        if not (long_signal or short_signal):
            return 0.0, "NEUTRAL"

        action = "LONG" if long_signal else "SHORT"
        return 70.0, action

    def get_parameters(self) -> Dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "tp_mode": self.tp_mode,
        }

    def set_parameters(self, params: Dict):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._set_mode_params()

from .registry import register_strategy

register_strategy("bollinger_reversion", lambda: BollingerReversionStrategy("neutral", us_session=False, no_monday=False, adx_max=25.0, tp_mode="opposite"))
register_strategy("bollinger_reversion_aggressive", lambda: BollingerReversionStrategy("aggressive"))
register_strategy("bollinger_reversion_neutral", lambda: BollingerReversionStrategy("neutral"))
register_strategy("bollinger_reversion_passive", lambda: BollingerReversionStrategy("passive"))

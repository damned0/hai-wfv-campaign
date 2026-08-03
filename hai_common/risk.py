# ===========================================
# HAI_EPV Engine ver.10 Final — core/risk.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: RiskManager (limity dzienne/pozycji, record_trade, cooldown po
# stratach), RiskResult. PnL = (exit-entry)*size_coins, leverage juz
# wliczony w size_coins (nie mnozymy ponownie).
# ===========================================
import logging
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from .config import config
from .state import state

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    allowed: bool
    reason: str = ""
    max_positions: int = 5
    current_positions: int = 0
    available_balance: float = 0.0
    risk_per_trade: float = 0.0


SL_COOLDOWN_MINUTES = 120  # blokada symbolu po SL

# Doktryna sekwencyjności:
#   SL bez piramidy → x0.1 (mocna redukcja — sygnał nie był potwierdzony)
#   SL z piramidą   → x0.5 (łagodniejsza — cena potwierdziła, potem zawróciła)
#   Pierwszy zysk   → reset do x1.0, odblokowanie piramidy
SIZE_SCALE_BY_LOSSES = {0: 1.0, 1: 0.10}  # 0 strat → 100%, 1+ strat → 10%
PYRAMID_SL_SCALE     = 0.50               # po SL z piramidą: 50% zamiast 10%


class RiskManager:

    def __init__(self):
        self.max_positions: int        = config.trading.max_positions
        self.max_leverage: int         = config.trading.leverage
        self.order_size_usdt: float    = config.trading.order_size_usdt
        self.max_daily_loss: float     = 50.0
        self.max_total_exposure: float = config.trading.max_total_exposure
        self._daily_pnl: float         = 0.0
        self._daily_trades: int        = 0
        self._last_reset_day: str      = ""
        self._consecutive_losses: int  = 0
        self._symbol_cooldown: Dict[str, datetime] = {}  # symbol → unblock_at
        self._pyramided_positions: Set[str] = set()      # symbole z aktywną piramidą
        self._pyramid_blocked: bool = False              # blok piramidy po pyramid SL

    # — Wejście ——————————————————————————————————————————————

    def check_entry(self, symbol: str, signal_type: str,
                    price: float, mode: str = None) -> RiskResult:
        mode           = mode or config.effective_mode
        if mode == "live" and self.is_circuit_breaker_active(mode):
            return RiskResult(allowed=False, reason="Circuit breaker active")
        if self.get_consecutive_losses() >= config.MAX_CONSECUTIVE_LOSSES:
            return RiskResult(allowed=False, reason="Max consecutive losses")
        if self.is_symbol_in_cooldown(symbol):
            until = self._symbol_cooldown.get(symbol)
            remaining = int((until - datetime.now(timezone.utc)).total_seconds() / 60) if until else 0
            return RiskResult(allowed=False, reason=f"SL cooldown {symbol}: jeszcze {remaining} min")
        open_positions = state.get_open_positions(mode)
        current_count  = len(open_positions)

        if current_count >= self.max_positions:
            return RiskResult(
                allowed=False,
                reason=f"Max pozycji: {current_count}/{self.max_positions}",
                max_positions=self.max_positions,
                current_positions=current_count,
            )

        if any(p["symbol"] == symbol for p in open_positions):
            return RiskResult(
                allowed=False,
                reason=f"Pozycja na {symbol} juz otwarta",
                current_positions=current_count,
            )

        balance = state.get_balance(mode)
        if balance < self.order_size_usdt:
            return RiskResult(
                allowed=False,
                reason=f"Za maly balans: {balance:.2f} < {self.order_size_usdt}",
                available_balance=balance,
                current_positions=current_count,
            )

        self._maybe_reset_daily()
        if self._daily_pnl <= -self.max_daily_loss:
            return RiskResult(
                allowed=False,
                reason=f"Dzienny limit strat: {self._daily_pnl:.2f} USD",
                current_positions=current_count,
            )

        total_exposure = sum(p.get("size_usdt", 0) for p in open_positions)
        if total_exposure + self.order_size_usdt > self.max_total_exposure:
            return RiskResult(
                allowed=False,
                reason=f"Max ekspozycja: {total_exposure:.2f} + {self.order_size_usdt} > {self.max_total_exposure}",
                current_positions=current_count,
            )

        return RiskResult(
            allowed=True,
            reason="OK",
            max_positions=self.max_positions,
            current_positions=current_count,
            available_balance=balance,
            risk_per_trade=self.order_size_usdt,
        )

    # ?? Wyjscie ????????????????????????????????????????????????

    def check_exit(self, symbol: str, mode: str = None) -> RiskResult:
        mode = mode or config.effective_mode
        if any(p["symbol"] == symbol for p in state.get_open_positions(mode)):
            return RiskResult(allowed=True, reason="OK")
        return RiskResult(allowed=False, reason=f"Brak pozycji dla {symbol}")

    # ?? Kalkulacje ?????????????????????????????????????????????

    def calculate_position_size(self, price: float,
                                leverage: int = None) -> Tuple[float, float]:
        """
        size_coins = (size_usdt * leverage) / price
        Leverage jest wbudowany w size_coins.
        PnL = (exit - entry) * size_coins — bez dodatkowego mnożenia.
        Rozmiar redukowany po kolejnych stratach (SIZE_SCALE_BY_LOSSES).
        """
        leverage  = leverage or self.max_leverage
        losses    = self.get_consecutive_losses()
        if losses > 0 and self._pyramid_blocked:
            scale = PYRAMID_SL_SCALE  # x0.5 po pyramid SL (potwierdzony setup który zawrócił)
            logger.info(f"Sekwencja [PYRAMID SL]: {losses} strat → x{scale:.1f} | size_usdt={round(self.order_size_usdt * scale, 2)}")
        else:
            scale = SIZE_SCALE_BY_LOSSES.get(losses, SIZE_SCALE_BY_LOSSES[max(SIZE_SCALE_BY_LOSSES)])
            if scale < 1.0:
                logger.info(f"Sekwencja: {losses} strat → x{scale:.1f} | size_usdt={round(self.order_size_usdt * scale, 2)} (doktryna x0.1)")
        size_usdt  = round(self.order_size_usdt * scale, 2)
        size_coins = (size_usdt * leverage) / price if price > 0 else 0.0
        return round(size_coins, 8), size_usdt

    def calculate_pnl(self, entry_price: float, exit_price: float,
                      size_coins: float) -> Tuple[float, float]:
        """
        Uwaga: size_coins zawiera juz dzwignie.
        pnl_pct = zmiana % ceny (bez mnozenia przez leverage ? juz w size_coins).
        """
        pnl     = (exit_price - entry_price) * size_coins
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
        return round(pnl, 2), round(pnl_pct, 2)

    # ?? Dzienne statystyki ?????????????????????????????????????

    def record_trade(self, pnl: float):
        self._maybe_reset_daily()
        self._daily_pnl    += pnl
        self._daily_trades += 1
        logger.debug(f"Trade zapisany | PnL: {pnl:.2f} | Dzienny: {self._daily_pnl:.2f}")

    def _maybe_reset_daily(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_day:
            self._daily_pnl      = 0.0
            self._daily_trades   = 0
            self._last_reset_day = today
            logger.info(f"Reset dzienny statystyk | {today}")

    def get_daily_stats(self) -> Dict:
        self._maybe_reset_daily()
        return {
            "daily_pnl":      round(self._daily_pnl, 2),
            "daily_trades":   self._daily_trades,
            "open_positions": len(state.get_open_positions()),
            "max_positions":  self.max_positions,
            "order_size":     self.order_size_usdt,
            "max_daily_loss": self.max_daily_loss,
        }

    # — Cooldown per symbol po SL ————————————————————————————

    def record_sl(self, symbol: str, minutes: int = SL_COOLDOWN_MINUTES):
        """Rejestruje SL hit: wpisuje stratę + blokuje symbol + ustawia pyramid_blocked jeśli była piramida."""
        had_pyramid = symbol in self._pyramided_positions
        self._pyramided_positions.discard(symbol)
        if had_pyramid:
            self._pyramid_blocked = True
            logger.info(f"SL z piramidą: {symbol} → pyramid_blocked=True, next_scale=x{PYRAMID_SL_SCALE}")
        self.record_consecutive_loss()
        self._symbol_cooldown[symbol] = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        logger.info(f"SL cooldown: {symbol} zablokowany na {minutes} min | "
                    f"consecutive_losses={self.get_consecutive_losses()} | pyramid_was={had_pyramid}")

    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        until = self._symbol_cooldown.get(symbol)
        if until is None:
            return False
        if datetime.now(timezone.utc) >= until:
            self._symbol_cooldown.pop(symbol, None)
            return False
        return True

    def get_cooldown_status(self) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        return {
            sym: f"{int((until - now).total_seconds() / 60)} min"
            for sym, until in list(self._symbol_cooldown.items())
            if until > now
        }

    def update_settings(self, settings: Dict):
        allowed = ["max_positions", "max_leverage", "order_size_usdt",
                   "max_daily_loss", "max_total_exposure"]
        for k in allowed:
            if k in settings:
                setattr(self, k, settings[k])
                logger.info(f"Risk setting: {k} = {settings[k]}")



    def is_circuit_breaker_active(self, mode: str = "paper") -> bool:
        if mode == "paper": return False
        balance = state.get_balance(mode)
        if balance <= 0: return True
        drawdown_pct = (1 - balance / 1000.0) * 100
        return drawdown_pct >= config.MAX_DRAWDOWN_PCT

    # — Pyramid zarządzanie ————————————————————————————————————

    def record_pyramid_open(self, symbol: str):
        """Rejestruje otwarcie piramidy — potrzebne do śledzenia pyramid SL."""
        self._pyramided_positions.add(symbol)

    def clear_pyramid(self, symbol: str):
        """Czyści piramidę przy zamknięciu pozycji (TP lub ręcznie)."""
        self._pyramided_positions.discard(symbol)

    def is_pyramid_active(self, symbol: str) -> bool:
        return symbol in self._pyramided_positions

    def is_pyramid_blocked(self) -> bool:
        """Czy piramida jest zablokowana (po SL z piramidą)?"""
        return self._pyramid_blocked

    # — Straty sekwencyjne ————————————————————————————————————

    def record_consecutive_loss(self):
        self._consecutive_losses += 1

    def reset_consecutive_losses(self):
        self._consecutive_losses = 0
        self._pyramid_blocked = False  # odblokuj piramidę po pierwszym zysku

    def get_consecutive_losses(self) -> int:
        return self._consecutive_losses
risk = RiskManager()

# ===========================================
# HAI_EPV Engine ver.10 Final — core/state.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: open_position()/close_position() (PnL side-aware LONG/SHORT,
# snapshot cech przy openie), get_open_positions(), add_log()/get_logs()
# (system/trading/ai kategorie), zarzadzanie saldem/DB session.
# ===========================================

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from .database import (
    get_session, init_db, Position,
    SystemLog, TradingLog, AILog
)
from .config import config

logger = logging.getLogger(__name__)


def _ledger(action: str, pos) -> None:
    """Append-only zapis do centralnego ledgera trade'ow (audyt 2026-08-01).
    Import lokalny + pelny try/except: awaria ledgera NIGDY nie moze
    przerwac tradingu. record_* sam w sobie tez nie rzuca."""
    try:
        from . import trade_ledger
        from .config import INSTANCE
        if action == "open":
            trade_ledger.record_open(INSTANCE, pos)
        elif action == "closed":
            trade_ledger.record_closed(INSTANCE, pos)
    except Exception as e:
        logger.warning(f"ledger hook ({action}) pominiety: {e}")


class StateManager:
    """Centralny zarzadca stanu ProjektHAI"""

    def __init__(self):
        init_db()
        self._session: Optional[Session] = None

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = get_session()
        return self._session

    def _exec(self, func, *args, **kwargs):
        """Bezpieczne wykonanie operacji na bazie z zamknieciem sesji"""
        try:
            result = func(*args, **kwargs)
            self.session.commit()
            return result
        except Exception as e:
            self.session.rollback()
            logger.error(f"DB error: {e}")
            raise
        finally:
            self.close()

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    # ============================================
    # POZYCJE
    # ============================================

    def open_position(
        self, symbol: str, side: str, entry_price: float,
        size_coins: float, size_usdt: float, leverage: int = 5,
        strategy: str = "manual", reason: str = None,
        mode: str = None,
        features_dict: Optional[Dict[str, float]] = None,
        tp_price: float = None, sl_price: float = None,
    ) -> int:
        """Otworz pozycje + zapisz snapshot features (jesli podany).

        Args:
            features_dict: 16-elementowy dict z core.features.build_features_live().
                          Bedzie zapisany jako JSON do Position.features_json.
                          Pozwala na deterministyczny retrening modeli.
        """
        mode = mode or config.effective_mode
        features_json = None
        if features_dict:
            try:
                features_json = json.dumps(features_dict, sort_keys=True)
            except Exception as e:
                logger.warning(f"open_position: nie udalo sie zserializowac features: {e}")

        pos = Position(
            symbol=symbol, side=side, mode=mode,
            exchange=config.DEFAULT_EXCHANGE,
            entry_price=entry_price, size_coins=size_coins,
            size_usdt=size_usdt, leverage=leverage,
            status="open", strategy=strategy, reason=reason,
            features_json=features_json,
            tp_price=tp_price, sl_price=sl_price,
        )
        self.session.add(pos)
        self.session.commit()
        self.session.refresh(pos)
        _ledger("open", pos)

        currency = "USDT" if mode == "live" else "vUSDT"
        self.add_log("trading", "SIGNAL", symbol=symbol,
                     action="OPEN", mode=mode,
                     message=f"OPEN {side} {symbol} @ {entry_price} | Size: {size_usdt} {currency}")

        from .events import bus, Events
        bus.emit(Events.POSITION_OPENED, {
            "symbol": symbol, "side": side, "entry": entry_price,
            "size_usdt": size_usdt, "mode": mode
        })
        return pos.id

    def pyramid_add(self, symbol: str, add_size_coins: float, add_size_usdt: float,
                    current_price: float, mode: str = None) -> bool:
        """Dokłada do istniejącej pozycji (pyramid). Aktualizuje avg entry i rozmiar."""
        mode = mode or config.effective_mode
        pos = (
            self.session.query(Position)
            .filter(Position.symbol == symbol)
            .filter(Position.status == "open")
            .filter(Position.mode == mode)
            .first()
        )
        if not pos:
            return False
        total_coins = pos.size_coins + add_size_coins
        avg_entry = (pos.entry_price * pos.size_coins + current_price * add_size_coins) / total_coins
        pos.size_coins = round(total_coins, 8)
        pos.size_usdt = round(pos.size_usdt + add_size_usdt, 2)
        pos.entry_price = round(avg_entry, 8)
        pos.reason = (pos.reason or "") + f" | PYR@{current_price:.4f}"
        self.session.commit()
        _ledger("open", pos)  # refresh snapshotu opens po pyramidzie
        currency = "USDT" if mode == "live" else "vUSDT"
        self.add_log("trading", "SIGNAL", symbol=symbol,
                     action="PYRAMID", mode=mode,
                     message=f"PYRAMID {symbol} +{add_size_usdt:.1f}{currency} @ {current_price} | avg_entry={avg_entry:.6f}")
        from .events import bus, Events
        bus.emit(Events.POSITION_OPENED, {
            "symbol": symbol, "side": pos.side, "entry": current_price,
            "size_usdt": add_size_usdt, "mode": mode, "pyramid": True,
        })
        return True

    def close_position(
        self, symbol: str, exit_price: float,
        mode: str = None, reason: str = None
    ) -> Optional[Dict]:
        mode = mode or config.effective_mode
        pos = (
            self.session.query(Position)
            .filter(Position.symbol == symbol)
            .filter(Position.status == "open")
            .filter(Position.mode == mode)
            .first()
        )
        if not pos:
            return None

        # FIX v5: poprawny PnL z obsluga SHORT + bez podwojnego leverage
        # size_coins juz zawiera leverage (risk.py)
        if pos.side == "SHORT":
            pnl = (pos.entry_price - exit_price) * pos.size_coins
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
        else:  # LONG
            pnl = (exit_price - pos.entry_price) * pos.size_coins
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100

        # KOSZTY wstawione do pnl pozycji (2026-08-02): fee taker x2 (open+close)
        # + slippage wejscia + slippage wyjscia. Nawias notional = size_usdt(+pyramid).
        # pnl jest podawany w USDT (juz * size_coins), koszty liczymy na notional usdt.
        try:
            notional = float(pos.size_usdt)  # usdt zainwestowane (okolo stala)
            fee = notional * config.trading.fee_taker * 2
            slip = notional * (config.trading.slippage_entry + config.trading.slippage_exit)
            cost = fee + slip
            pnl -= cost
        except Exception as e:
            logger.warning(f"close_position: nie policzono kosztow ({e}) — pnl bez kosztow")

        pos.exit_price = exit_price
        pos.pnl = round(pnl, 2)
        pos.pnl_pct = round(pnl_pct, 2)
        pos.status = "closed"
        pos.exit_time = datetime.now(timezone.utc)
        pos.reason = reason
        self.session.commit()
        _ledger("closed", pos)

        currency = "USDT" if mode == "live" else "vUSDT"
        self.add_log("trading", "SIGNAL", symbol=symbol,
                     action="CLOSE", mode=mode, pnl=round(pnl, 2),
                     message=f"CLOSE {symbol} @ {exit_price} | PnL: {pnl:.2f} {currency} ({pnl_pct:.1f}%)")

        from .events import bus, Events
        bus.emit(Events.POSITION_CLOSED, {
            "symbol": symbol, "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2), "mode": mode
        })
        return {"symbol": symbol, "entry": pos.entry_price,
                "exit": exit_price, "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2)}

    def close_all_positions(self, current_prices: Dict[str, float],
                            mode: str = None) -> List[Dict]:
        mode = mode or config.effective_mode
        results = []
        for pos in self.get_open_positions(mode):
            symbol = pos["symbol"]
            if symbol in current_prices:
                result = self.close_position(symbol, current_prices[symbol], mode, "close_all")
                if result:
                    results.append(result)
        return results

    def emergency_close_live(self, current_prices: Dict[str, float]):
        results = self.close_all_positions(current_prices, mode="live")
        config.ai.trade_enabled = False
        self.add_log("system", "WARNING", component="emergency",
                     message="Awaryjne zamkniecie LIVE | AI TRADE OFF")
        from .events import bus, Events
        bus.emit(Events.PANIC_MODE, {"reason": "emergency_close_live"})
        return results

    def emergency_full_stop(self, current_prices: Dict[str, float]):
        self.close_all_positions(current_prices, mode="live")
        self.close_all_positions(current_prices, mode="paper")
        config.TRADE_ENABLED = False
        config.AI_ENABLED = False
        config.AI_TRADE_ENABLED = False
        config.AI_LEARN_ENABLED = False
        self.add_log("system", "CRITICAL", component="emergency",
                     message="AWARYJNY STOP | Wszystko OFF")
        from .events import bus, Events
        bus.emit(Events.PANIC_MODE, {"reason": "emergency_full_stop"})
        return {"status": "full_stop"}

    def get_open_positions(self, mode: str = None) -> List[Dict]:
        query = self.session.query(Position).filter(Position.status == "open")
        if mode:
            query = query.filter(Position.mode == mode)
        return [{
            "symbol": p.symbol, "side": p.side, "entry_price": p.entry_price,
            "buy_price": p.entry_price, "open_price": p.entry_price,
            "size_coins": p.size_coins, "size_usdt": p.size_usdt,
            "leverage": p.leverage,
            "entry_time": p.entry_time.isoformat() if p.entry_time else None,
            "strategy": p.strategy, "pnl_pct": p.pnl_pct, "mode": p.mode,
            "tp_price": p.tp_price, "sl_price": p.sl_price,
        } for p in query.all()]

    def get_closed_pnl(self, limit: int = 20, mode: str = None) -> List[Dict]:
        query = self.session.query(Position).filter(Position.status == "closed")
        if mode:
            query = query.filter(Position.mode == mode)
        query = query.order_by(Position.exit_time.desc()).limit(limit)
        return [{
            "symbol": p.symbol, "side": p.side,
            # cena otwarcia / kupna = entry_price; cena zamknięcia / sprzedaży = exit_price
            "entry": p.entry_price, "exit": p.exit_price,
            "buy_price": p.entry_price, "sell_price": p.exit_price,
            "open_price": p.entry_price, "close_price": p.exit_price,
            "size_usdt": p.size_usdt, "size_coins": p.size_coins, "leverage": p.leverage,
            "pnl": p.pnl, "pnl_pct": p.pnl_pct, "mode": p.mode,
            "entry_time": p.entry_time.isoformat() if p.entry_time else None,
            "exit_time": p.exit_time.isoformat() if p.exit_time else None,
        } for p in query.all()]

    # ============================================
    # LOGI
    # ============================================

    def add_log(self, log_type: str, level: str, **kwargs):
        try:
            if log_type == "system":
                log = SystemLog(level=level, component=kwargs.get("component", "core"),
                                message=kwargs.get("message", ""))
            elif log_type == "trading":
                log = TradingLog(level=level, symbol=kwargs.get("symbol"),
                                 action=kwargs.get("action"),
                                 mode=kwargs.get("mode", config.effective_mode),
                                 message=kwargs.get("message", ""),
                                 pnl=kwargs.get("pnl", 0.0))
            elif log_type == "ai":
                # AILog nie ma kolumny 'level' - zeby filtr ERROR dzialal w
                # przekroju przez wszystkie kategorie (TRAINING/WFV/BACKTEST/
                # SCREENING/MODEL_LOAD), prefiksujemy tresc centralnie tutaj
                # zamiast w kazdym call-site osobno (audyt 2026-07-04).
                msg = kwargs.get("message", "")
                if level == "ERROR" and not msg.startswith("[ERROR]"):
                    msg = f"[ERROR] {msg}"
                log = AILog(model_name=kwargs.get("model_name", "unknown"),
                            event=kwargs.get("event", "training"),
                            accuracy=kwargs.get("accuracy", 0.0),
                            samples_count=kwargs.get("samples_count", 0),
                            message=msg)
            else:
                return
            self.session.add(log)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Log error: {e}")

    def get_logs(self, log_type: str = "system", limit: int = 50) -> List[Dict]:
        """FIX v6.0: AILog nie ma pola 'level' - mapujemy z 'event'."""
        try:
            if log_type == "system":
                query = self.session.query(SystemLog).order_by(SystemLog.timestamp.desc())
                logs = query.limit(limit).all()
                return list(reversed([{
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                    "level": l.level, "message": l.message,
                } for l in logs]))
            elif log_type == "trading":
                query = self.session.query(TradingLog).order_by(TradingLog.timestamp.desc())
                logs = query.limit(limit).all()
                return list(reversed([{
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                    "level": l.level, "message": l.message,
                    "symbol": l.symbol, "action": l.action,
                    "pnl": l.pnl, "mode": l.mode,
                } for l in logs]))
            elif log_type == "ai":
                query = self.session.query(AILog).order_by(AILog.timestamp.desc())
                logs = query.limit(limit).all()
                # FIX v6.0: AILog ma 'event' zamiast 'level'
                return list(reversed([{
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                    "level": (l.event or "INFO").upper(),  # mapowanie event -> level
                    "model_name": l.model_name,
                    "event": l.event,
                    "accuracy": l.accuracy,
                    "samples_count": l.samples_count,
                    "message": l.message,
                } for l in logs]))
            else:
                return []
        except Exception as e:
            logger.error(f"Get logs error ({log_type}): {e}")
            return []

    # ============================================
    # BALANS
    # ============================================

    def get_balance(self, mode: str = None) -> float:
        """Saldo paper = kapital startowy + suma PnL zamknietych pozycji.
        Kapital startowy byl ZASZYTY (500.0) w dwoch miejscach — teraz z env
        PAPER_START_BALANCE (2026-07-14, reset stanu instancji na 200 USD)."""
        mode = mode or config.effective_mode
        start = float(os.environ.get("PAPER_START_BALANCE", "200"))
        # TODO: dla live pobierac z exchange.get_balance()
        try:
            positions = self.session.query(Position).filter(
                Position.mode == mode, Position.status == "closed"
            ).all()
            total_pnl = sum(p.pnl for p in positions)
            return start + total_pnl
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return start

    def get_paper_balance(self) -> float:
        return self.get_balance("paper")

    def get_live_balance(self) -> float:
        return self.get_balance("live")

    def get_setting(self, key: str, default: str = "0") -> str:
        return str(default)


state = StateManager()

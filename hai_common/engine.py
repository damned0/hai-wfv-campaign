# ===========================================
# HAI_EPV Engine ver.10 Final — core/engine.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - Engine.start()/stop() - cykl zycia glownej petli tradingowej
# - _main_loop() - co interval: score symboli (ProcessPoolExecutor), sygnaly AI,
#   otwieranie/zamykanie pozycji, logi "Analyze"/"Working..."
# - _tp_sl_monitor() - niezalezny monitor TP/SL dla otwartych pozycji
# - _compute_features() - wrapper na core.features.build_features_live
# - _process_signal() - egzekucja sygnalu (order, snapshot cech, pyramid)
# ===========================================
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .config import config
from .state import state
from .events import bus, Events
from .risk import risk
from .ensemble import ensemble
from .features import build_features_live
from .symbols import TRADING_SYMBOLS, is_supported

import exchanges.bitget
from .strategies import ai_strategy  # v6.3: zarejestruj AIStrategy w registry
from exchanges.registry import get_exchange
from .strategies.registry import get_strategy
from .strategies.bollinger_reversion import BollingerReversionStrategy

logger = logging.getLogger(__name__)

_FETCH_SEMAPHORE = asyncio.Semaphore(8)
_PROCESS_POOL = ProcessPoolExecutor(max_workers=2)


def _score_symbols_worker(strategy_params: dict, symbols: List[str],
                          data_1h: Dict, data_4h: Dict, data_1d: Dict,
                          volumes_1h: Dict) -> List[Dict]:
    """Top-level dla ProcessPoolExecutor (musi byc pickleable)."""
    # v6.4: worker uzywa registry zamiast hardcodowanej strategii
    # zeby ai_strategy/momentum/bollinger tez dzialaly w paraleli
    from .strategies import ai_strategy    # rejestracja AI strategy w worker procesie
    from .strategies.registry import get_strategy
    strategy_name = strategy_params.get("name", "ai_strategy")
    strategy = get_strategy(strategy_name)
    if strategy is None:
        # Fallback gdy registry nie ma tej nazwy
        strategy = get_strategy("ai_strategy")
    # v6.4 DEBUG: log ensemble status w workerze
    import logging as _logging
    _log = _logging.getLogger("worker")
    if hasattr(strategy, '_get_ensemble'):
        try:
            ens = strategy._get_ensemble()
            _log.info(f"WORKER ensemble: active={ens.active}, models={list(ens.models.keys())}, name={strategy.name}")
        except Exception as e:
            _log.error(f"WORKER ensemble error: {e}")
    else:
        _log.info(f"WORKER strategy={strategy.name} (no ensemble)")

    scored = []
    neutral_count = 0
    for sym in symbols:
        p1 = data_1h.get(sym, [])
        if len(p1) < strategy.min_history:
            continue
        p4 = data_4h.get(sym, [])
        pd = data_1d.get(sym, [])
        vol = volumes_1h.get(sym)
        score, action = strategy.score_symbol(sym, p1, p4, pd, vol)
        if action in ("LONG", "SHORT"):
            scored.append({"symbol": sym, "score": score, "action": action})
        else:
            neutral_count += 1
    _log.info(f"WORKER scoring done: {len(scored)} signals, {neutral_count} NEUTRAL")
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:strategy.top_n]


class TradingEngine:

    def __init__(self):
        self._running = False
        self._exchange = None
        self._strategy = None
        self._prices: Dict[str, float] = {}
        self._price_history_1h: Dict[str, List] = {}
        self._price_history_4h: Dict[str, List] = {}
        self._price_history_1d: Dict[str, List] = {}
        self._top_symbols: List[str] = []
        self._loop_interval: int = config.trading.loop_interval_sec
        self._loop_interval_min: int = 15
        self._loop_interval_max: int = 300
        self._market_activity: float = 0.0
        # Pyramid tracking
        self._pyramided: Set[str] = set()
        self._position_entry_time: Dict[str, datetime] = {}
        # Derivatives cache: symbol → (deriv_dict, fetched_at)
        self._deriv_cache: Dict[str, Tuple[dict, datetime]] = {}
        self._deriv_cache_ttl: int = 28800  # 8h (funding rate changes every 8h)

    async def start(self):
        logger.info("Engine startuje...")
        state.add_log("system", "INFO", component="engine",
                      message="Engine startuje (ver.10)")
        bus.emit(Events.ENGINE_STARTED, {})

        self._exchange = get_exchange()
        if not self._exchange:
            logger.error("Brak gieldy!")
            state.add_log("system", "ERROR", component="engine",
                          message="Brak gieldy - engine nie startuje")
            return

        await self._exchange.connect()
        self._strategy = get_strategy("ai_strategy")

        await self._load_top_symbols()
        await self._load_history()
        await self._exchange.start_ws(self._top_symbols[:30], self._on_ticker)

        self._running = True
        asyncio.create_task(self._main_loop())
        asyncio.create_task(self._tp_sl_monitor())
        logger.info(f"Engine dziala | strategia: {self._strategy.name} "
                    f"| symbole: {len(self._top_symbols)} "
                    f"| interval: {self._loop_interval}s")

    async def stop(self):
        logger.info("Engine zatrzymuje sie...")
        self._running = False
        if self._exchange:
            await self._exchange.stop_ws()
            await self._exchange.disconnect()
        state.add_log("system", "INFO", component="engine",
                      message="Engine zatrzymany")
        bus.emit(Events.ENGINE_STOPPED, {})

    async def _main_loop(self):
        await asyncio.sleep(10)
        while self._running:
            try:
                if not config.TRADE_ENABLED:
                    logger.info("Working...")
                    await asyncio.sleep(10)
                    continue

                logger.info("Analyze")
                await self._load_history()
                self._loop_interval = self._calc_interval()

                data_1h, volumes_1h = self._prepare_data("1H")
                data_4h, _ = self._prepare_data("4H")
                data_1d, _ = self._prepare_data("1D")

                strategy_params = {"mode": getattr(self._strategy, "mode", "neutral"), "name": self._strategy.name if self._strategy else "ai_strategy"}
                symbols_to_score = self._top_symbols[:50]

                loop = asyncio.get_event_loop()
                top5 = await loop.run_in_executor(
                    _PROCESS_POOL,
                    _score_symbols_worker,
                    strategy_params,
                    symbols_to_score,
                    data_1h, data_4h, data_1d, volumes_1h,
                )

                logger.info(f"TOP5: {[(t['symbol'], t['score']) for t in top5]}")
                bus.emit(Events.TOP5_UPDATE, {"top5": top5})

                open_map: Dict[str, Dict] = {
                    p["symbol"]: p for p in state.get_open_positions()
                }

                for item in top5:
                    symbol = item["symbol"]
                    prices = data_1h.get(symbol, [])
                    if len(prices) < self._strategy.min_history:
                        continue
                    in_position = symbol in open_map
                    # high/low/timestamp do analyze (2026-08-07): bez nich sciezka
                    # SYGNALU liczyla cechy na samych close — a to ona decyduje
                    # o wejsciu w pozycje. Patrz _ohlc_aux.
                    _h, _l, _t = self._ohlc_aux(symbol, prices)
                    result = self._strategy.analyze(
                        symbol, prices,
                        volumes=volumes_1h.get(symbol),
                        prices_4h=data_4h.get(symbol, []),
                        prices_1d=data_1d.get(symbol, []),
                        in_position=in_position,
                        entry_price=open_map[symbol]["entry_price"] if in_position else 0.0,
                        highs_1h=_h, lows_1h=_l, timestamps_1h=_t,
                    )
                    await self._process_signal(symbol, result,
                                               prices_1h=prices,
                                               prices_4h=data_4h.get(symbol, []),
                                               prices_1d=data_1d.get(symbol, []),
                                               volumes_1h=volumes_1h.get(symbol, []))

                await self._check_pyramid_positions()

            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
            logger.info("Working...")
            await asyncio.sleep(self._loop_interval)

    async def _tp_sl_monitor(self):
        await asyncio.sleep(15)
        while self._running:
            try:
                for mode in ["paper", "live"]:
                    if mode == "live" and config.effective_mode != "live":
                        continue
                    for pos in state.get_open_positions(mode):
                        symbol = pos["symbol"]
                        # FIX 2026-08-02: WS streamuje tylko top30 symboli, więc wyrazy
                        # poza top30 NIE maja ceny w _prices -> TP/SL nigdy by ich nie
                        # zamknal. Fallback: ostatni close z _price_history_1h (laduje
                        # wszystkie symbole z _load_history).
                        cur = self._prices.get(symbol)
                        if cur is None:
                            h1 = self._price_history_1h.get(symbol, [])
                            if not h1 or h1[-1].get("close") is None:
                                continue  # brak ceny - nie mozna ocenic TP/SL
                            cur = h1[-1]["close"]
                        entry = pos["entry_price"]
                        # FIX v5: pnl_pct z obsluga SHORT
                        if pos["side"] == "SHORT":
                            pnl_pct = ((entry - cur) / entry) * 100
                        else:  # LONG
                            pnl_pct = ((cur - entry) / entry) * 100
                        # Per-position ATR-based TP/SL (z strategii) z fallbackiem
                        # do globalnych % z config (TAKE_PROFIT_PCT/STOP_LOSS_PCT).
                        tp = sl = None
                        if pos.get("tp_price") and pos.get("sl_price"):
                            if pos["side"] == "SHORT":
                                tp = ((entry - pos["tp_price"]) / entry) * 100
                                sl = ((pos["sl_price"] - entry) / entry) * 100
                            else:
                                tp = ((pos["tp_price"] - entry) / entry) * 100
                                sl = ((entry - pos["sl_price"]) / entry) * 100
                        if not tp:
                            tp = config.trading.take_profit_pct
                        if not sl:
                            sl = config.trading.stop_loss_pct

                        # 2026-08-04: partial-TP + trailing (koncept Hauzera,
                        # RAPORT_EDGE.md §12, backtest PF 2.45->5.21, WR 57%->81%).
                        # Faza 1 (pelna pozycja): 50% do TP -> zamknij 75% pozycji.
                        # Faza 2 (partial_closed): 75% do TP -> aktywuj trailing na
                        # resztce (25%). Faza 3 (trailing_active): cel 150% oryg. TP,
                        # albo trailing-stop = 85% szczytowego pnl_pct od aktywacji.
                        # Istniejace juz otwarte pozycje maja partial_closed/trailing_
                        # active domyslnie False -> od razu wchodza w Faze 1.
                        if pos.get("trailing_active"):
                            peak_price = pos.get("peak_price") or cur
                            is_new_peak = (cur > peak_price) if pos["side"] != "SHORT" else (cur < peak_price)
                            if is_new_peak:
                                peak_price = cur
                                state.update_position_meta(symbol, mode, peak_price=peak_price)
                            if pos["side"] == "SHORT":
                                peak_pnl_pct = ((entry - peak_price) / entry) * 100
                            else:
                                peak_pnl_pct = ((peak_price - entry) / entry) * 100
                            trail_stop_pct = peak_pnl_pct * 0.85
                            full_target_pct = tp * 1.5

                            if pnl_pct >= full_target_pct:
                                result = state.close_position(symbol, cur, mode, f"TP150 +{pnl_pct:.1f}%")
                                if result:
                                    risk.record_trade(result["pnl"])
                                    risk.reset_consecutive_losses()
                                    risk.clear_pyramid(symbol)
                                    self._pyramided.discard(symbol)
                                    self._position_entry_time.pop(symbol, None)
                            elif pnl_pct <= trail_stop_pct:
                                result = state.close_position(symbol, cur, mode, f"TRAIL {pnl_pct:.1f}% (peak {peak_pnl_pct:.1f}%)")
                                if result:
                                    risk.record_trade(result["pnl"])
                                    risk.clear_pyramid(symbol)
                                    self._pyramided.discard(symbol)
                                    self._position_entry_time.pop(symbol, None)
                            continue

                        if pos.get("partial_closed"):
                            if pnl_pct >= tp * 0.75:
                                state.update_position_meta(symbol, mode, trailing_active=True, peak_price=cur)
                            elif pnl_pct <= -sl:
                                result = state.close_position(symbol, cur, mode, f"SL {abs(pnl_pct):.1f}% (po partial)")
                                if result:
                                    risk.record_sl(symbol)
                                    risk.record_trade(result["pnl"])
                                    self._pyramided.discard(symbol)
                                    self._position_entry_time.pop(symbol, None)
                            continue

                        if pnl_pct >= tp * 0.5:
                            result = state.partial_close_position(symbol, cur, 0.75, mode, f"PARTIAL +{pnl_pct:.1f}% (50% do TP)")
                            if result:
                                risk.record_trade(result["pnl"])
                        elif pnl_pct <= -sl:
                            result = state.close_position(symbol, cur, mode, f"SL {abs(pnl_pct):.1f}%")
                            if result:
                                risk.record_sl(symbol)   # obsługuje pyramid_blocked wewnętrznie
                                risk.record_trade(result["pnl"])
                                self._pyramided.discard(symbol)
                                self._position_entry_time.pop(symbol, None)
            except Exception as e:
                logger.error(f"TP/SL monitor error: {e}")
            await asyncio.sleep(5)

    async def _load_top_symbols(self):
        """Laduje DOKLADNIE wszystkie symbole z TRADING_SYMBOLS (whitelist) bez filtra volume."""
        # v6.4: whitelist JEST naszym filterem, nie potrzebujemy drugiego po volume
        self._top_symbols = list(TRADING_SYMBOLS)
        logger.info(f"Zaladowano {len(self._top_symbols)} symboli z PAPER_WHITELIST")

    async def _load_history(self):
        symbols = self._top_symbols[:50]

        async def _load_one(symbol: str):
            async with _FETCH_SEMAPHORE:
                for tf, store in [
                    ("1H", self._price_history_1h),
                    ("4H", self._price_history_4h),
                    ("1D", self._price_history_1d),
                ]:
                    try:
                        ohlcv = await self._exchange.fetch_ohlcv(symbol, tf, 300)
                        if ohlcv:
                            store[symbol] = self._parse_candles(ohlcv)
                        else:
                            logger.debug(f"Brak danych {tf} dla {symbol}")
                    except Exception as e:
                        logger.debug(f"Blad fetch {tf} {symbol}: {e}")

        await asyncio.gather(*[_load_one(s) for s in symbols])

    @staticmethod
    def _parse_candles(raw: list) -> List[Dict]:
        result = []
        for c in raw:
            if isinstance(c, (list, tuple)) and len(c) >= 6:
                result.append({
                    "timestamp": c[0], "open": c[1], "high": c[2],
                    "low": c[3], "close": c[4], "volume": c[5],
                })
            elif isinstance(c, dict):
                result.append(c)
        return result

    def _prepare_data(self, tf: str) -> Tuple[Dict, Dict]:
        store = {
            "1H": self._price_history_1h,
            "4H": self._price_history_4h,
            "1D": self._price_history_1d,
        }.get(tf, {})
        prices, volumes = {}, {}
        for sym, candles in store.items():
            prices[sym] = [c["close"] for c in candles if c.get("close") is not None]
            volumes[sym] = [c["volume"] for c in candles if c.get("volume") is not None]
        return prices, volumes

    async def _on_ticker(self, ticker):
        self._prices[ticker.symbol] = ticker.price
        if ticker.symbol in self._price_history_1h and self._price_history_1h[ticker.symbol]:
            last = self._price_history_1h[ticker.symbol][-1].copy()
            last["close"] = ticker.price
            last["volume"] = (last.get("volume") or 0) + (ticker.volume or 0)
            self._price_history_1h[ticker.symbol][-1] = last
        bus.emit(Events.PRICE_UPDATE, {
            "symbol": ticker.symbol,
            "price": ticker.price,
            "volume": ticker.volume,
            "change_24h": ticker.change_24h,
        })

    async def _process_signal(self, symbol: str, result,
                              prices_1h: List[float] = None,
                              prices_4h: List[float] = None,
                              prices_1d: List[float] = None,
                              volumes_1h: List[float] = None):
        if not result.signal:
            return
        signal = result.signal
        mode = config.effective_mode

        if signal.action in ("LONG", "SHORT"):
            if mode == "live" and not config.AI_TRADE_ENABLED:
                return

            # Whitelist check (defensive)
            if not is_supported(symbol):
                logger.info(f"BLOCKED {symbol} - poza whitelist (core.symbols)")
                return

            check = risk.check_entry(symbol, signal.action, signal.price)
            if not check.allowed:
                logger.warning(f"Zablokowane: {symbol} | {check.reason}")
                bus.emit(Events.RISK_BLOCKED, {
                    "symbol": symbol, "reason": check.reason
                })
                return

            # Fetch live derivatives (funding rate, OI) z cache 8h
            deriv = await self._fetch_deriv(symbol)

            # === AI ENSEMBLE FILTER — sygnał pochodzi z score_symbol (już zwalidowany) ===
            # score_symbol zawiera override logikę (oversold, BB, confidence). Nie powtarzamy.
            features_dict = None
            if config.AI_TRADE_ENABLED and ensemble.active:
                features_dict = self._compute_features(symbol, prices_1h, prices_4h,
                                                       prices_1d, volumes_1h, deriv)
                min_conf = self._strategy.min_confidence if hasattr(self._strategy, "min_confidence") else 0.30
                sig_conf = signal.confidence if hasattr(signal, "confidence") else 0.0
                if sig_conf < min_conf:
                    logger.info(f"AI BLOCK {symbol}: low confidence {sig_conf:.2f} < {min_conf:.2f}")
                    return
                logger.info(f"AI OK {symbol}: {signal.action} conf={sig_conf:.2f}")

            # Snapshot features nawet gdy AI_TRADE_ENABLED=false (dla retreningu)
            if features_dict is None:
                features_dict = self._compute_features(symbol, prices_1h, prices_4h,
                                                       prices_1d, volumes_1h, deriv)

            size_coins, size_usdt = risk.calculate_position_size(signal.price)

            if mode == "live" and self._exchange and self._exchange.connected:
                order = await self._exchange.place_order(
                    symbol=symbol,
                    side=signal.action.lower(),
                    size=size_coins,
                    leverage=config.trading.leverage,
                    reduce_only=False,
                )
                if not order.success:
                    logger.error(f"Order failed: {symbol} | {order.error}")
                    state.add_log("system", "ERROR", component="engine",
                                  message=f"Order FAILED: {symbol} | {order.error}")
                    return
                if order.price and order.price > 0:
                    signal.price = order.price
                logger.info(f"Order LIVE: {signal.action} {symbol} | {order.order_id}")

            _md = getattr(signal, "metadata", {}) or {}
            _ind = _md.get("indicators", {}) or {}
            _tp = _ind.get("tp")
            _sl = _ind.get("sl")
            state.open_position(
                symbol=symbol, side=signal.action,
                entry_price=signal.price,
                size_coins=size_coins, size_usdt=size_usdt,
                strategy=self._strategy.name, reason=signal.reason,
                features_dict=features_dict,
                tp_price=float(_tp) if _tp else None,
                sl_price=float(_sl) if _sl else None,
            )
            self._position_entry_time[symbol] = datetime.now(timezone.utc)
            self._pyramided.discard(symbol)
            risk.clear_pyramid(symbol)  # reset przy otwarciu nowej pozycji na tym symbolu
            logger.info(f"OPEN {signal.action} {symbol} @ {signal.price} | {mode.upper()}")

        elif signal.action == "CLOSE":
            if mode == "live" and self._exchange and self._exchange.connected:
                order = await self._exchange.close_position(symbol)
                if not order.success:
                    logger.error(f"Close order failed: {symbol} | {order.error}")
                    return
            r = state.close_position(symbol, signal.price, reason=signal.reason)
            if r:
                risk.record_trade(r["pnl"])
                risk.clear_pyramid(symbol)
                self._pyramided.discard(symbol)
                self._position_entry_time.pop(symbol, None)
                logger.info(f"CLOSE {symbol} @ {signal.price} | PnL: {r['pnl']} | {mode.upper()}")

    async def _fetch_deriv(self, symbol: str) -> dict:
        """Pobiera funding rate i OI dla symbolu z cache (TTL 8h)."""
        cached = self._deriv_cache.get(symbol)
        if cached:
            data, fetched_at = cached
            if (datetime.now(timezone.utc) - fetched_at).total_seconds() < self._deriv_cache_ttl:
                return data
        if self._exchange and self._exchange.connected:
            try:
                data = await self._exchange.fetch_open_interest(symbol)
                self._deriv_cache[symbol] = (data, datetime.now(timezone.utc))
                return data
            except Exception as e:
                logger.debug(f"_fetch_deriv {symbol}: {e}")
        return {"open_interest": 0.0, "funding_rate": 0.0}

    async def _check_pyramid_positions(self):
        """Sprawdza otwarte pozycje pod kątem warunku pyramid (pierwsza potwierdzająca świeczka)."""
        mode = config.effective_mode
        open_positions = state.get_open_positions(mode)
        if not open_positions:
            return
        now = datetime.now(timezone.utc)
        for pos in open_positions:
            symbol = pos["symbol"]
            if symbol in self._pyramided:
                continue
            if risk.is_pyramid_blocked():
                continue  # blok piramidy po poprzednim pyramid SL
            entry_time = self._position_entry_time.get(symbol)
            if entry_time is None:
                continue
            elapsed = (now - entry_time).total_seconds()
            # Sprawdzamy tylko w oknie 1h-4h po wejściu (pierwsza świeczka potwierdzająca)
            if elapsed < 3600 or elapsed > 14400:
                continue
            candles = self._price_history_1h.get(symbol, [])
            if len(candles) < 3:
                continue
            # Świeczka potwierdzająca: zamknęła się w kierunku pozycji
            last_close = candles[-1].get("close", 0)
            prev_close = candles[-2].get("close", 0)
            entry_price = pos["entry_price"]
            side = pos["side"]
            if side == "LONG":
                confirms = last_close > prev_close and last_close > entry_price
            else:  # SHORT
                confirms = last_close < prev_close and last_close < entry_price
            if not confirms:
                continue
            # Pyramid: 1% oryginalnego rozmiaru (doktryna - ostrozna dokladka)
            cur_price = self._prices.get(symbol, last_close)
            add_coins, add_usdt = risk.calculate_position_size(cur_price)
            add_coins = round(add_coins * 0.01, 8)
            add_usdt = round(add_usdt * 0.01, 2)
            if mode == "live" and self._exchange and self._exchange.connected:
                order = await self._exchange.place_order(
                    symbol=symbol, side=side.lower(),
                    size=add_coins, leverage=config.trading.leverage,
                    reduce_only=False,
                )
                if not order.success:
                    logger.warning(f"PYRAMID order failed {symbol}: {order.error}")
                    continue
            ok = state.pyramid_add(symbol, add_coins, add_usdt, cur_price, mode)
            if ok:
                self._pyramided.add(symbol)
                risk.record_pyramid_open(symbol)  # rejestracja w RiskManager (dla pyramid SL doctrine)
                logger.info(f"PYRAMID {side} {symbol} +{add_usdt:.1f}USDT @ {cur_price} "
                            f"| elapsed={elapsed/3600:.1f}h | conf_candle={last_close:.4f}")

    def _ohlc_aux(self, symbol: str, prices_1h):
        """(highs, lows, timestamps) z historii 1h albo (None, None, None).

        FIX 2026-08-07: engine przekazywal do build_features_live SAME CLOSE.
        Zmierzone na AVAX: 22 cechy w live mialy zla wartosc — czesc None
        (r_cci_20, r_di_spread, e_* SMC, x_parkinson_24h), a czesc, grozniejsza,
        0.0 zamiast realnej liczby (sr_node_strength 0 zamiast 49, vwap_dev 0
        zamiast -1.11). Zero nie znaczy "brak" — dla skalera to skrajna wartosc
        OOD, wiec modele dostawaly smieciowe wejscie. Swiece maja high/low/
        timestamp od zawsze, nikt ich nie podawal.

        Dlugosci MUSZA sie zgadzac: gdy wolajacy podal wlasne prices_1h o innej
        dlugosci, dolaczenie high/low z historii przesunieoby szeregi wzgledem
        siebie — cicha korupcja gorsza niz brak cech.
        """
        hist = self._price_history_1h.get(symbol, [])
        if not hist:
            return None, None, None
        if len(hist) != len(prices_1h or []):
            logger.debug(f"{symbol}: historia {len(hist)} != prices_1h "
                         f"{len(prices_1h or [])} — pomijam high/low")
            return None, None, None
        try:
            return ([c["high"] for c in hist], [c["low"] for c in hist],
                    [c["timestamp"] for c in hist])
        except (KeyError, TypeError):
            return None, None, None

    def _compute_features(self, symbol: str,
                          prices_1h: Optional[List[float]],
                          prices_4h: Optional[List[float]],
                          prices_1d: Optional[List[float]],
                          volumes_1h: Optional[List[float]],
                          deriv: Optional[dict] = None) -> Optional[Dict]:
        """Wrapper na core.features.build_features_live z singleton strategy."""
        try:
            if not prices_1h:
                prices_1h = [c["close"] for c in self._price_history_1h.get(symbol, [])]
                volumes_1h = [c["volume"] for c in self._price_history_1h.get(symbol, [])]
                prices_4h = [c["close"] for c in self._price_history_4h.get(symbol, [])]
                prices_1d = [c["close"] for c in self._price_history_1d.get(symbol, [])]

            # === FIX 2026-08-07: engine NIE przekazywal high/low ani symbolu ===
            # Skutek zmierzony na AVAX: 22 cechy w live mialy zla wartosc. Czesc
            # byla None (r_cci_20, r_di_spread, e_* SMC, x_parkinson_24h), a czesc
            # — grozniejsza — miala 0.0 zamiast realnej liczby: sr_node_strength
            # 0 zamiast 49, vwap_dev 0 zamiast -1.11, fib_dist_pct 0 zamiast 0.41.
            # Zero nie znaczy "brak": dla skalera to skrajna wartosc OOD, wiec
            # modele dostawaly smieciowe wejscie i nikt tego nie widzial.
            # Swiece maja te pola od zawsze (patrz _price_history_1h: timestamp/
            # open/high/low/close/volume) — po prostu nikt ich nie podawal.
            _highs, _lows, _ts = self._ohlc_aux(symbol, prices_1h)

            strategy = self._strategy
            if strategy is None:
                strategy = get_strategy("ai_strategy")

            d = deriv or {}
            # Derywaty LIVE z magazynu (fix 2026-07-20): oi_zscore/oi_change/
            # ls_ratio/taker + funding_change - wczesniej NIE liczone w live
            # (tylko funding_rate+oi_total_log) -> modele fit2/cat dostawaly
            # zera dla kluczowych cech i zwracaly same NEUTRAL. Snapshot z
            # gieldy (deriv) ma priorytet dla biezacego funding/OI, reszta
            # (zscore/zmiana/ls) z magazynu tymi samymi wzorami co backtester.
            from .features import latest_deriv_live
            wh = latest_deriv_live(symbol)
            _funding = d.get("funding_rate") or wh["funding_rate"]
            _oi_log = (float(__import__("math").log1p(d.get("open_interest", 0.0)))
                       if d.get("open_interest") else wh["oi_total_log"])
            return build_features_live(
                strategy=strategy,
                prices_1h=prices_1h or [],
                prices_4h=prices_4h or [],
                prices_1d=prices_1d or [],
                volumes_1h=volumes_1h or [],
                funding_rate=_funding,
                funding_change_24h=wh["funding_change_24h"],
                oi_total_log=_oi_log,
                oi_change_24h=wh["oi_change_24h"],
                oi_zscore_30d=wh["oi_zscore_30d"],
                taker_buy_ratio=wh["taker_buy_ratio"],
                ls_ratio=wh["ls_ratio"],
                ls_ratio_chg_24h=wh["ls_ratio_chg_24h"],
                # high/low: odblokowuja SMC, Ichimoku, VWAP, S/R, Parkinson.
                highs_1h=_highs,
                lows_1h=_lows,
                # symbol + timestampy: mapa likwidacji (dist_*_liq) — ta sama
                # funkcja co w treningu i backtescie, patrz liqmap_features.
                symbol=symbol,
                timestamps_1h=_ts,
            )
        except Exception as e:
            logger.error(f"_compute_features error {symbol}: {e}")
            return None

    def _calc_interval(self) -> int:
        try:
            btc_prices = self._price_history_1h.get("BTC/USDT:USDT", [])
            if len(btc_prices) < 20:
                return self._loop_interval_max
            prices = [c["close"] for c in btc_prices[-20:]]
            volumes = [c["volume"] for c in btc_prices[-20:]]
            import numpy as np
            arr = np.array(prices)
            atr = float(np.abs(np.diff(arr)).mean())
            atr_pct = atr / arr[-1] * 100
            avg_vol = float(np.mean(volumes[:-1]))
            last_vol = float(volumes[-1])
            vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
            activity = min(1.0, (atr_pct / 0.5) * 0.5 + (vol_ratio / 2.0) * 0.5)
            self._market_activity = round(activity, 2)
            if activity > 0.8:
                interval = self._loop_interval_min
            elif activity > 0.5:
                interval = 60
            elif activity > 0.2:
                interval = 120
            else:
                interval = self._loop_interval_max
            if interval != self._loop_interval:
                logger.info(f"Loop interval: {self._loop_interval}s -> {interval}s "
                            f"| activity={activity:.2f}")
            return interval
        except Exception:
            return self._loop_interval_max

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "exchange": self._exchange.name if self._exchange else "none",
            "strategy": self._strategy.name if self._strategy else "none",
            "symbols_loaded": len(self._top_symbols),
            "prices_tracked": len(self._prices),
            "history_1h": len(self._price_history_1h),
            "history_4h": len(self._price_history_4h),
            "history_1d": len(self._price_history_1d),
            "loop_interval": self._loop_interval,
            "market_activity": self._market_activity,
            "ai_active": ensemble.active,
            "ai_trade_enabled": config.AI_TRADE_ENABLED,
            "ai_learn_enabled": getattr(config, "AI_LEARN_ENABLED", False),
            "mode": getattr(config, "effective_mode", getattr(config, "MODE", "paper")),
            "ai_confidence_min": self._strategy.min_confidence if self._strategy and hasattr(self._strategy, "min_confidence") else 0.65,
        }


engine = TradingEngine()

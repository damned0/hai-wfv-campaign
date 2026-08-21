# ===========================================
# HAI_EPV Engine ver.10 Final — strategies/ai_strategy.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: AIStrategy (jedyna aktywna strategia) - score_symbol() (doktryna
# BB extreme/ADX/sesja/regime + ensemble.predict), select_top5(), analyze().
# Doktryna: BB_LONG_MAX/BB_SHORT_MIN, BB_WIDTH_MAX_PCT, ADX_MIN (zdefiniowany,
# NIE uzywany w score_symbol - luka z audytu 2026-07-05), _MODE_PARAMS
# (aggressive/neutral/passive), REGIME_CONF_ADJUST, sesje tradingowe.
# ===========================================
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseStrategy, StrategyResult
from ..features import build_features_live, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Doktryna: reżim trendujący → strategia kontrariańska wyłączona.
# High-vol → wyższy próg pewności (zmienność to kapitał, ale wymaga jakości).
# Mean-reversion → optymalne warunki, normalny próg.
REGIME_CONF_ADJUST = {
    0: +0.02,  # trend_following → minimalny narzut (+2%), 3-class modele radzą
    1: +0.00,  # mean_reversion  → idealne warunki
    2: +0.03,  # high_volatility → niski narzut (+3%), zbyt duży blokuje dobre setupy
}

# BB position thresholds: cena musi być przy wstędze (doktryna ATR/BB)
BB_LONG_MAX  = 0.30  # LONG tylko gdy cena w dolnych 30% wstęgi BB (było 0.20)
BB_SHORT_MIN = 0.70  # SHORT tylko gdy cena w górnych 30% wstęgi BB (było 0.80)

# STREFA KONTYNUACJI (audyt 2026-07-05 - nowa strategia "regime-adaptive",
# toggle `regime_adaptive`, domyslnie WYLACZONY, zero zmiany istniejacego
# zachowania). Zamiast zawsze grac powrot do sredniej (mean-reversion), w
# reżimie trend_following (0) gra sie KONTYNUACJE: plytki pullback W
# KIERUNKU trendu (nie ekstremum PRZECIW niemu). To roznica FILOZOFII, nie
# kolejny filtr - w silnym trendzie "kupuj tanio sprzedawaj drogo" (mean-
# reversion) systematycznie przegrywa z "kupuj dolek w trendzie wzrostowym"
# (kontynuacja). Wymaga zgodnosci z trend_1d (nie samego BB pos).
CONTINUATION_ZONE_LONG  = (0.25, 0.55)  # trend_1d=UP:  plytki pullback -> LONG
CONTINUATION_ZONE_SHORT = (0.45, 0.75)  # trend_1d=DOWN: plytkie odbicie -> SHORT

# BB Bandwidth filter: zbyt szerokie pasma = wysoki vol = mean-reversion zawodne
# bb_width_pct = 4σ/mean (szerokość całego kanału BB jako % ceny)
BB_WIDTH_MAX_PCT = 0.12  # blok gdy pasma > 12% ceny (było 0.08 — zbyt restrykcyjne)

# ADX hard filter: BB extreme ma sens tylko gdy ruch jest wystarczająco silny.
# ADX < 25 → chop/range (false extreme) → skip. ADX >= 25 → realne momentum do reversal.
ADX_MIN = 22  # próg ADX (closes-only proxy)

# Sesje tradingowe (UTC). Dead zone = brak płynności → wyższy próg.
# EU open: 07:00 UTC, US open: 13:30 UTC (key moments z filozofii tradingowej)
_SESSION_DEAD_HOURS = frozenset({22, 23, 0})   # 22:00-01:00 UTC — blok (niska płynność)
_SESSION_PRIME_HOURS = frozenset({7, 8, 13, 14, 15})  # EU open + US open → -0.03 próg (bonus)
_SESSION_CONF_ADJUST = {
    "dead":  None,   # blok
    "prime": -0.03,  # kluczowe momenty sesji → obniżamy próg (lepsza płynność)
    "normal": 0.00,
}

_MODE_PARAMS = {
    "aggressive": dict(
        min_confidence=0.55,
        require_macro_trend=False,
        min_quote_volume_24h=2_000_000,
        atr_tp_mult=2.5,
        atr_sl_mult=1.5,
    ),
    "neutral": dict(
        min_confidence=0.30,
        require_macro_trend=False,
        min_quote_volume_24h=5_000_000,
        atr_tp_mult=3.5,
        atr_sl_mult=1.5,
    ),
    "passive": dict(
        min_confidence=0.65,
        require_macro_trend=False,
        min_quote_volume_24h=10_000_000,
        atr_tp_mult=5.0,
        atr_sl_mult=1.5,
    ),
}

class AIStrategy(BaseStrategy):
    def __init__(self, mode: str = "neutral"):
        super().__init__()
        self.name = "ai_strategy"
        self.top_n = 5
        self.min_history = 220
        self._ensemble = None
        # Regime-adaptive doktryna (audyt 2026-07-05) - toggle, domyslnie
        # WYLACZONY (zero zmiany zachowania). Wlacz przez set_parameters
        # ({"regime_adaptive": True}) lub bezposrednio atrybutem - do testu
        # "strategia trzymajaca sie doktryn" (kontynuacja zamiast reversal
        # w regime=0 trend_following).
        self.regime_adaptive = False
        # doctrine_free (fix 2026-07-20): pomija KIERUNKOWY filtr BB
        # (mean-reversion: LONG tylko przy niskim bb, SHORT przy wysokim) -
        # dokladnie jak _DOCTRINE_FREE w backtesterze. Configi trend-following
        # (CatBoost, fit2) byly WALIDOWANE w WFV z doctrine_free=true, wiec
        # sygnalizuja KONTYNUACJE (LONG przy wysokim bb) - zywa doktryna
        # mean-reversion blokowala je wszystkie -> same NEUTRAL. Wlaczane env
        # STRATEGY_DOCTRINE_FREE=1 per instancja (deploy == walidacja).
        # Filtry SRODOWISKA (bb_width, ADX, sesja) zostaja - to nie kierunek.
        import os as _os
        self.doctrine_free = _os.environ.get("STRATEGY_DOCTRINE_FREE", "0") == "1"
        self.set_mode(mode)

    def _get_ensemble(self):
        if self._ensemble is None:
            from ..ensemble import ensemble
            self._ensemble = ensemble
            if not ensemble.active:
                ensemble.load_models()
        return self._ensemble

    def set_mode(self, mode: str):
        mode = mode.lower()
        if mode not in _MODE_PARAMS:
            logger.warning(f"Nieznany tryb: {mode} -> uzycie neutral")
            mode = "neutral"
        self.mode = mode
        self.version = f"1.4.0-{mode}"
        for k, v in _MODE_PARAMS[mode].items():
            setattr(self, k, v)

        # Prog wejscia per instancja (2026-07-12). UWAGA: to NIE jest
        # AI_CONFIDENCE_MIN — tamta zmienna zasila config.ai.confidence_min,
        # czyli prog FILTRA AI (AI_MODE=filter), co innego niz prog wejscia
        # strategii. Mylnie podobne nazwy. Brak zmiennej = prog z _MODE_PARAMS
        # (zero zmiany zachowania dla instancji, ktore jej nie ustawiaja).
        override = os.getenv("STRATEGY_MIN_CONFIDENCE")
        if override:
            try:
                self.min_confidence = float(override)
                logger.info(f"min_confidence override z env: {self.min_confidence}")
            except ValueError:
                logger.warning(f"STRATEGY_MIN_CONFIDENCE='{override}' nie jest liczba — ignoruje")

        logger.info(f"Strategia: {self.name} v{self.version} | "
                    f"min_conf={self.min_confidence} | macro_filter={self.require_macro_trend}")

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0).sum() / period
        losses = -np.where(deltas < 0, deltas, 0).sum() / period
        if losses == 0:
            return 100.0
        rs = gains / losses
        return float(100 - 100 / (1 + rs))

    def calculate_ema(self, prices: List[float], period: int) -> float:
        if not prices:
            return 0.0
        if len(prices) < period:
            return float(np.mean(prices))
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return float(ema)

    def calculate_atr(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 0.0
        moves = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return float(np.mean(moves[-period:]))

    def calculate_roc(self, prices: List[float], period: int = 10) -> float:
        if len(prices) < period + 1:
            return 0.0
        return float((prices[-1] / prices[-1 - period] - 1) * 100)

    def detect_trend(self, prices: List[float],
                     short_period: int = 21, long_period: int = 50) -> str:
        if len(prices) < long_period:
            return "SIDEWAYS"
        ema_s = self.calculate_ema(prices, short_period)
        ema_l = self.calculate_ema(prices, long_period)
        if ema_l == 0:
            return "SIDEWAYS"
        diff_pct = (ema_s - ema_l) / ema_l * 100
        if diff_pct > 0.5:
            return "UP"
        if diff_pct < -0.5:
            return "DOWN"
        return "SIDEWAYS"

    @staticmethod
    def _calc_adx_closes(prices: List[float], period: int = 14) -> float:
        """ADX z samych close (proxy DM+ = max(0, c[i]-c[i-1]), DM- = max(0, c[i-1]-c[i])).
        Wystarczająco koreluje z prawdziwym ADX(H,L,C) do filtrowania setupów."""
        n = len(prices)
        if n < period * 2 + 2:
            return 0.0
        dm_plus, dm_minus, trs = [], [], []
        for i in range(1, n):
            diff = prices[i] - prices[i - 1]
            dm_plus.append(max(diff, 0.0))
            dm_minus.append(max(-diff, 0.0))
            trs.append(abs(diff))
        atr = sum(trs[:period])
        pdm = sum(dm_plus[:period])
        mdm = sum(dm_minus[:period])
        dx_vals = []
        for i in range(period, len(trs)):
            atr = atr - atr / period + trs[i]
            pdm = pdm - pdm / period + dm_plus[i]
            mdm = mdm - mdm / period + dm_minus[i]
            pdi = pdm / atr * 100 if atr > 0 else 0.0
            mdi = mdm / atr * 100 if atr > 0 else 0.0
            dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0.0
            dx_vals.append(dx)
        return round(sum(dx_vals[-period:]) / period, 1) if dx_vals else 0.0

    def calc_stochastic(self, prices: List[float],
                        period: int = 14, smooth: int = 3) -> Tuple[float, float]:
        """Stochastic %K/%D na bazie close (proxy high=low=close). Wystarczy do potwierdzenia."""
        if len(prices) < period + smooth:
            return 50.0, 50.0
        k_vals = []
        for i in range(len(prices) - smooth, len(prices)):
            chunk = prices[max(0, i - period + 1): i + 1]
            lo, hi = min(chunk), max(chunk)
            k_vals.append(100.0 * (prices[i] - lo) / (hi - lo) if hi > lo else 50.0)
        k = k_vals[-1]
        d = float(np.mean(k_vals))
        return round(k, 2), round(d, 2)

    @staticmethod
    def _get_session(ts: Optional[datetime] = None) -> str:
        """Zwraca nazwę sesji tradingowej. ts=None → teraz (live); ts → historyczny (backtest)."""
        hour = (ts or datetime.now(timezone.utc)).hour
        if hour in _SESSION_DEAD_HOURS:
            return "dead"
        if hour in _SESSION_PRIME_HOURS:
            return "prime"
        return "normal"

    @staticmethod
    def calculate_sr_levels(prices_1h: List[float], prices_4h: List[float] = None,
                            atr: float = None) -> Dict:
        """Oblicza wsparcia i opory techniczne z danych 1h i 4h.

        Metody:
        - Swing highs/lows (lokalne ekstrema z okna 5 świec)
        - Pivot points klasyczne (H+L+C)/3 z ostatniej zamkniętej świecy 4h
        - BB bands (dynamiczne wsparcie/opór)
        - Poziomy psychologiczne (round numbers)
        """
        if not prices_1h or len(prices_1h) < 20:
            return {}
        arr = np.asarray(prices_1h, dtype=np.float64)
        cur = float(arr[-1])

        # --- Swing highs / lows (lookback 100 świec, okno 5) ---
        window = 5
        lookback = min(100, len(arr))
        sub = arr[-lookback:]
        swing_highs, swing_lows = [], []
        for i in range(window, len(sub) - window):
            if sub[i] == sub[i-window:i+window+1].max():
                swing_highs.append(float(sub[i]))
            if sub[i] == sub[i-window:i+window+1].min():
                swing_lows.append(float(sub[i]))

        # Grupuj bliskie poziomy (±0.5% od siebie)
        def cluster(levels: list, tol: float = 0.005) -> list:
            if not levels:
                return []
            levels = sorted(set(levels))
            groups, g = [], [levels[0]]
            for v in levels[1:]:
                if v <= g[-1] * (1 + tol):
                    g.append(v)
                else:
                    groups.append(round(sum(g) / len(g), 6))
                    g = [v]
            groups.append(round(sum(g) / len(g), 6))
            return groups

        resistance_levels = cluster(swing_highs)
        support_levels = cluster(swing_lows)

        # --- Pivot points z 4h (H+L+C)/3 ---
        pivot, r1, s1, r2, s2 = None, None, None, None, None
        src = prices_4h if prices_4h and len(prices_4h) >= 3 else prices_1h
        if len(src) >= 3:
            h = max(src[-24:]) if len(src) >= 24 else max(src)
            l = min(src[-24:]) if len(src) >= 24 else min(src)
            c = float(src[-1])
            pivot = round((h + l + c) / 3, 6)
            r1 = round(2 * pivot - l, 6)
            s1 = round(2 * pivot - h, 6)
            r2 = round(pivot + (h - l), 6)
            s2 = round(pivot - (h - l), 6)

        # --- BB bands ---
        bb_window = np.asarray(prices_1h[-20:], dtype=np.float64)
        bb_mean = float(bb_window.mean())
        bb_std = float(bb_window.std())
        bb_upper = round(bb_mean + 2 * bb_std, 6)
        bb_lower = round(bb_mean - 2 * bb_std, 6)

        # --- ATR-based TP/SL od ceny bieżącej ---
        if atr is None and len(prices_1h) >= 15:
            moves = [abs(prices_1h[i] - prices_1h[i-1]) for i in range(1, len(prices_1h))]
            atr = float(np.mean(moves[-14:]))
        tp_long  = round(cur + 3.5 * atr, 6) if atr else None
        sl_long  = round(cur - 1.5 * atr, 6) if atr else None
        tp_short = round(cur - 3.5 * atr, 6) if atr else None
        sl_short = round(cur + 1.5 * atr, 6) if atr else None

        # --- Poziom psychologiczny (najbliższy okrągły poziom) ---
        mag = 10 ** (len(str(int(cur))) - 2) if cur >= 1 else 0.001
        psych = round(round(cur / mag) * mag, 6)

        # Podziel S/R na powyżej/poniżej ceny
        sr_above = sorted([v for v in resistance_levels + ([r1,r2] if r1 else []) if v > cur * 1.002])
        sr_below = sorted([v for v in support_levels   + ([s1,s2] if s1 else []) if v < cur * 0.998], reverse=True)

        return {
            "current_price": round(cur, 6),
            "resistance": sr_above[:3],        # top 3 opory powyżej ceny
            "support":    sr_below[:3],        # top 3 wsparcia poniżej ceny
            "bb_upper":   bb_upper,
            "bb_lower":   bb_lower,
            "pivot":      pivot,
            "r1": r1, "r2": r2,
            "s1": s1, "s2": s2,
            "tp_long":    tp_long,
            "sl_long":    sl_long,
            "tp_short":   tp_short,
            "sl_short":   sl_short,
            "psych_level": psych,
            "atr":        round(atr, 6) if atr else None,
        }

    def _check_bb_extreme(self, features: Dict, action: str) -> bool:
        """Cena musi być przy wstędze BB — doktryna ATR/BB, nie handlujemy w środku."""
        bb_pos = features.get("price_position_bb", 0.5)
        if action == "LONG":
            return bb_pos <= BB_LONG_MAX
        if action == "SHORT":
            return bb_pos >= BB_SHORT_MIN
        return False

    def _check_doctrine_zone(self, features: Dict, action: str, regime: Optional[int]) -> bool:
        """Regime-adaptive doktryna (audyt 2026-07-05) - gdy self.regime_adaptive
        jest False (domyslnie), zachowuje sie IDENTYCZNIE jak _check_bb_extreme
        (zero zmiany istniejacego zachowania). Gdy True I regime==0
        (trend_following), zamiast ekstremum BB (mean-reversion) sprawdza
        strefe KONTYNUACJI zgodna z trend_1d - filozofia "kupuj dolek w
        trendzie", nie "kupuj tanio sprzedawaj drogo"."""
        if not getattr(self, "regime_adaptive", False) or regime != 0:
            return self._check_bb_extreme(features, action)
        bb_pos = features.get("price_position_bb", 0.5)
        trend_1d = features.get("trend_1d", 0.0)
        if action == "LONG":
            return trend_1d > 0 and CONTINUATION_ZONE_LONG[0] <= bb_pos <= CONTINUATION_ZONE_LONG[1]
        if action == "SHORT":
            return trend_1d < 0 and CONTINUATION_ZONE_SHORT[0] <= bb_pos <= CONTINUATION_ZONE_SHORT[1]
        return False

    @staticmethod
    def _check_bb_width(prices_1h: List[float]) -> bool:
        """True = pasma OK (wąskie/normalne). False = blok (zbyt szerokie = ekstremalny vol)."""
        if len(prices_1h) < 20:
            return True
        arr = np.asarray(prices_1h[-20:], dtype=np.float64)
        mean = arr.mean()
        if mean <= 0:
            return True
        bb_width_pct = 4 * arr.std() / mean
        return bb_width_pct <= BB_WIDTH_MAX_PCT

    def _check_volume(self, prices_1h: List[float], volumes_1h: List[float]) -> bool:
        if not volumes_1h or len(volumes_1h) < 24 or not prices_1h:
            return False
        last_24v = volumes_1h[-24:]
        last_24p = prices_1h[-24:]
        if len(last_24p) != len(last_24v):
            return False
        quote_vols = [v * p for v, p in zip(last_24v, last_24p)]
        total_24h = float(sum(quote_vols))
        return total_24h >= self.min_quote_volume_24h

    def _check_macro_trend(self, prices_1d: List[float], side: str) -> bool:
        if not self.require_macro_trend:
            return True
        if not prices_1d or len(prices_1d) < 50:
            return True
        ema200_1d = self.calculate_ema(prices_1d, min(200, len(prices_1d) - 1))
        cur_1d = prices_1d[-1]
        if ema200_1d == 0:
            return True
        if cur_1d > ema200_1d * 1.01:
            return side == "LONG"
        elif cur_1d < ema200_1d * 0.99:
            return side == "SHORT"
        return True

    def diagnoza_bramek(self, symbol, prices_1h, prices_4h, prices_1d, volumes_1h,
                        features, akcja, pewnosc, regime=None, timestamp=None):
        """Zwraca stan KAZDEJ bramki dla podanego sygnalu — do podgladu w panelu.

        WAZNE: wola DOKLADNIE te same metody co score_symbol (_check_volume,
        _check_doctrine_zone, _check_bb_width, _calc_adx_closes, _check_macro_trend).
        Zadnej kopii logiki — inaczej panel i handel rozjechalyby sie po pierwszej
        zmianie progu, co w tym projekcie zdarzylo sie juz wielokrotnie
        (ADX liczony dwa razy, _filters w ctrl.py, piramidowanie).

        Kolejnosc 1:1 z score_symbol. Zwraca liste slownikow:
          {nazwa, ok, opis} — ok=None gdy bramka nieaktywna (np. doktryna
          przy doctrine_free=1).
        """
        b = []

        def _dod(n, ok, o):
            # _check_* zwracaja czasem numpy.bool_, ktorego FastAPI nie umie
            # zserializowac ("'numpy.bool' object is not iterable"). Rzutujemy
            # na czysty bool; None zostaje None (bramka nieaktywna).
            b.append({"nazwa": n, "ok": None if ok is None else bool(ok), "opis": str(o)})

        _dod("historia", len(prices_1h) >= self.min_history,
             f"{len(prices_1h)}/{self.min_history} swiec")
        _dod("wolumen", self._check_volume(prices_1h, volumes_1h or []), "filtr plynnosci")

        sesja = self._get_session(timestamp)
        _dod("sesja", sesja != "dead", f"{sesja}" + (" (22-01 UTC = blok)" if sesja == "dead" else ""))

        _dod("kierunek ensembla", akcja in ("LONG", "SHORT"),
             f"{akcja} conf={pewnosc:.3f}")

        if getattr(self, "doctrine_free", False):
            _dod("doktryna BB", None, "pominieta (STRATEGY_DOCTRINE_FREE=1)")
        elif akcja in ("LONG", "SHORT"):
            _dod("doktryna BB", self._check_doctrine_zone(features, akcja, regime),
                 f"bb_pos={features.get('price_position_bb', 0):.2f} regime={regime}")

        try:
            import numpy as _np
            _a = _np.asarray(prices_1h[-20:], dtype=float)
            _bw = round(4 * _a.std() / _a.mean() * 100, 1) if _a.mean() > 0 else 0
        except Exception:
            _bw = 0
        _dod("szerokosc BB", self._check_bb_width(prices_1h),
             f"{_bw}% (max {BB_WIDTH_MAX_PCT*100:.0f}%)")

        _adx = self._calc_adx_closes(prices_1h, period=14)
        _dod("ADX", _adx >= ADX_MIN, f"{_adx:.1f} (min {ADX_MIN})")

        _dod("prog pewnosci", pewnosc >= self.min_confidence,
             f"{pewnosc:.3f} (min {self.min_confidence:.2f})")

        if akcja in ("LONG", "SHORT"):
            _dod("trend makro", self._check_macro_trend(prices_1d, akcja), "zgodnosc z 1d")

        return b

    def score_symbol(self, symbol: str, prices_1h: List, prices_4h: List,
                     prices_1d: List, volumes_1h: List = None,
                     funding_rate: float = 0.0, funding_change_24h: float = 0.0,
                     oi_total_log: float = 0.0, oi_change_24h: float = 0.0,
                     oi_zscore_30d: float = 0.0, funding_extreme: float = 0.0,
                     timestamp: Optional[datetime] = None,
                     # 2026-08-07: bez tych trzech cechy SMC/Ichimoku/VWAP/S-R/
                     # likwidacji sa None albo 0.0 (engine._ohlc_aux). Domyslne
                     # None = zachowanie jak dotad, wiec starzy wolajacy dzialaja.
                     highs_1h: Optional[List] = None,
                     lows_1h: Optional[List] = None,
                     timestamps_1h: Optional[List] = None):
        if not prices_1h or len(prices_1h) < self.min_history:
            return 0.0, "NEUTRAL"
        if not self._check_volume(prices_1h, volumes_1h or []):
            return 0.0, "NEUTRAL"
        ensemble = self._get_ensemble()
        if not ensemble.active:
            logger.debug(f"{symbol}: ensemble nieaktywny, skip")
            return 0.0, "NEUTRAL"

        # Detekcja reżimu
        try:
            from ..regime_detector import regime_detector as _rd
            regime = _rd.detect_from_closes(prices_1h, volumes_1h or [])
        except Exception:
            regime = None

        # Session gate: dead zone 22:00-01:00 UTC → blok
        session = self._get_session(timestamp)
        if session == "dead":
            logger.debug(f"{symbol}: dead zone (session) → skip")
            return 0.0, "NEUTRAL"

        # Próg pewności: reżim + sesja
        # PARYTET (2026-08-13): korekta reżimu miała tu ODWROTNY ZNAK niż w
        # walidatorze. backtester.py:1391 robi `_cpre = batch_conf + _radj`,
        # czyli DODAJE korektę do WYNIKU (dodatnia = łatwiej przejść). Tutaj było
        # `effective_min_conf = min_confidence + regime_adjust`, czyli dodawanie
        # do PROGU (dodatnia = TRUDNIEJ przejść). Ten sam słownik
        # {0: 0.02, 1: 0.00, 2: 0.03}, przeciwny skutek — live był surowszy od
        # zwalidowanego układu o 0.02–0.03, a przy sesji prime o kolejne 0.03.
        # Teraz obie ścieżki modyfikują WYNIK; próg zostaje czysty.
        regime_adjust = REGIME_CONF_ADJUST.get(regime, 0.0) or 0.0
        session_adjust = _SESSION_CONF_ADJUST.get(session, 0.0) or 0.0
        conf_adjust = regime_adjust + session_adjust
        effective_min_conf = self.min_confidence

        features = build_features_live(
            strategy=self,
            prices_1h=prices_1h,
            prices_4h=prices_4h or [],
            prices_1d=prices_1d or [],
            volumes_1h=volumes_1h or [],
            funding_rate=funding_rate,
            funding_change_24h=funding_change_24h,
            oi_total_log=oi_total_log,
            oi_change_24h=oi_change_24h,
            oi_zscore_30d=oi_zscore_30d,
            funding_extreme=funding_extreme,
            timestamp=timestamp,
            # 2026-08-07 — patrz engine._ohlc_aux
            highs_1h=highs_1h,
            lows_1h=lows_1h,
            symbol=symbol,
            timestamps_1h=timestamps_1h,
        )
        if not features:
            logger.debug(f"{symbol}: build_features_live zwrocil None")
            return 0.0, "NEUTRAL"

        try:
            result = ensemble.predict(features, regime=regime)
        except Exception as e:
            logger.error(f"{symbol}: ensemble.predict failed: {e}")
            return 0.0, "NEUTRAL"

        action = result.get("action", "NEUTRAL")
        confidence = result.get("confidence", 0.0)

        if action == "NEUTRAL":
            return 0.0, "NEUTRAL"

        # Doktryna BB (lub strefa kontynuacji gdy regime_adaptive+regime=0).
        # doctrine_free (fix 2026-07-20): pomija filtr KIERUNKOWY - model sam
        # decyduje kierunek (spojne z WFV doctrine_free=true na ktorym te
        # configi dostaly GO). Filtry srodowiska (bb_width/ADX) dalej dzialaja.
        if not getattr(self, "doctrine_free", False):
            if not self._check_doctrine_zone(features, action, regime):
                bb_pos = features.get("price_position_bb", 0.5)
                logger.debug(f"{symbol}: doktryna nie spelniona ({action}, bb_pos={bb_pos:.2f}, regime={regime}) → skip")
                return 0.0, "NEUTRAL"

        # BB Bandwidth filter: zbyt szerokie pasma → ekstremalny vol → mean-reversion zawodne
        if not self._check_bb_width(prices_1h):
            arr = np.asarray(prices_1h[-20:], dtype=np.float64)
            bw = round(4 * arr.std() / arr.mean() * 100, 1) if arr.mean() > 0 else 0
            logger.debug(f"{symbol}: bb_width={bw}% > {BB_WIDTH_MAX_PCT*100:.0f}% → skip (wide bands)")
            return 0.0, "NEUTRAL"

        # ADX hard filter (audyt 2026-07-05 - zdefiniowany ale nigdy nie
        # sprawdzany, luka wzgledem core/backtester.py ktory to egzekwuje
        # od dawna: adx_arr >= ADX_MIN). BB extreme ma sens tylko gdy ruch
        # jest wystarczajaco silny - ADX < 22 = chop/range = false extreme.
        adx = self._calc_adx_closes(prices_1h, period=14)
        if adx < ADX_MIN:
            logger.debug(f"{symbol}: ADX={adx:.1f} < {ADX_MIN} → skip (chop/range)")
            return 0.0, "NEUTRAL"

        # Stochastic jako filtr potwierdzający (bonus pewności)
        stoch_k, stoch_d = self.calc_stochastic(prices_1h, period=14, smooth=3)
        stoch_confirms = (
            (action == "LONG"  and stoch_k < 25 and stoch_d < 25) or
            (action == "SHORT" and stoch_k > 75 and stoch_d > 75)
        )
        # PARYTET (2026-08-13): korekty reżimu i sesji idą do WYNIKU, nie do progu
        # — kolejność 1:1 z backtester.py:1390-1396 (najpierw reżim + sesja,
        # dopiero potem mnożnik stochastyczny, na końcu porównanie z progiem).
        if conf_adjust:
            confidence = max(0.0, min(confidence + conf_adjust, 0.99))

        if stoch_confirms:
            confidence = min(confidence * 1.08, 0.99)  # +8% przy potwierdzeniu stoch
            logger.debug(f"{symbol}: stochastic confirming {action} (K={stoch_k} D={stoch_d})")

        if confidence < effective_min_conf:
            logger.debug(f"{symbol}: conf={confidence:.3f} (korekta {conf_adjust:+.2f}) "
                         f"< {effective_min_conf:.3f} (regime={regime}, sesja={session}) → skip")
            return 0.0, "NEUTRAL"

        if not self._check_macro_trend(prices_1d, action):
            logger.debug(f"{symbol}: macro filter block ({action})")
            return 0.0, "NEUTRAL"

        score = round(confidence * 100, 1)
        return score, action

    def analyze(self, symbol: str, prices: List, volumes=None, **kwargs) -> StrategyResult:
        if len(prices) < self.min_history:
            return StrategyResult(
                opportunities=[f"Za malo danych: {len(prices)}/{self.min_history}"]
            )
        prices_4h = kwargs.get("prices_4h", [])
        prices_1d = kwargs.get("prices_1d", [])
        volumes_1h = kwargs.get("volumes_1h", volumes)
        in_position = kwargs.get("in_position", False)

        deriv = {
            'funding_rate': kwargs.get('funding_rate', 0.0),
            'funding_change_24h': kwargs.get('funding_change_24h', 0.0),
            'oi_total_log': kwargs.get('oi_total_log', 0.0),
            'oi_change_24h': kwargs.get('oi_change_24h', 0.0),
            'oi_zscore_30d': kwargs.get('oi_zscore_30d', 0.0),
            'funding_extreme': kwargs.get('funding_extreme', 0.0),
            # 2026-08-07: high/low/timestamp przekazywane przez engine. Bez nich
            # cechy SMC/Ichimoku/VWAP/S-R/likwidacji byly None albo 0.0 (patrz
            # engine._ohlc_aux). Brak kluczy = stare zachowanie, nic nie peka.
            'highs_1h': kwargs.get('highs_1h'),
            'lows_1h': kwargs.get('lows_1h'),
            'timestamps_1h': kwargs.get('timestamps_1h'),
        }

        score, action = self.score_symbol(symbol, prices, prices_4h, prices_1d, volumes_1h, **deriv)
        cur = prices[-1]
        rsi = self.calculate_rsi(prices, 14)
        trend_1h = self.detect_trend(prices)
        atr = self.calculate_atr(prices, 14)
        roc = self.calculate_roc(prices, 10)

        indicators = {
            "rsi": round(rsi, 2),
            "trend": trend_1h,
            "score": score,
            "action": action,
            "atr": round(atr, 6),
            "momentum": round(roc, 2),
            "current_price": cur,
            "mode": self.mode,
            "confidence": score / 100,
        }

        signal = None
        opportunities = []
        if not in_position and action in ("LONG", "SHORT"):
            confidence = score / 100
            if atr > 0:
                if action == "LONG":
                    indicators["tp"] = round(cur + atr * self.atr_tp_mult, 6)
                    indicators["sl"] = round(cur - atr * self.atr_sl_mult, 6)
                else:
                    indicators["tp"] = round(cur - atr * self.atr_tp_mult, 6)
                    indicators["sl"] = round(cur + atr * self.atr_sl_mult, 6)
            signal = self.create_signal(
                action=action, symbol=symbol,
                reason=(f"AI-{self.mode} | conf={confidence:.2f} | "
                        f"RSI={rsi:.1f} | {trend_1h} | RoC={roc:.2f}%"),
                confidence=confidence, price=cur,
                indicators=indicators,
            )
            opportunities.append(
                f"{action} {symbol} | AI conf={confidence:.2f} | "
                f"ATR={atr:.4f} [{self.mode.upper()}]"
            )
        return StrategyResult(
            signal=signal, indicators=indicators,
            opportunities=opportunities, score=score,
        )

    def select_top5(self, symbols: List[str], data_1h: Dict,
                    data_4h: Dict, data_1d: Dict,
                    volumes_1h: Dict = None) -> List[Dict]:
        volumes_1h = volumes_1h or {}
        scored = []
        for sym in symbols:
            p1 = data_1h.get(sym, [])
            if len(p1) < self.min_history:
                continue
            score, action = self.score_symbol(
                sym, p1,
                data_4h.get(sym, []),
                data_1d.get(sym, []),
                volumes_1h.get(sym),
            )
            if action in ("LONG", "SHORT"):
                scored.append({"symbol": sym, "score": score, "action": action})
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:self.top_n]
        logger.info(f"TOP{self.top_n} [AI-{self.mode.upper()}]: "
                    f"{[(s['symbol'], s['score'], s['action']) for s in top]}")
        return top

    def get_parameters(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and not callable(v)}

    def set_parameters(self, params: Dict):
        if "mode" in params:
            self.set_mode(params.pop("mode"))
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        logger.info(f"{self.name} v{self.version} - parametry zaktualizowane")

from .registry import register_strategy
register_strategy("ai_strategy",            lambda: AIStrategy("neutral"))
register_strategy("ai_strategy_aggressive", lambda: AIStrategy("aggressive"))
register_strategy("ai_strategy_neutral",    lambda: AIStrategy("neutral"))
register_strategy("ai_strategy_passive",    lambda: AIStrategy("passive"))

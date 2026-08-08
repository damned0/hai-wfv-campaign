# ===========================================
# HAI_EPV Engine ver.10 Final — core/features.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: build_features_live() (cechy na zywo dla strategii/backtestu),
# build_feature_sequence_live() (sekwencje dla modeli neuralnych, (seq_len,24)),
# FEATURE_NAMES (19, drzewa) / NEURAL_FEATURE_NAMES (24, MLP/LSTM/TCN/Transformer).
# ===========================================
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Macro extended cache (audyt 2026-07-05) - Gold/Oil/SP500/VIX/US10Y/DXY/BTC
# dominance, zmiana % dzien-do-dnia. Ladowane raz na X minut (nie za kazdym
# wywolaniem - to plik na dysku odswiezany max co kilka godzin przez cron),
# zeby CAT (jedyny model ktory tego uzywa - core/ml_trainer.py MODEL_FEATURES)
# mial te same cechy w LIVE co w treningu - inaczej powtorzylibysmy blad
# fear_greed/btc_context (uzywane w treningu, NIGDY nie dostarczane w live).
_MACRO_EXT_LIVE_CACHE: Dict[str, float] = {}
_MACRO_EXT_LIVE_TS: Optional[float] = None
_MACRO_EXT_TICKERS = ['gold', 'oil_wti', 'sp500', 'vix', 'us10y_yield', 'dxy', 'btc_dominance']
_MACRO_EXT_REFRESH_SEC = 3600  # 1h - dane makro sa dzienne, nie trzeba czesciej


def _get_macro_extended_live() -> Dict[str, float]:
    """Zwraca {name}_chg (zmiana % dzien do dnia) dla 7 serii makro, z
    cache'em odswiezanym raz na godzine (dane zrodlowe sa dzienne)."""
    global _MACRO_EXT_LIVE_CACHE, _MACRO_EXT_LIVE_TS
    import time as _time
    now = _time.time()
    if _MACRO_EXT_LIVE_TS is not None and (now - _MACRO_EXT_LIVE_TS) < _MACRO_EXT_REFRESH_SEC:
        return _MACRO_EXT_LIVE_CACHE
    from pathlib import Path
    import pandas as pd
    import os as _os
    macro_dir = Path(_os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / "macro"
    out = {}
    for name in _MACRO_EXT_TICKERS:
        try:
            df = pd.read_parquet(macro_dir / f"{name}.parquet")
            val_col = "close" if "close" in df.columns else "value"
            vals = df[val_col].values
            # btc_dominance.value juz JEST gotowa 7d% zmiana BTC mcap (nie
            # poziomem) - liczenie kolejnej % zmiany dawalo bezsensowna
            # "zmiane zmiany" (audyt 2026-07-05, znalezione empirycznie:
            # 38% "dzienna zmiana" - artefakt dzielenia blisko zera)
            if name == "btc_dominance":
                out[f"{name}_chg"] = float(vals[-1]) if len(vals) else 0.0
            elif len(vals) >= 2 and vals[-2] != 0:
                out[f"{name}_chg"] = float((vals[-1] / vals[-2] - 1) * 100)
            else:
                out[f"{name}_chg"] = 0.0
        except Exception as e:
            logger.debug(f"macro_extended live {name}: {e}")
            out[f"{name}_chg"] = 0.0
    _MACRO_EXT_LIVE_CACHE = out
    _MACRO_EXT_LIVE_TS = now
    return out


FEATURE_NAMES = [
    # Przyciete do 19 cech (audyt 2026-07-03) — patrz core/ml_trainer.py po uzasadnienie
    "rsi", "rsi_4h", "rsi_1d",
    "ema_slow_r", "ema_mid_r",
    "atr_pct",
    "trend_4h", "trend_1d",
    "funding_rate",
    "oi_change_24h", "oi_zscore_30d",
    "price_position_bb", "bb_bandwidth_pct",
    "hour_sin", "hour_cos", "day_of_week",
    "adx_14",
    "macd_hist", "sr_node_strength",
]

# Neural models (MLP/LSTM/TCN/Transformer) were trained with 24 features including
# 4 deriv features (funding_rate, oi_total_log, oi_change_24h, oi_zscore_30d) that
# are always 0.0 in live mode. Kept here verbatim to avoid breaking saved pkl files.
NEURAL_FEATURE_NAMES = [
    "rsi", "rsi_4h", "rsi_1d",
    "ema_slow_r", "ema_mid_r",
    "atr_pct", "momentum",
    "trend_4h", "trend_1d",
    "volume_ratio",
    "funding_rate",
    "price_position_bb", "bb_bandwidth_pct",
    "oi_total_log", "oi_change_24h", "oi_zscore_30d",
    "hour_sin", "day_of_week",
    "adx_14",
    "ema_200_dist_pct",
    "rsi_slope_5h",
    "vol_trend",
    "vwap_dev",
    "taker_buy_ratio",
]

MIN_PRICES_1H = 50
MIN_PRICES_4H = 20
MIN_PRICES_1D = 15


def _calc_adx_live(prices: List[float], period: int = 14) -> float:
    """ADX proxy z samych close prices (rolling window 50)."""
    window = prices[-50:] if len(prices) > 50 else prices
    n = len(window)
    min_len = period * 2 + 2
    if n < min_len:
        return 0.0
    dm_plus, dm_minus, trs = [], [], []
    for k in range(1, n):
        diff = window[k] - window[k - 1]
        dm_plus.append(max(diff, 0.0))
        dm_minus.append(max(-diff, 0.0))
        trs.append(abs(diff))
    if len(trs) < period:
        return 0.0
    atr = sum(trs[:period])
    pdm = sum(dm_plus[:period])
    mdm = sum(dm_minus[:period])
    dx_vals = []
    for k in range(period, len(trs)):
        atr = atr - atr / period + trs[k]
        pdm = pdm - pdm / period + dm_plus[k]
        mdm = mdm - mdm / period + dm_minus[k]
        pdi = pdm / atr * 100 if atr > 0 else 0.0
        mdi = mdm / atr * 100 if atr > 0 else 0.0
        dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0.0
        dx_vals.append(dx)
    return round(sum(dx_vals[-period:]) / period, 1) if dx_vals else 0.0


def _trend_jak_trening(prices) -> float:
    """Trend dokladnie jak ml_trainer.precompute_trend_series: EMA(9) vs EMA(21),
    prog +-0.3%.

    FIX 2026-08-08 (tools/test_parytet_cech): trend_4h/trend_1d liczylo sie tu
    przez strategy.detect_trend(), a to CZWARTA definicja tej samej cechy —
    ai_strategy uzywa EMA(21)/EMA(50) z progiem 0.5%, base.py EMA(21)/EMA(200)
    z progiem 0.5%, backtester uzywal ROC(10) z progiem 0.1, a trening EMA(9)/
    EMA(21) z progiem 0.3%. Model uczyl sie jednej definicji, a na produkcji
    dostawal inna — przy innych okresach EMA potrafi to dac przeciwny znak.
    """
    try:
        import numpy as _np
        p = _np.asarray(prices, dtype=_np.float64)
        if len(p) < 21:
            return 0.0
        _s = pd.Series(p) if "pd" in globals() else None
        if _s is None:
            import pandas as _pd
            _s = _pd.Series(p)
        ef = _s.ewm(span=9, adjust=False).mean().values[-1]
        es = _s.ewm(span=21, adjust=False).mean().values[-1]
        if es == 0:
            return 0.0
        d = (ef - es) / es * 100
        return 1.0 if d > 0.3 else (-1.0 if d < -0.3 else 0.0)
    except Exception:
        return 0.0


def _trend_to_float(trend: Any) -> float:
    if isinstance(trend, str):
        t = trend.upper()
        if t == "UP":
            return 1.0
        if t == "DOWN":
            return -1.0
        return 0.0
    try:
        return float(trend)
    except Exception:
        return 0.0

def _rsi_series_wilder(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI jako seria — IDENTYCZNE z ml_trainer.calc_rsi (parytet div_rsi)."""
    if len(prices) <= period:
        return np.full(len(prices), 50.0)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices, dtype=np.float64)
    rsi[:period] = 50
    rsi[period] = 100 - 100 / (1 + rs) if down != 0 else 50
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        up = (up * (period - 1) + max(delta, 0)) / period
        down = (down * (period - 1) + -min(delta, 0)) / period
        rs = up / down if down != 0 else 100
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi


def _atr_series_wilder(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR jako seria — IDENTYCZNE z ml_trainer.calc_atr (parytet sd_prox)."""
    n = len(closes)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.zeros(n, dtype=np.float64)
    if n <= period:
        atr[:] = tr[:max(1, n)].mean()
        return atr
    atr[:period] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _ema_series_live(prices: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema


def _macd_hist_live(prices_1h: List[float]) -> float:
    """MACD histogram (12/26/9), spojny z core/ml_trainer.calc_macd_hist."""
    if len(prices_1h) < 40:
        return 0.0
    arr = np.asarray(prices_1h[-200:], dtype=np.float64)
    ema12 = _ema_series_live(arr, 12)
    ema26 = _ema_series_live(arr, 26)
    macd_line = ema12 - ema26
    signal_line = _ema_series_live(macd_line, 9)
    hist = float(macd_line[-1] - signal_line[-1])
    cur = float(arr[-1])
    return round(hist / cur * 100, 4) if cur > 0 else 0.0


def _swing_sr_live(highs_1h: List[float], lows_1h: List[float],
                    lookback: int = 5, tolerance: float = 0.003,
                    max_window: int = 200):
    """PhantomFlow S/R — wersja live (tylko przeszlosc, brak lookahead z definicji:
    swing przy indeksie i wymaga i+lookback <= m-1, czyli tylko potwierdzone punkty
    scisle sprzed biezacej swiecy). Zwraca (sh_level, sh_strength, sl_level, sl_strength)."""
    n = len(highs_1h)
    if n < lookback * 4 + 1:
        return None, 0.0, None, 0.0
    h = np.asarray(highs_1h[-max_window:], dtype=np.float64)
    l = np.asarray(lows_1h[-max_window:], dtype=np.float64)
    m = len(h)

    best_sh = None; best_sh_dist = None; best_sh_strength = 0.0
    best_sl = None; best_sl_dist = None; best_sl_strength = 0.0
    cur = h[-1] if m > 0 else 0.0  # placeholder, nadpisane ponizej przez wywolujacego
    for i in range(lookback, m - lookback):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == window_h.max():
            band = h[i] * tolerance
            touches = int((np.abs(h[:m - lookback] - h[i]) <= band).sum() +
                          (np.abs(l[:m - lookback] - h[i]) <= band).sum())
            if best_sh is None or touches >= best_sh_strength:
                best_sh, best_sh_strength = float(h[i]), touches
        if l[i] == window_l.min():
            band = l[i] * tolerance
            touches = int((np.abs(h[:m - lookback] - l[i]) <= band).sum() +
                          (np.abs(l[:m - lookback] - l[i]) <= band).sum())
            if best_sl is None or touches >= best_sl_strength:
                best_sl, best_sl_strength = float(l[i]), touches
    return best_sh, best_sh_strength, best_sl, best_sl_strength


def _fib_dist_live(cur: float, sh_level: Optional[float], sl_level: Optional[float]) -> float:
    if sh_level is None or sl_level is None or cur <= 0:
        return 0.0
    hi, lo = max(sh_level, sl_level), min(sh_level, sl_level)
    rng = hi - lo
    if rng <= 0:
        return 0.0
    dist = 999.0
    for f in (0.236, 0.382, 0.5, 0.618, 0.786):
        level = hi - rng * f
        dist = min(dist, abs(cur - level) / cur * 100)
    return round(dist, 4)


# ── Cechy RPTR: RF/ET specjalistyczne (2026-08-05, external feature-eng review) ──
# Wszystkie licza sie WYLACZNIE z danych do biezacej swiecy (highs_1h/lows_1h/
# prices_1h/volumes_1h to serie do "teraz" wlacznie) - zero lookaheadu z definicji.
# basis_zscore (futures-spot) pominiety - brak danych spot w magazynie.
# funding_zscore_8h/24h pominiete - brak progowo dostepnej historii funding jako
# serii w tym miejscu (tylko punktowy funding_rate + delta 24h), nie zmyslamy
# fałszywego z-score bez realnej dystrybucji.
def _stoch_rsi_live(rsi_series: np.ndarray, period: int = 14) -> float:
    if len(rsi_series) < period:
        return 50.0
    window = rsi_series[-period:]
    lo, hi = float(window.min()), float(window.max())
    if hi - lo < 1e-8:
        return 50.0
    return float((rsi_series[-1] - lo) / (hi - lo) * 100)


def _cci_live(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3.0
    sma = float(tp.mean())
    mad = float(np.abs(tp - sma).mean())
    if mad < 1e-8:
        return 0.0
    return float((tp[-1] - sma) / (0.015 * mad))


def _di_spread_live(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """DI+ minus DI- (Wilder). Znak = kierunek dominujacego ruchu."""
    n = len(closes)
    if n < period * 3:
        return 0.0
    h, l, c = highs[-(period * 3):], lows[-(period * 3):], closes[-(period * 3):]
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = float(tr[-period:].mean())
    if atr < 1e-8:
        return 0.0
    pdi = float(dm_plus[-period:].mean()) / atr * 100
    mdi = float(dm_minus[-period:].mean()) / atr * 100
    return float(pdi - mdi)


def _range_compression_live(highs: np.ndarray, lows: np.ndarray, short: int = 6, long: int = 24) -> float:
    """Stosunek zakresu krotkiego okna do dlugiego - <1 = kompresja (przed wybiciem)."""
    if len(highs) < long:
        return 1.0
    short_range = float(highs[-short:].max() - lows[-short:].min())
    long_range = float(highs[-long:].max() - lows[-long:].min())
    if long_range < 1e-8:
        return 1.0
    return float(short_range / long_range)


def _structure_bias_live(highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> float:
    """HH+HL = 1.0 (struktura byczno), LH+LL = -1.0 (niedzwiedzia), inaczej 0.0."""
    n = len(highs)
    if n < lookback * 6:
        return 0.0
    swing_highs, swing_lows = [], []
    for i in range(lookback, n - lookback):
        if highs[i] == highs[i - lookback:i + lookback + 1].max():
            swing_highs.append(highs[i])
        if lows[i] == lows[i - lookback:i + lookback + 1].min():
            swing_lows.append(lows[i])
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0.0
    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    if hh and hl:
        return 1.0
    if not hh and not hl:
        return -1.0
    return 0.0


def _equal_highs_lows_live(highs: np.ndarray, lows: np.ndarray, window: int = 20, tolerance: float = 0.0015) -> float:
    """Liczba par prawie-rownych high/low w oknie (SMC 'liquidity pools')."""
    if len(highs) < window:
        return 0.0
    h, l = highs[-window:], lows[-window:]
    cnt = 0
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if abs(h[i] - h[j]) / max(h[i], 1e-8) <= tolerance:
                cnt += 1
            if abs(l[i] - l[j]) / max(l[i], 1e-8) <= tolerance:
                cnt += 1
    return float(cnt)


def _liquidity_sweep_live(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 20):
    """Ostatnia swieca zrobila nowy high/low z lookback, ale zamknela sie z powrotem
    w poprzednim zakresie - klasyczny SMC 'liquidity grab' + reversal."""
    if len(highs) < lookback + 2:
        return 0.0, 0.0
    prior_high = float(highs[-lookback - 1:-1].max())
    prior_low = float(lows[-lookback - 1:-1].min())
    swept_high = 1.0 if (highs[-1] > prior_high and closes[-1] < prior_high) else 0.0
    swept_low = 1.0 if (lows[-1] < prior_low and closes[-1] > prior_low) else 0.0
    return swept_high, swept_low


def _fvg_live(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
    """Fair Value Gap na ostatnich 3 swiecach (ICT/SMC): luka miedzy high[t-2] a low[t]
    (bull) lub low[t-2] a high[t] (bear), jeszcze niewypelniona."""
    if len(highs) < 3 or closes[-1] <= 0:
        return 0.0, 0.0
    h0, l0 = float(highs[-3]), float(lows[-3])
    h2, l2 = float(highs[-1]), float(lows[-1])
    cur = float(closes[-1])
    if l2 > h0:
        return float((l2 - h0) / cur * 100), 1.0
    if h2 < l0:
        return float((l0 - h2) / cur * 100), -1.0
    return 0.0, 0.0


def _bos_choch_live(highs: np.ndarray, lows: np.ndarray, lookback: int = 10) -> float:
    """Break of Structure: 1.0 = przebicie ostatniego swing high, -1.0 = swing low, 0 = brak."""
    if len(highs) < lookback * 2 + 2:
        return 0.0
    cur_h = float(highs[-lookback:].max())
    cur_l = float(lows[-lookback:].min())
    prior_h = float(highs[-(lookback * 2):-lookback].max())
    prior_l = float(lows[-(lookback * 2):-lookback].min())
    if cur_h > prior_h:
        return 1.0
    if cur_l < prior_l:
        return -1.0
    return 0.0


def _ichimoku_live(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
    """Cloud thickness (%ceny) + TK cross. Okresy standardowe 9/26/52, liczone
    WYLACZNIE z okien do biezacej swiecy (bez przesuniecia Senkou w przyszlosc -
    tu liczymy wartosc biezaca chmury, nie jej klasyczna wizualizacje +26 do przodu)."""
    if len(highs) < 52 or closes[-1] <= 0:
        return 0.0, 0.0
    tenkan = float((highs[-9:].max() + lows[-9:].min()) / 2)
    kijun = float((highs[-26:].max() + lows[-26:].min()) / 2)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = float((highs[-52:].max() + lows[-52:].min()) / 2)
    cur = float(closes[-1])
    thickness = abs(senkou_a - senkou_b) / cur * 100
    tk_cross = 1.0 if tenkan > kijun else (-1.0 if tenkan < kijun else 0.0)
    return float(thickness), float(tk_cross)


def _volume_delta_imbalance_live(closes: np.ndarray, volumes: np.ndarray, window: int = 20) -> float:
    """CVD proxy: (wolumen swiec wzrostowych - spadkowych) / wolumen total w oknie."""
    if len(closes) < window + 1 or len(volumes) < window:
        return 0.0
    rets = np.diff(closes[-(window + 1):])
    v = volumes[-window:]
    buy = float(v[rets > 0].sum())
    sell = float(v[rets < 0].sum())
    total = float(v.sum())
    if total < 1e-8:
        return 0.0
    return float((buy - sell) / total)


def _orderblock_strength_live(closes: np.ndarray, lookback: int = 20) -> float:
    """Sila najsilniejszego impulsu w oknie, skalowana odwrotnie do wieku (proxy
    'order block' - swieca inicjujaca impuls jest tym silniejszym sygnalem, im
    swiezszy)."""
    if len(closes) < lookback + 2:
        return 0.0
    rets = np.diff(closes[-(lookback + 1):]) / closes[-(lookback + 1):-1]
    idx = int(np.argmax(np.abs(rets)))
    age = lookback - idx
    return float(abs(float(rets[idx])) * 100 / max(age, 1))


# ── Derywaty LIVE z magazynu (fix 2026-07-20) ────────────────────────────────
# build_features_live przyjmowala oi_change_24h/oi_zscore_30d/ls_ratio jako
# PARAMETRY z domyslnym 0 - a engine._compute_features ich NIE liczyl (podawal
# tylko funding_rate + oi_total_log). Efekt: modele fit2/cat (oparte na
# derywatach) dostawaly w LIVE zera dla kluczowych cech -> same NEUTRAL, zero
# pozycji od zmiany configow. Helper liczy NAJNOWSZE wartosci z tych samych
# parquetow i wzorow co backtester (spojnosc trening<->backtest<->live).
_DERIV_WH = Path("/root/ProjektHAI/data_warehouse/derivatives") if False else None

def latest_deriv_live(symbol: str) -> Dict[str, float]:
    """Najnowsze cechy derywatow dla symbolu (funding/OI/ls_ratio/taker) z
    magazynu. Zwraca neutralne wartosci gdy brak danych. Cache per-symbol 1h
    (dane dzienne/godzinowe, nie ma sensu czytac parquet co petle)."""
    from pathlib import Path as _P
    import pandas as _pd, numpy as _np, time as _t
    global _DERIV_LIVE_CACHE, _DERIV_LIVE_TS
    try:
        _DERIV_LIVE_CACHE
    except NameError:
        _DERIV_LIVE_CACHE = {}; _DERIV_LIVE_TS = {}
    stem = symbol.split("/")[0].split(":")[0].replace("_", "")
    now = _t.time()
    if stem in _DERIV_LIVE_CACHE and (now - _DERIV_LIVE_TS.get(stem, 0)) < 3600:
        return _DERIV_LIVE_CACHE[stem]
    wh = _P(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / "derivatives"
    out = {"funding_rate": 0.0, "funding_change_24h": 0.0, "oi_total_log": 0.0,
           "oi_change_24h": 0.0, "oi_zscore_30d": 0.0, "taker_buy_ratio": 0.5,
           "ls_ratio": 1.0, "ls_ratio_chg_24h": 0.0}

    def _hourly(path, col):
        """Wczytaj serie i ZREINDEXUJ do siatki GODZINOWEJ (ffill) - KLUCZOWE:
        zrodla maja rozna granulacje (OI dzienne, funding 8h, ls/taker 1h).
        Backtester ffilluje wszystko do godzin przed liczeniem cech, wiec
        24-krokowe lookbacki i rolling(720) znacza to samo (24h / 30 dni) w
        KAZDYM zrodle. Live MUSI robic identycznie, inaczej oi_zscore/funding
        _change licza sie na innym oknie niz trening (fix 2026-07-21, wykryty
        przy weryfikacji semantyki: OI a[-25] na dziennych = 25 DNI zamiast 24h,
        rolling(30*24) na dziennych = 720 DNI zamiast 30)."""
        if not path.exists():
            return None
        v = _pd.read_parquet(path).dropna()
        if col not in v.columns or v.empty:
            return None
        s = _pd.Series(v[col].values.astype(float),
                       index=_pd.to_datetime(v["timestamp"])).sort_index()
        # godzinowa siatka od pierwszego do ostatniego ts, ffill
        idx = _pd.date_range(s.index[0].floor("h"), s.index[-1].floor("h"), freq="h")
        return s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)

    try:
        fr = _hourly(wh / "funding_rates" / f"{stem}.parquet", "funding_rate")
        if fr is None:
            fr = _hourly(wh / "funding_rates" / f"{stem}.parquet", "close")
        if fr is not None and len(fr):
            a = fr.values * 100
            out["funding_rate"] = float(a[-1])
            out["funding_change_24h"] = float(a[-1] - a[-25]) if len(a) > 24 else 0.0
    except Exception:
        pass
    try:
        oi = _hourly(wh / "open_interest" / f"{stem}.parquet", "close")
        if oi is not None and len(oi):
            a = oi.values
            out["oi_total_log"] = float(_np.log1p(a[-1]))
            if len(a) > 24 and a[-25] > 0:
                out["oi_change_24h"] = float((a[-1] - a[-25]) / a[-25])
            r = oi.rolling(30 * 24, min_periods=24)  # 720h = 30 dni na siatce godzinowej
            mean = r.mean().values[-1]; std = r.std().values[-1]
            if std and not _np.isnan(std):
                out["oi_zscore_30d"] = float(_np.clip((a[-1] - mean) / max(std, 1e-8), -5, 5))
    except Exception:
        pass
    try:
        tk = _hourly(wh / "taker_ratio" / f"{stem}.parquet", "taker_buy_ratio")
        if tk is not None and len(tk):
            out["taker_buy_ratio"] = float(tk.values[-1])
    except Exception:
        pass
    try:
        ls = _hourly(wh / "ls_ratio" / f"{stem}.parquet", "ls_ratio")
        if ls is not None and len(ls):
            a = ls.values
            out["ls_ratio"] = float(a[-1])
            out["ls_ratio_chg_24h"] = float(a[-1] - a[-25]) if len(a) > 24 else 0.0
    except Exception:
        pass
    _DERIV_LIVE_CACHE[stem] = out; _DERIV_LIVE_TS[stem] = now
    return out


def build_features_live(
    strategy,
    prices_1h: List[float],
    prices_4h: List[float],
    prices_1d: List[float],
    volumes_1h: List[float],
    funding_rate: float = 0.0,
    funding_change_24h: float = 0.0,
    oi_total_log: float = 0.0,
    oi_change_24h: float = 0.0,
    oi_zscore_30d: float = 0.0,
    funding_extreme: float = 0.0,
    timestamp: Optional[datetime] = None,
    # v8.4 new signals
    highs_1h: Optional[List[float]] = None,
    lows_1h: Optional[List[float]] = None,
    taker_buy_ratio: float = 0.5,
    ls_ratio: float = 1.0,
    ls_ratio_chg_24h: float = 0.0,
    # === PARYTET LIVE dla cech RF (2026-08-07) ===
    # RF-sniper-v7 uzywa dist_above_liq/dist_below_liq/cvd_x_adx. Wszystkie trzy
    # liczyly sie w treningu (ml_trainer) i backtescie (backtester), ale NIE
    # tutaj — model dawal WR 70.8% w WFV i byl NIEWDRAZALNY. Zlapane przez
    # scripts/mapa_cech.py.
    symbol: Optional[str] = None,        # potrzebny do mapy likwidacji (OI/ls z magazynu)
    timestamps_1h: Optional[List] = None,  # siatka 1h do wyrownania OI/ls
    of_cvd_chg_24h: float = 0.0,         # zmiana CVD 24h; brak orderflow -> 0.0 (jak trening)
) -> Optional[Dict[str, float]]:
    if not prices_1h or len(prices_1h) < MIN_PRICES_1H:
        return None
    if not volumes_1h or len(volumes_1h) < 30:
        return None
    try:
        cur = float(prices_1h[-1])
        if cur <= 0:
            return None

        rsi = float(strategy.calculate_rsi(prices_1h, 14))
        ema_f = float(strategy.calculate_ema(prices_1h, 9))
        ema_s = float(strategy.calculate_ema(prices_1h, 21))
        ema_m = float(strategy.calculate_ema(prices_1h, 50))
        atr = float(strategy.calculate_atr(prices_1h, 14))
        momentum = float(strategy.calculate_roc(prices_1h, 10))

        atr_pct = (atr / cur * 100) if cur > 0 else 0.0
        ema_fast_r = ((cur / ema_f - 1) * 100) if ema_f > 0 else 0.0
        ema_slow_r = ((cur / ema_s - 1) * 100) if ema_s > 0 else 0.0
        ema_mid_r  = ((cur / ema_m - 1) * 100) if ema_m > 0 else 0.0
        trend_1h   = 1.0 if ema_slow_r > 0.5 else (-1.0 if ema_slow_r < -0.5 else 0.0)

        if prices_4h and len(prices_4h) >= MIN_PRICES_4H:
            rsi_4h = float(strategy.calculate_rsi(prices_4h, 14))
            trend_4h = _trend_jak_trening(prices_4h)   # parytet z treningiem (2026-08-08)
        else:
            rsi_4h = 50.0
            trend_4h = 0.0

        if prices_1d and len(prices_1d) >= MIN_PRICES_1D:
            rsi_1d = float(strategy.calculate_rsi(prices_1d, 14))
            trend_1d = _trend_jak_trening(prices_1d)   # parytet z treningiem (2026-08-08)
        else:
            rsi_1d = 50.0
            trend_1d = 0.0

        vol_window = np.asarray(volumes_1h[-30:], dtype=np.float64)
        avg_vol = float(vol_window.mean())
        vol_std  = float(vol_window.std()) if vol_window.std() > 0 else 1.0
        last_vol = float(volumes_1h[-2] if len(volumes_1h) >= 2 else volumes_1h[-1])
        volume_ratio  = (last_vol / avg_vol) if avg_vol > 0 else 1.0
        volume_zscore = (last_vol - avg_vol) / vol_std if vol_std > 0 else 0.0

        if len(prices_1h) >= 20:
            bb_window = np.asarray(prices_1h[-20:], dtype=np.float64)
            bb_mean = float(bb_window.mean())
            bb_std = float(bb_window.std())
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            bb_width = bb_upper - bb_lower
            price_position_bb = ((cur - bb_lower) / bb_width) if bb_width > 0 else 0.5
            bb_bandwidth_pct = (4 * bb_std / bb_mean) if bb_mean > 0 else 0.0
        else:
            price_position_bb = 0.5
            bb_bandwidth_pct = 0.0

        adx_14 = _calc_adx_live(prices_1h)

        # Intraday seasonality
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        hour_val = now.hour + now.minute / 60.0
        hour_sin = float(np.sin(2 * np.pi * hour_val / 24))
        hour_cos = float(np.cos(2 * np.pi * hour_val / 24))
        dow = float(now.weekday())

        # === NEURAL EXTRA FEATURES (bear market calibration) ===
        # EMA200 distance — positive = above 200 EMA (bull), negative = below (bear)
        if len(prices_1h) >= 200:
            ema_200 = float(strategy.calculate_ema(prices_1h, 200))
            ema_200_dist_pct = round((cur / ema_200 - 1) * 100, 4) if ema_200 > 0 else 0.0
        else:
            ema_200_dist_pct = 0.0

        # RSI slope over last 5 bars (acceleration/deceleration)
        if len(prices_1h) >= 20:
            rsi_5ago = float(strategy.calculate_rsi(prices_1h[:-5], 14)) if len(prices_1h) > 5 else rsi
            rsi_slope_5h = round(rsi - rsi_5ago, 3)
        else:
            rsi_slope_5h = 0.0

        # Dywergencja cena vs RSI (okno 20) — parytet 1:1 z ml_trainer.calc_divergence.
        # +1 byczy rozjazd (cena nizej, RSI wyzej), -1 niedzwiedzi, 0 brak.
        LB_DIV = 20
        if len(prices_1h) > LB_DIV + 14:
            _closes = np.asarray(prices_1h, dtype=np.float64)
            _rsi_ser = _rsi_series_wilder(_closes, 14)
            dprice = _closes[-1] - _closes[-1 - LB_DIV]
            drsi = _rsi_ser[-1] - _rsi_ser[-1 - LB_DIV]
            div_rsi = 1.0 if (dprice < 0 and drsi > 0) else (-1.0 if (dprice > 0 and drsi < 0) else 0.0)
        else:
            div_rsi = 0.0

        # S&D proximity (parytet z ml_trainer.calc_sd_proximity). Wymaga highs/lows.
        # +  = blisko strefy PODAZY (kontrarianski short), -  = blisko POPYTU (long).
        WIN_SD = 50
        sd_prox = 0.0
        if (highs_1h is not None and lows_1h is not None
                and len(highs_1h) == len(prices_1h) and len(prices_1h) >= WIN_SD + 2):
            _h = np.asarray(highs_1h, dtype=np.float64)
            _l = np.asarray(lows_1h, dtype=np.float64)
            _c = np.asarray(prices_1h, dtype=np.float64)
            _atr_ser = _atr_series_wilder(_h, _l, _c, 14)
            atr_last = _atr_ser[-1]
            last_i = len(_c) - 1
            demand_lvl = np.nan
            supply_lvl = np.nan
            for i in range(last_i, WIN_SD - 2, -1):
                if np.isnan(demand_lvl) and _c[i] > _c[i - 1] and _l[i] == _l[i - WIN_SD + 1:i + 1].min():
                    demand_lvl = _l[i]
                if np.isnan(supply_lvl) and _c[i] < _c[i - 1] and _h[i] == _h[i - WIN_SD + 1:i + 1].max():
                    supply_lvl = _h[i]
                if not np.isnan(demand_lvl) and not np.isnan(supply_lvl):
                    break
            if atr_last > 0 and not np.isnan(demand_lvl) and not np.isnan(supply_lvl):
                val = (cur - demand_lvl) / atr_last - (supply_lvl - cur) / atr_last
                sd_prox = float(np.clip(val, -15.0, 15.0))

        # Wiek trendu: bary od przeciecia EMA20/50 ze znakiem (parytet z
        # ml_trainer.calc_bars_since_cross — ta sama EMA). Insight v4-pro.
        bars_cross = 0.0
        if len(prices_1h) >= 52:
            _cc = np.asarray(prices_1h, dtype=np.float64)
            _ef = _ema_series_live(_cc, 20)
            _es = _ema_series_live(_cc, 50)
            _spread = _ef - _es
            _sg = np.sign(_spread)
            _bars = 0
            for _i in range(1, len(_cc)):
                if _sg[_i] != _sg[_i - 1] and _sg[_i] != 0:
                    _bars = 0
                else:
                    _bars += 1
            _last_spread = _spread[-1]
            bars_cross = float(np.clip(_bars * (1.0 if _last_spread > 0 else (-1.0 if _last_spread < 0 else 0.0)), -100.0, 100.0))

        # Volume trend: recent 5 bars vs rolling 30 bars
        vol_arr = np.asarray(volumes_1h[-30:], dtype=np.float64)
        vol_5 = float(vol_arr[-5:].mean()) if len(vol_arr) >= 5 else float(vol_arr.mean())
        vol_trend = round(vol_5 / avg_vol, 4) if avg_vol > 0 else 1.0

        # v8.4: VWAP deviation (24h rolling)
        vwap_window = 24
        if (highs_1h is not None and lows_1h is not None
                and len(highs_1h) >= vwap_window and len(lows_1h) >= vwap_window
                and len(volumes_1h) >= vwap_window):
            h_w  = np.asarray(highs_1h[-vwap_window:], dtype=np.float64)
            l_w  = np.asarray(lows_1h[-vwap_window:], dtype=np.float64)
            c_w  = np.asarray(prices_1h[-vwap_window:], dtype=np.float64)
            v_w  = np.asarray(volumes_1h[-vwap_window:], dtype=np.float64)
            tp   = (h_w + l_w + c_w) / 3
            tvol = float(v_w.sum())
            vwap_24h = float((tp * v_w).sum() / tvol) if tvol > 0 else cur
            vwap_dev = round((cur / vwap_24h - 1) * 100, 4) if vwap_24h > 0 else 0.0
        else:
            vwap_dev = 0.0

        # === Nowe cechy (audyt sesji 2026-07-03): MACD, PhantomFlow S/R, Fibonacci ===
        macd_hist = _macd_hist_live(prices_1h)
        if highs_1h is not None and lows_1h is not None and len(highs_1h) == len(prices_1h):
            sh_level, sh_strength, sl_level, sl_strength = _swing_sr_live(highs_1h, lows_1h)
            dist_to_res = ((sh_level - cur) / cur * 100) if sh_level is not None else 999.0
            dist_to_sup = ((cur - sl_level) / cur * 100) if sl_level is not None else 999.0
            if dist_to_res < dist_to_sup:
                sr_dist_pct = round(-dist_to_res, 4)
                sr_node_strength = sh_strength
            else:
                sr_dist_pct = round(dist_to_sup, 4)
                sr_node_strength = sl_strength
            fib_dist_pct = _fib_dist_live(cur, sh_level, sl_level)
        else:
            sr_dist_pct = 0.0
            sr_node_strength = 0.0
            fib_dist_pct = 0.0

        macro_ext = _get_macro_extended_live()

        # === MAPA LIKWIDACJI — parytet live (2026-08-07) ===
        # NIE wlasna kopia formuly: wolamy DOKLADNIE te sama funkcje, ktorej
        # uzywaja ml_trainer i backtester (liqmap_features naglowek: "WSPOLNE
        # zrodlo dla treningu i backtestu (parzystosc!)") — live bylo brakujacym
        # trzecim odbiorca. Wlasna implementacja rozjechalaby sie przy pierwszej
        # zmianie parametrow (LEVERAGES/DECAY_H sa tam ZAMROZONE).
        # Fallback to DIST_DEFAULT (30.0), NIGDY 0.0 — zero znaczy "klaster
        # dokladnie na cenie", czyli semantyke ODWROTNA, i jest OOD dla scalera.
        _liq_feats: Dict[str, float] = {}
        try:
            from .liqmap_features import compute_liq_dist_from_warehouse, DIST_DEFAULT
        except Exception:
            DIST_DEFAULT = 30.0
            compute_liq_dist_from_warehouse = None
        _db = _da = DIST_DEFAULT
        if compute_liq_dist_from_warehouse and symbol and timestamps_1h is not None \
                and highs_1h is not None and lows_1h is not None:
            try:
                import pandas as pd  # lokalnie: features.py nie importuje pandas globalnie
                _n = min(len(timestamps_1h), len(prices_1h), len(highs_1h), len(lows_1h))
                if _n >= 50:
                    _df1h = pd.DataFrame({
                        "timestamp": pd.to_datetime(pd.Series(list(timestamps_1h)[-_n:]),
                                                    utc=True).dt.tz_localize(None),
                        "close": list(prices_1h)[-_n:],
                        "high": list(highs_1h)[-_n:],
                        "low": list(lows_1h)[-_n:]})
                    _b, _a = compute_liq_dist_from_warehouse(_df1h, symbol)
                    if len(_b) and len(_a):
                        _db = float(_b[-1]) if np.isfinite(_b[-1]) else DIST_DEFAULT
                        _da = float(_a[-1]) if np.isfinite(_a[-1]) else DIST_DEFAULT
            except Exception as _liq_e:
                logger.debug(f"liq live skip ({_liq_e}) -> DIST_DEFAULT")
        _liq_feats = {"dist_below_liq": round(_db, 4), "dist_above_liq": round(_da, 4)}

        # === EKSPERYMENTALNE CECHY (2026-08-05, inwentaryzacja §5.4) ===
        # X_* są liczone i zwracane w rekordzie, ale NIE wchodzą do FEATURE_NAMES /
        # MODEL_FEATURES — żaden model ich nie używa. Cel: mają być na stanie,
        # gotowe do podpięcia, ale zero wpływu na istniejące modele/backtest/live.
        x_feats: Dict[str, float] = {}
        try:
            c = np.asarray(prices_1h, dtype=np.float64)
            # realized volatility (1h returns) 6h/24h + Parkinson (high-low)
            if len(c) > 25:
                rets = np.diff(c) / c[:-1]
                x_feats["x_real_vol_6h"] = round(float(rets[-6:].std() * 100), 4) if len(rets) >= 6 else 0.0
                x_feats["x_real_vol_24h"] = round(float(rets[-24:].std() * 100), 4) if len(rets) >= 24 else 0.0
                if highs_1h is not None and lows_1h is not None and len(highs_1h) > 25 and len(lows_1h) > 25:
                    _hh = np.asarray(highs_1h, dtype=np.float64)[-25:]
                    _ll = np.asarray(lows_1h, dtype=np.float64)[-25:]
                    hl = np.log(_hh / _ll)
                    park = np.sqrt((1.0 / (4 * np.log(2))) * (hl**2).mean()) * 100
                    x_feats["x_parkinson_24h"] = round(float(park), 4)
            # RSI wielookresowo (7/21) z tego samego Wildera co rsi
            if len(c) > 25:
                _r7 = _rsi_series_wilder(c, 7)
                _r21 = _rsi_series_wilder(c, 21)
                x_feats["x_rsi_7"] = round(float(_r7[-1]), 2)
                x_feats["x_rsi_21"] = round(float(_r21[-1]), 2)
            # ROC wielookresowo (4/24/168)
            if len(c) > 170:
                for _bars in (4, 24, 168):
                    x_feats[f"x_roc_{_bars}"] = round(float((c[-1] / c[-1 - _bars] - 1) * 100), 4) if c[-1 - _bars] else 0.0
            # EMA ratio pary (9/21, 21/55, 50/200) — cross/herbst
            if len(c) > 200:
                for _f, _s in ((9, 21), (21, 55), (50, 200)):
                    _ef = _ema_series_live(c, _f)[-1]
                    _es = _ema_series_live(c, _s)[-1]
                    x_feats[f"x_ema{_f}_over_{_s}"] = round(float(_ef / _es - 1), 6) if _es else 0.0
            # OBV slope: znormalizowany gradient OBV (stosunek netto buy-pressure
            # do calkowitego wolumenu w oknie) — NIE dzielenie przez obv[0] (zawsze 0).
            if volumes_1h and len(volumes_1h) > 25 and len(c) > 25:
                _v = np.asarray(volumes_1h[-25:], dtype=np.float64)
                _cc = c[-26:]
                _dv = np.sign(np.diff(_cc))
                obv = np.zeros(25); obv[0] = 0.0
                for _i in range(1, 25):
                    obv[_i] = obv[_i - 1] + (_dv[_i - 1] * _v[_i - 1] if _dv[_i - 1] != 0 else 0.0)
                _vsum = float(_v.sum())
                x_feats["x_obv_slope_24h"] = round(float(obv[-1]) / (_vsum + 1e-8), 6)
            # weekend flag (sob/niedz UTC)
            x_feats["x_weekend"] = 1.0 if now.weekday() in (5, 6) else 0.0
            # BTC lead-lag proxy: odchylenie zachowania symbolu od dominacji BTC.
            # Gdy BTC dominacja rosnie, a symbol nie nadaza (stoi/neutralnie) -> okazja
            # catch-up (hipoteza §6.2). NIE duplikat btc_dominance_chg: tu porownujemy
            # temp BTC-dom vs momentum symbolu.
            _btc_chg = float(macro_ext.get("btc_dominance_chg", 0.0))
            x_feats["x_btc_leadlag"] = round(_btc_chg - float(momentum), 4)
        except Exception as _xe:
            logger.debug(f"experimental features error: {_xe}")

        # === CECHY RPTR: RF/ET specjalistyczne (2026-08-05) ===
        # Jak x_feats: liczone i zwracane, ale NIE w FEATURE_NAMES/MODEL_FEATURES -
        # zero wplywu, dopoki config nie doda ich jawnie przez feature_mix.
        rptr_feats: Dict[str, float] = {}
        try:
            _c = np.asarray(prices_1h, dtype=np.float64)
            _has_hl = (highs_1h is not None and lows_1h is not None
                       and len(highs_1h) == len(prices_1h) and len(lows_1h) == len(prices_1h))
            _h = np.asarray(highs_1h, dtype=np.float64) if _has_hl else None
            _l = np.asarray(lows_1h, dtype=np.float64) if _has_hl else None
            _v = np.asarray(volumes_1h, dtype=np.float64) if volumes_1h else None

            # --- RF specialist ---
            if len(_c) > 20:
                _rsi_ser = _rsi_series_wilder(_c, 14)
                rptr_feats["r_stoch_rsi"] = round(_stoch_rsi_live(_rsi_ser, 14), 3)
            if _has_hl and len(_c) > 20:
                rptr_feats["r_cci_20"] = round(_cci_live(_h, _l, _c, 20), 3)
                rptr_feats["r_di_spread"] = round(_di_spread_live(_h, _l, _c, 14), 3)
                rptr_feats["r_range_compression"] = round(_range_compression_live(_h, _l, 6, 24), 4)
                rptr_feats["r_structure_bias"] = _structure_bias_live(_h, _l, 5)
            if len(_c) > 40:
                _e12 = _ema_series_live(_c[-200:], 12)
                _e26 = _ema_series_live(_c[-200:], 26)
                _macd_line = _e12 - _e26
                _signal = _ema_series_live(_macd_line, 9)
                rptr_feats["r_macd_signal_dist"] = round(float(_macd_line[-1] - _signal[-1]) / float(_c[-1]) * 100, 4) if _c[-1] else 0.0
            if len(_c) > 21:
                rptr_feats["r_momentum_20"] = round(float((_c[-1] / _c[-21] - 1) * 100), 4)
            if len(_c) > 20:
                _atr_ser = _atr_series_wilder(_h, _l, _c, 14) if _has_hl else None
                if _atr_ser is not None and len(_atr_ser) > 20:
                    _aw = _atr_ser[-20:]
                    _astd = float(_aw.std())
                    rptr_feats["r_atr_zscore_20"] = round(float((_atr_ser[-1] - _aw.mean()) / _astd), 4) if _astd > 1e-8 else 0.0
            if volumes_1h and len(volumes_1h) > 50:
                _v50 = np.asarray(volumes_1h[-50:], dtype=np.float64)
                _v50std = float(_v50.std())
                rptr_feats["r_volume_zscore_50"] = round(float((_v50[-1] - _v50.mean()) / _v50std), 4) if _v50std > 1e-8 else 0.0
            rptr_feats["r_vwap_distance"] = vwap_dev  # alias vwap_dev (juz liczone wyzej)
            rptr_feats["r_htf_trend_4h"] = trend_4h   # alias (juz liczone wyzej)

            # --- ET specialist (SMC + Ichimoku + interakcje) ---
            if _has_hl and len(_c) > 25:
                _sh, _sl = _liquidity_sweep_live(_h, _l, _c, 20)
                rptr_feats["e_liquidity_sweep_high"] = _sh
                rptr_feats["e_liquidity_sweep_low"] = _sl
                _fvg_size, _fvg_dir = _fvg_live(_h, _l, _c)
                rptr_feats["e_fvg_size"] = round(_fvg_size, 4)
                rptr_feats["e_fvg_direction"] = _fvg_dir
                rptr_feats["e_bos_choch"] = _bos_choch_live(_h, _l, 10)
                rptr_feats["e_equal_highs_lows_count"] = _equal_highs_lows_live(_h, _l, 20, 0.0015)
            if _has_hl and len(_c) > 52:
                _thick, _tk = _ichimoku_live(_h, _l, _c)
                rptr_feats["e_ichimoku_cloud_thickness"] = round(_thick, 4)
                rptr_feats["e_tk_cross"] = _tk
            if _v is not None and len(_c) > 25:
                rptr_feats["e_volume_delta_imbalance"] = round(_volume_delta_imbalance_live(_c, _v, 20), 4)
            if len(_c) > 25:
                rptr_feats["e_orderblock_strength"] = round(_orderblock_strength_live(_c, 20), 4)
            rptr_feats["e_phantomflow_score"] = sr_node_strength  # alias (juz liczone wyzej, PhantomFlow S/R)
            rptr_feats["e_rsi_x_funding"] = round(float(rsi) * float(funding_rate), 4)
            rptr_feats["e_atr_pct_x_vol_zscore"] = round(float(atr_pct) * float(volume_zscore), 4)
        except Exception as _re:
            logger.debug(f"rptr features error: {_re}")

        return {
            **x_feats,
            **rptr_feats,
            "rsi": round(rsi, 2),
            "rsi_4h": round(rsi_4h, 2),
            "rsi_1d": round(rsi_1d, 2),
            "ema_fast_r": round(ema_fast_r, 4),
            "ema_slow_r": round(ema_slow_r, 4),
            "ema_mid_r": round(ema_mid_r, 4),
            "trend_1h": trend_1h,
            "atr_pct": round(atr_pct, 4),
            "momentum": round(momentum, 4),
            "trend_4h": trend_4h,
            "trend_1d": trend_1d,
            "volume_ratio": round(volume_ratio, 4),
            "volume_zscore": round(volume_zscore, 4),
            "funding_rate": float(funding_rate),
            "funding_change_24h": float(funding_change_24h),
            "price_position_bb": round(price_position_bb, 4),
            "bb_bandwidth_pct": round(bb_bandwidth_pct, 4),
            "oi_total_log": float(oi_total_log),
            "oi_change_24h": float(oi_change_24h),
            "oi_zscore_30d": float(oi_zscore_30d),
            # ls_ratio + interakcja (fix 2026-07-20, spojne z ml_trainer/backtester)
            "ls_ratio": float(ls_ratio),
            "ls_ratio_chg_24h": float(ls_ratio_chg_24h),
            "funding_x_oizscore": float(funding_rate) * float(oi_zscore_30d),
            "hour_sin": round(hour_sin, 4),
            "hour_cos": round(hour_cos, 4),
            "day_of_week": dow,
            "adx_14": round(adx_14, 1),
            # Neural extra
            "ema_200_dist_pct": ema_200_dist_pct,
            "rsi_slope_5h": rsi_slope_5h,
            "vol_trend": vol_trend,
            # v8.4
            "vwap_dev": vwap_dev,
            "taker_buy_ratio": round(float(taker_buy_ratio), 6),
            "macd_hist": macd_hist,
            "div_rsi": div_rsi,
            "sd_prox": sd_prox,
            "bars_cross": bars_cross,
            "sr_dist_pct": sr_dist_pct,
            "sr_node_strength": sr_node_strength,
            "fib_dist_pct": fib_dist_pct,
            # Macro extended (audyt 2026-07-05) - wpiete tez tu (nie tylko
            # trening), zeby CAT faktycznie z tego korzystal w live/backtest
            "gold_chg": macro_ext.get("gold_chg", 0.0),
            "oil_wti_chg": macro_ext.get("oil_wti_chg", 0.0),
            "sp500_chg": macro_ext.get("sp500_chg", 0.0),
            "vix_chg": macro_ext.get("vix_chg", 0.0),
            "us10y_chg": macro_ext.get("us10y_yield_chg", 0.0),
            "dxy_chg": macro_ext.get("dxy_chg", 0.0),
            "btc_dominance_chg": macro_ext.get("btc_dominance_chg", 0.0),
            # === PARYTET LIVE cech RF (2026-08-07) ===
            # cvd_x_adx: wzor odtworzony EMPIRYCZNIE z datasetu (4.1M wierszy,
            # blad wzgledny 0.0) = of_cvd_chg_24h * adx_14. Gdy wolajacy nie ma
            # orderflow, of_cvd_chg_24h=0.0 -> cecha 0.0, dokladnie jak w treningu
            # (tam 1.4M z 4.1M wierszy ma zero z tego samego powodu).
            "of_cvd_chg_24h": float(of_cvd_chg_24h),
            "cvd_x_adx": float(of_cvd_chg_24h) * round(adx_14, 1),
            **_liq_feats,
        }
    except Exception as e:
        logger.error(f"build_features_live error: {e}", exc_info=False)
        return None

def features_to_vector(features: Dict[str, float]) -> List[float]:
    out = []
    for name in FEATURE_NAMES:
        if name not in features:
            logger.warning(f"features_to_vector: brak '{name}', używam 0.0")
            out.append(0.0)
        else:
            out.append(float(features[name]))
    return out


def build_feature_sequence_live(
    strategy,
    prices_1h: List[float],
    prices_4h: List[float],
    prices_1d: List[float],
    volumes_1h: List[float],
    seq_len: int = 20,
    **kwargs,
) -> Optional[np.ndarray]:
    """Build (seq_len, 22) feature matrix for LSTM/TCN inference.

    Uses the last seq_len 1h bars. 4h/1d features taken from latest bar
    (they change slowly). Returns None if insufficient data.
    """
    n = len(prices_1h)
    if n < seq_len + MIN_PRICES_1H or len(volumes_1h) < seq_len + 30:
        return None
    rows = []
    for k in range(seq_len, 0, -1):  # k=seq_len → oldest bar, k=1 → current
        end = n - k + 1
        p1h = prices_1h[:end]
        v1h = volumes_1h[:end]
        feat = build_features_live(
            strategy=strategy,
            prices_1h=p1h, prices_4h=prices_4h, prices_1d=prices_1d,
            volumes_1h=v1h, **kwargs,
        )
        row = [feat.get(f, 0.0) for f in NEURAL_FEATURE_NAMES] if feat else [0.0] * len(NEURAL_FEATURE_NAMES)
        rows.append(row)
    return np.array(rows, dtype=np.float32)  # (seq_len, 22)

# ===========================================
# HAI_EPV Engine ver.10 Final — core/backtester.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - run_full_ai() - pelny backtest (wektoryzowane wskazniki, doktryna produkcyjna:
#   BB extreme, ADX>=22, session/regime filtry, slippage+funding, circuit breaker)
# - run_wfv() - Walk-Forward Validation (N okien, embargo, werdykt GO/NO_GO/WARNING)
# - _calc_stats() - agregacja wynikow: PF/WR/DD, model_attribution (kto zlapal
#   transakcje), model_votes + feature_snapshot per trade
# ===========================================
import asyncio
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from .strategies.registry import get_strategy
from .strategies import ai_strategy  # noqa: F401 — rejestracja
from .strategies.ai_strategy import CONTINUATION_ZONE_LONG, CONTINUATION_ZONE_SHORT

# Regime-adaptive doktryna (audyt 2026-07-05, "napisz nowa strategie
# trzymajaca sie doktryn") - globalny toggle, ustawiany przez endpoint PRZED
# wywolaniem run_full_ai()/run_wfv() (patrz routes/trading.py). Domyslnie
# WYLACZONY - zero zmiany istniejacego zachowania backtestu.
_REGIME_ADAPTIVE_MODE = False

# MULTI-HORIZON (2026-08-08): lgb_multi_horizon jest trenowany na STACKOWANYM
# zbiorze [4,6,8,16,24,36,48,72] z 'horizon_hours' jako CECHA wejsciowa.
# Backtester tej cechy nie liczyl -> feat_src.get('horizon_hours', zeros) -> model
# dostawal ZERA (ten sam wzorzec buga co rptr/x_ naprawiony 2026-08-08). W
# predykcji sondujemy kazdy horyzont z osobna i uodredniamy prawdopodobienstwo
# (soft-vote) — to, co model faktycznie widzial w treningu.
MULTI_HORIZONS = [float(x) for x in os.environ.get(
    "HAI_MULTI_HORIZONS", "4,6,8,16,24,36,48,72").split(",")]

# Mechanizm glosowania ensemble (audyt 2026-07-06, "testy mechanizmow
# glosowania") - "weighted" (domyslny, bez zmian) = wazona suma prawdopodobienstw
# per model + prog 0.40. "majority" = kazdy model oddaje 1 glos (LONG/SHORT/
# wstrzymanie wg wlasnego progu 0.52), decyzja wg frakcji glosow > 0.40 i wiekszej
# niz przeciwna strona. "winner_take_all" = MAX zamiast SUMY wazonych glosow -
# decyduje pojedynczy najsilniejszy model w danej swiecy, reszta ignorowana -
# patrz run_simulation_ai() krok 4.
_VOTING_MODE = "weighted"

# Prog decyzyjny - bazowy default systemu (audyt 2026-07-07, na prosbe usera
# "threshold ustaw jakas bazowa default dla systemu maksymalnie wysrodkowany
# l/s"). Zmieniony 0.40 -> 0.35: srodek miedzy testowanym-ostroznym 0.40
# (backtester) a permisywnym-live 0.30 (ensemble) - domyka rozjazd
# backtest<->live (NewHorizonts K4). UWAGA: sam prog symetryczny NIE centruje
# L/S (obecnie 82-99% short) - o stronie decyduje long_score vs short_score,
# prog decyduje tylko CZY wejsc. Nizszy prog wpuszcza marginalnie wiecej
# longow (slabsze sygnaly long przechodza), ale prawdziwe centrowanie wymaga
# progow ASYMETRYCZNYCH (nizszy dla LONG, wyzszy dla SHORT) - do implementacji
# osobno. Skrzywienie short wynika strukturalnie z: bessa w danych (BTC -50%),
# modele uczone bear-heavy, doktryna BB.
_DECISION_THRESHOLD = 0.35

# Progi ASYMETRYCZNE (audyt 2026-07-07, na prosbe usera - prawdziwy lewar
# centrowania L/S, patrz analiza: system 82-99% short strukturalnie). Gdy
# ustawione (nie None), LONG uzywa _THRESHOLD_LONG, SHORT uzywa
# _THRESHOLD_SHORT - zamiast wspolnego _DECISION_THRESHOLD. Idea: obnizyc
# poprzeczke dla LONG (wpuscic wiecej longow) i podniesc dla SHORT (odsiac
# slabe shorty) -> zblizyc liczbe wejsc per strona. None = zachowanie
# symetryczne (fallback na _DECISION_THRESHOLD, pelna wsteczna kompatybilnosc).
_THRESHOLD_LONG = None
_THRESHOLD_SHORT = None

# Confidence-based position sizing (audyt 2026-07-07, z planu "Opusa") -
# domyslnie OFF (bez zmian istniejacego zachowania - sizing zalezy tylko od
# SIZE_SCALE_BY_LOSSES). Gdy True, mnozy istniejacy scale przez dodatkowy
# czynnik 0.5-1.5x zalezny od marginesu pewnosci ponad prog decyzyjny -
# slabe sygnaly (tuz nad progiem) dostaja mniejsza pozycje, mocne wieksza.
# Nigdy wczesniej nie testowane w calej sesji.
_CONF_SIZING_ENABLED = False

# GLEBOKOSC KONSENSUSU (audyt 2026-07-07, K1 z NewHorizonts) - najsilniejszy
# sygnal w danych: liczba modeli zgodnych z kierunkiem przewiduje WR
# MONOTONICZNIE (3/5->64.1%, 4/5->66.7%, 5/5->70.4%), a wazony score jest
# PLASKI. To NIE tryb glosowania - to BRAMKA: trade wazny tylko gdy >=
# _CONSENSUS_MIN modeli zgadza sie z wybranym kierunkiem (glos >0.52).
# 0 = wylaczone (pelna wsteczna kompatybilnosc). Laczy sie z meta-labelem
# (oba filtry aplikowane do _valid).
_CONSENSUS_MIN = 0
# CONSENSUS SIZING (audyt 2026-07-07, "zmienne wagi" - K1 sizing z NewHorizonts).
# Wariant SIZING (nie gating): skaluje WIELKOSC pozycji glebokoscia konsensu -
# wiecej zgodnych modeli = wieksza pozycja (5/5=70% WR > 3/5=64% WR). Analogicznie
# do conf_sizing: scale *= (0.5 + depth/n_models) -> zakres 0.5x (0 zgodnych) do
# 1.5x (pelny konsensus). False = wylaczone.
_CONSENSUS_SIZING = False
# GATE GLOSU (audyt 2026-07-07) - prog powyzej ktorego model "glosuje" na
# kierunek (lp/sp > gate). Domyslnie 0.52 (bez zmian, strojony pod CORE-v1 -
# ostre prawdopodobienstwa). CORE-v2 (3-klasowy ExtraTrees z balansem) ma
# PLASKIE prawdopodobienstwa (max ~0.59, p95 0.41) i przy 0.52 prawie nigdy
# nie glosuje -> 0 transakcji. Nizszy gate (0.40/0.35) pozwala mu grac.
_VOTE_GATE = 0.52
# DOKTRYNA WOLNA (audyt 2026-07-07) - dla modeli TREND-FOLLOWING (CORE-v2).
# Backtester ma doktryne MEAN-REVERSION (long tylko gdy bb<=0.30, short gdy
# bb>=0.70) + wymog doc_dir==batch_action. To DLAWI trend-followera: CORE-v2
# chce kupowac kontynuacje trendu (wysokie bb), a doktryna to blokuje. Gdy
# True: pomijamy KIERUNKOWY filtr BB - model sam decyduje kierunek, zostaja
# tylko filtry SRODOWISKA (szerokosc BB, ADX, sesja). False = bez zmian.
_DOCTRINE_FREE = False
# TRADE_LOG W OKNACH WFV (F0 gen.Dir-v1, 2026-07-18) - parametr zgubiony przy
# migracji instancji do hai_common (stare AD*/routes/trading.py mialy
# include_trade_log, nowy endpoint nie). Gdy True: kazde okno WFV zapisuje
# pelny trade_log (z confidence/dominant_model/model_votes) do JSON wyniku -
# jedyne zrodlo danych do krzywej precyzja-vs-prog i selekcji kierunkowej.
# Koszt: wiekszy JSON (~1-2 MB/okno przy setkach tradow), zero wplywu na wynik.
_INCLUDE_TRADE_LOG = False

logger = logging.getLogger(__name__)

WH_BASE        = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / "ohlcv" / "binance"
BACKTEST_DIR   = Path(__file__).resolve().parent.parent / "data" / "backtest"
NEURAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "neural"

SIZE_SCALE_BY_LOSSES = {0: 1.0, 1: 0.10}
PYRAMID_SL_SCALE     = 0.50
SL_COOLDOWN_MINUTES  = 120

# Ile symboli równolegle — konfigurowalne env-var (audyt 2026-07-03, jak w LAB),
# domyślnie 6 (tyle rdzeni ma VPS) zamiast sztywnych 4
MAX_CONCURRENT_SYMBOLS = int(os.getenv("HAI_MAX_CONCURRENT_SYMBOLS", "6"))


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED INDICATORS — liczone RAZ na całej serii, O(n) zamiast O(n²)
# ─────────────────────────────────────────────────────────────────────────────

def _vec_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 50.0)
    if n < period + 2:
        return out
    delta = np.diff(prices)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = gain[:period].mean()
    avg_l = loss[:period].mean()
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        out[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l > 0 else 100.0
    return out


def _vec_ema(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = prices[:period].mean()
    for i in range(period, n):
        out[i] = prices[i] * alpha + out[i - 1] * (1 - alpha)
    return out


def _vec_atr(candles: List[Dict], period: int = 14) -> np.ndarray:
    n = len(candles)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    highs  = np.array([c["high"]  for c in candles], dtype=np.float64)
    lows   = np.array([c["low"]   for c in candles], dtype=np.float64)
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_val = tr[:period].mean()
    out[period] = atr_val
    for i in range(period, n - 1):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        out[i + 1] = atr_val
    return out


def _vec_bb(prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Zwraca (bb_pos, bb_bw_pct) jako serie."""
    n = len(prices)
    bb_pos    = np.full(n, 0.5)
    bb_bw_pct = np.zeros(n)
    if n < period:
        return bb_pos, bb_bw_pct
    cs  = np.cumsum(prices)
    cs2 = np.cumsum(prices ** 2)
    mean = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
    var  = (cs2[period-1:] - np.concatenate([[0], cs2[:-period]])) / period - mean ** 2
    std  = np.sqrt(np.maximum(var, 0))
    bw   = 4 * std
    pos  = np.where(bw > 0, (prices[period-1:] - (mean - 2*std)) / bw, 0.5)
    bwp  = np.where(mean > 0, bw / mean, 0.0)
    bb_pos[period-1:]    = pos
    bb_bw_pct[period-1:] = bwp
    return bb_pos, bb_bw_pct


def _vec_adx(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX proxy z closes (jak _calc_adx_closes w ai_strategy)."""
    n = len(prices)
    out = np.zeros(n)
    if n < period * 2 + 2:
        return out
    delta = np.diff(prices)
    dmp = np.where(delta > 0, delta, 0.0)
    dmm = np.where(delta < 0, -delta, 0.0)
    trs = np.abs(delta)
    atr = trs[:period].sum()
    pdm = dmp[:period].sum()
    mdm = dmm[:period].sum()
    dx_buf = np.zeros(len(trs) - period)
    for i in range(period, len(trs)):
        atr = atr - atr / period + trs[i]
        pdm = pdm - pdm / period + dmp[i]
        mdm = mdm - mdm / period + dmm[i]
        pdi = pdm / atr * 100 if atr > 0 else 0.0
        mdi = mdm / atr * 100 if atr > 0 else 0.0
        dx_buf[i - period] = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0.0
    # Wilder's smooth of DX
    if len(dx_buf) < period:
        return out
    adx_val = dx_buf[:period].mean()
    j = period * 2
    out[j] = adx_val
    for i in range(period, len(dx_buf)):
        adx_val = (adx_val * (period - 1) + dx_buf[i]) / period
        out[i + period + 1] = adx_val
    return out


def _vec_stoch(prices: np.ndarray, period: int = 14, smooth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic %K/%D na closes."""
    n = len(prices)
    k = np.full(n, 50.0)
    for i in range(period - 1, n):
        chunk = prices[max(0, i - period + 1): i + 1]
        lo, hi = chunk.min(), chunk.max()
        k[i] = 100.0 * (prices[i] - lo) / (hi - lo) if hi > lo else 50.0
    d = np.full(n, 50.0)
    for i in range(smooth - 1, n):
        d[i] = k[max(0, i - smooth + 1): i + 1].mean()
    return k, d


def _vec_roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
    out = np.zeros(len(prices))
    for i in range(period, len(prices)):
        base = prices[i - period]
        out[i] = (prices[i] / base - 1) * 100 if base > 0 else 0.0
    return out


def _vec_macd_hist(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram, znormalizowany do % ceny — spojny z ml_trainer.calc_macd_hist.

    _vec_ema zwraca NaN w okresie rozgrzewki (warmup); rekurencyjne EMA nad
    takim wejsciem (linia sygnalu) zatruloby sie NaN na cala reszte serii —
    dlatego czyscimy NaN->0 na macd_line przed drugim przebiegiem EMA.
    """
    ema_f = _vec_ema(prices, fast)
    ema_s = _vec_ema(prices, slow)
    macd_line = np.nan_to_num(ema_f - ema_s, nan=0.0)
    signal_line = _vec_ema(macd_line, signal)
    hist = np.nan_to_num(macd_line - signal_line, nan=0.0)
    return np.where(prices > 0, hist / prices * 100, 0.0)


def _vec_volume_zscore(volumes: np.ndarray, period: int = 30) -> np.ndarray:
    vs = pd.Series(volumes)
    rmean = vs.rolling(period, min_periods=1).mean()
    rstd = vs.rolling(period, min_periods=1).std().replace(0, 1.0).fillna(1.0)
    return ((vs - rmean) / rstd).fillna(0.0).values


def _vec_swing_sr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                   lookback: int = 5, tolerance: float = 0.003,
                   strength_window: int = 100):
    """PhantomFlow reaktywacja, spojna z ml_trainer.calc_swing_sr — patrz tam po opis.

    No-lookahead: swing przy idx potwierdzony dopiero w confirm_idx=idx+lookback,
    sila liczona tylko z danych <= confirm_idx (zero przyszlosci)."""
    n = len(closes)
    if n < lookback * 4:
        z = np.zeros(n)
        return z, z, np.full(n, np.nan), np.full(n, np.nan)

    highs_s = pd.Series(highs)
    lows_s = pd.Series(lows)
    win = 2 * lookback + 1
    roll_max = highs_s.rolling(win, center=True, min_periods=win).max()
    roll_min = lows_s.rolling(win, center=True, min_periods=win).min()
    is_sh = (highs_s == roll_max).fillna(False).values
    is_sl = (lows_s == roll_min).fillna(False).values

    sh_level = np.full(n, np.nan)
    sh_strength = np.zeros(n)
    for idx in np.where(is_sh)[0]:
        confirm_idx = idx + lookback
        if confirm_idx >= n:
            continue
        level = highs[idx]
        band = level * tolerance
        w0, w1 = max(0, idx - strength_window), confirm_idx + 1
        touches = int((np.abs(highs[w0:w1] - level) <= band).sum() +
                      (np.abs(lows[w0:w1] - level) <= band).sum())
        sh_level[confirm_idx] = level
        sh_strength[confirm_idx] = touches

    sl_level = np.full(n, np.nan)
    sl_strength = np.zeros(n)
    for idx in np.where(is_sl)[0]:
        confirm_idx = idx + lookback
        if confirm_idx >= n:
            continue
        level = lows[idx]
        band = level * tolerance
        w0, w1 = max(0, idx - strength_window), confirm_idx + 1
        touches = int((np.abs(highs[w0:w1] - level) <= band).sum() +
                      (np.abs(lows[w0:w1] - level) <= band).sum())
        sl_level[confirm_idx] = level
        sl_strength[confirm_idx] = touches

    sh_level_ff    = pd.Series(sh_level).ffill().values
    sh_strength_ff = pd.Series(np.where(np.isnan(sh_level), np.nan, sh_strength)).ffill().fillna(0).values
    sl_level_ff    = pd.Series(sl_level).ffill().values
    sl_strength_ff = pd.Series(np.where(np.isnan(sl_level), np.nan, sl_strength)).ffill().fillna(0).values

    dist_to_res = np.where(~np.isnan(sh_level_ff), (sh_level_ff - closes) / closes * 100, 999.0)
    dist_to_sup = np.where(~np.isnan(sl_level_ff), (closes - sl_level_ff) / closes * 100, 999.0)
    closer_to_res = dist_to_res < dist_to_sup
    sr_dist_pct = np.where(closer_to_res, -dist_to_res, dist_to_sup)
    sr_node_strength = np.where(closer_to_res, sh_strength_ff, sl_strength_ff)
    return sr_dist_pct, sr_node_strength, sh_level_ff, sl_level_ff


def _vec_fib_dist(closes: np.ndarray, sh_level_ff: np.ndarray, sl_level_ff: np.ndarray) -> np.ndarray:
    n = len(closes)
    hi = np.maximum(sh_level_ff, sl_level_ff)
    lo = np.minimum(sh_level_ff, sl_level_ff)
    rng = hi - lo
    valid = (~np.isnan(rng)) & (rng > 0)
    dist = np.full(n, 999.0)
    for f in (0.236, 0.382, 0.5, 0.618, 0.786):
        level = hi - rng * f
        d = np.abs(closes - level) / np.where(closes > 0, closes, 1.0) * 100
        dist = np.where(valid, np.minimum(dist, d), dist)
    return np.where(valid, dist, 0.0)


_MACRO_EXT_BT_CACHE: Optional[Dict] = None
_MACRO_EXT_TICKERS = ['gold', 'oil_wti', 'sp500', 'vix', 'us10y_yield', 'dxy', 'btc_dominance']


def _load_macro_extended_bt() -> Dict:
    """Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance - zmiana % dzien-do-dnia,
    cache'owane raz na caly backtest (dane globalne, nie per-symbol - audyt
    2026-07-05, wpiete tez tu zeby CAT (jedyny model ktory tego uzywa) mial
    te same cechy w backteście co w treningu/live)."""
    global _MACRO_EXT_BT_CACHE
    if _MACRO_EXT_BT_CACHE is not None:
        return _MACRO_EXT_BT_CACHE
    out = {}
    macro_dir = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / "macro"
    for name in _MACRO_EXT_TICKERS:
        try:
            df = pd.read_parquet(macro_dir / f"{name}.parquet")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
            df = df.sort_values("timestamp").reset_index(drop=True)
            val_col = "close" if "close" in df.columns else "value"
            vals = df[val_col].values.astype(np.float64)
            # btc_dominance.value juz JEST gotowa 7d% zmiana BTC mcap (nie
            # poziomem) - patrz core/features.py dla pelnego wyjasnienia
            if name == "btc_dominance":
                chg = vals.copy()
            else:
                chg = np.zeros(len(vals))
                chg[1:] = np.where(vals[:-1] != 0, (vals[1:] / vals[:-1] - 1) * 100, 0.0)
            out[f"{name}_times"] = (df["timestamp"].values.astype("datetime64[ms]").astype(np.int64))
            out[f"{name}_chg"] = chg
        except Exception as e:
            logger.debug(f"macro_extended_bt {name}: {e}")
            out[f"{name}_times"] = np.array([], dtype=np.int64)
            out[f"{name}_chg"] = np.array([], dtype=np.float64)
    _MACRO_EXT_BT_CACHE = out
    return out


def _map_tf_to_1h(series: np.ndarray, times_tf: np.ndarray,
                  times_1h: np.ndarray, default: float = 50.0) -> np.ndarray:
    """Mapuje serię z wyższego timeframe na każdą świecę 1h (latest value ≤ ts)."""
    out = np.full(len(times_1h), default)
    if len(times_tf) == 0:
        return out
    for i, ts in enumerate(times_1h):
        idx = np.searchsorted(times_tf, ts, side="right") - 1
        if 0 <= idx < len(series) and not np.isnan(series[idx]):
            out[i] = series[idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL CACHE — dyskowy cache wyników N-BEATS/Transformer
# Klucz: symbol + last_timestamp_ms → nie rekomputujemy gdy dane niezmienione
# ─────────────────────────────────────────────────────────────────────────────

def _neural_cache_path(symbol: str, last_ts_ms: int) -> Path:
    NEURAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NEURAL_CACHE_DIR / f"{symbol}_{last_ts_ms}.npz"


def _neural_cache_load(symbol: str, last_ts_ms: int) -> Optional[Dict[str, np.ndarray]]:
    path = _neural_cache_path(symbol, last_ts_ms)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    except Exception:
        return None


def _neural_cache_save(symbol: str, last_ts_ms: int, nb: np.ndarray, tr: np.ndarray) -> None:
    # Usuń stare cache dla tego symbolu (inne timestampy)
    for old in NEURAL_CACHE_DIR.glob(f"{symbol}_*.npz"):
        if old != _neural_cache_path(symbol, last_ts_ms):
            try:
                old.unlink()
            except Exception:
                pass
    try:
        np.savez_compressed(_neural_cache_path(symbol, last_ts_ms),
                            nbeats=nb, transformer=tr)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# WAREHOUSE — buforowane odczyty (lru_cache żyje przez cały czas procesu)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _read_parquet_cached(path: str) -> Optional[pd.DataFrame]:
    """Czyta parquet raz i trzyma w pamięci przez cały backtest."""
    try:
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.debug(f"parquet read error {path}: {e}")
        return None


def _df_to_candles(df: pd.DataFrame, cutoff_ts: Optional[int] = None) -> List[Dict]:
    if df is None or df.empty:
        return []
    if cutoff_ts is not None:
        cutoff = pd.Timestamp(cutoff_ts, unit="ms", tz="UTC")
        df = df[df["timestamp"] <= cutoff]
    result = []
    for row in df.itertuples(index=False):
        ts = row.timestamp
        ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts) // 1_000_000
        result.append({
            "timestamp": ts_ms,
            "open":   float(row.open),
            "high":   float(row.high),
            "low":    float(row.low),
            "close":  float(row.close),
            "volume": float(row.volume),
        })
    return result


# Cechy zapisywane w kazdej transakcji ("kto/czemu/za ile" - audyt 2026-07-04),
# uniwersalne top cechy z analizy waznosci calej floty instancji.
_TOP_FEATURES_SNAPSHOT = [
    "oi_change_24h", "rsi_1d", "rsi_4h", "atr_pct",
    "ema_mid_r", "sr_node_strength", "trend_1d", "adx_14",
    "ls_ratio", "ls_ratio_chg_24h",
]


# === OBRONA POZYCJI — zamek zysku (2026-08-08) ==============================
# HAI_SL_LOCK=1 wlacza; progi w % ZWROTU NA POZYCJI (nie ruchu ceny).
# Domyslnie 25 -> 10: po osiagnieciu +25% SL ladzie na +10%.
_SL_LOCK         = os.environ.get("HAI_SL_LOCK") == "1"
_SL_LOCK_TRIGGER = float(os.environ.get("HAI_SL_LOCK_TRIGGER", "25"))
_SL_LOCK_AT      = float(os.environ.get("HAI_SL_LOCK_AT", "10"))


def _trend_ema_like(closes: np.ndarray) -> np.ndarray:
    """Trend jak w ml_trainer.precompute_trend_series: EMA(9) vs EMA(21),
    prog +-0.3%. JEDNA definicja dla treningu i backtestu (2026-08-08).

    Wolamy funkcje z ml_trainer, a nie wlasna kopie — kopia rozjechalaby sie
    przy pierwszej zmianie progu, czyli powtorzylaby blad, ktory ten fix usuwa.
    Fallback tylko na wypadek braku importu."""
    try:
        from .ml_trainer import precompute_trend_series
        return precompute_trend_series(closes, 9, 21).astype(np.float64)
    except Exception:
        if len(closes) < 21:
            return np.zeros(len(closes))
        _s = pd.Series(closes)
        ef = _s.ewm(span=9, adjust=False).mean().values
        es = _s.ewm(span=21, adjust=False).mean().values
        d = np.where(es != 0, (ef - es) / es * 100, 0.0)
        return np.where(d > 0.3, 1.0, np.where(d < -0.3, -1.0, 0.0))


class Backtester:
    """v9.1 — pełna doktryna + asyncio parallelizm."""

    def __init__(self):
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

        self.capital            = 1000.0
        self.base_leverage      = 5.0
        self.order_size         = 20.0
        self.fee_taker          = 0.0006
        self.slippage_entry     = 0.0005
        self.slippage_sl        = 0.0005
        self.funding_daily_rate = 0.0001
        self.stats              = {"status": "idle"}

    # ─────────────────────────────────────────────────────────────────────
    # WAREHOUSE LOADER
    # ─────────────────────────────────────────────────────────────────────

    def _sym_to_filename(self, symbol: str) -> str:
        """BTC / BTC/USDT:USDT / BTC/USDT → 'BTC'."""
        s = symbol.split("/")[0].split(":")[0]
        return s.replace("_", "")

    def load_candles_from_warehouse(self, symbol: str, tf: str,
                                    days: int = 90) -> List[Dict]:
        stem = self._sym_to_filename(symbol)
        path = WH_BASE / tf / f"{stem}.parquet"
        if not path.exists():
            return []
        df = _read_parquet_cached(str(path))
        if df is None:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days + 30)
        df_cut = df[df["timestamp"] >= cutoff]
        return _df_to_candles(df_cut)

    def _load_window(self, symbol: str, tf: str,
                     offset_start: int, offset_end: int) -> List[Dict]:
        stem = self._sym_to_filename(symbol)
        path = WH_BASE / tf / f"{stem}.parquet"
        if not path.exists():
            return []
        df = _read_parquet_cached(str(path))
        if df is None:
            return []
        now   = datetime.now(timezone.utc)
        start = now - timedelta(days=offset_start)
        end   = now - timedelta(days=offset_end)
        df_w  = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        return _df_to_candles(df_w)

    def _list_symbols(self) -> List[str]:
        syms = sorted(f.stem for f in (WH_BASE / "1h").glob("*.parquet"))
        # Opcjonalny whitelist przez HAI_SYMBOLS (CSV) — do lekkiego WFV/BT A/B
        # bez liczenia calej hurtowni. Puste = wszystkie (stare zachowanie).
        _wl = os.getenv("HAI_SYMBOLS", "").strip()
        if _wl:
            wl = {s.strip().upper() for s in _wl.split(",") if s.strip()}
            syms = [s for s in syms if s.upper() in wl]
        return syms

    # ─────────────────────────────────────────────────────────────────────
    # CORE SIMULATION — v9.2 vectorized (sync, uruchamiana w ThreadPoolExecutor)
    # ─────────────────────────────────────────────────────────────────────

    def _load_deriv_arrays(self, symbol: str, _tdf, n: int) -> Dict[str, np.ndarray]:
        """Szeregi derywatow (taker/funding/OI/ls_ratio) z magazynu, wyrownane
        do siatki 1h backtestu.

        WYDZIELONE 2026-08-08 z run_simulation_ai — z tego samego powodu co
        _build_feat_src: test parytetu (tools/test_parytet_cech.py) musi wolac
        TE SAMA sciezke, ktorej uzywa symulacja. Wczesniej test budowal wlasny
        slownik `neural` z zerami i zglaszal to jako blad backtestera, choc
        backtester liczyl poprawnie — falszywy alarm na 6 cechach.
        """
        def _ts_col_to_ms(col):
            """Kolumna timestamp z warehouse -> int64 ms. FIX 2026-07-18 (F1
            gen.Dir-v1): parquety maja datetime64[ms], a stary kod robil
            .astype('int64')//1_000_000 zakladajac ns -> wartosci ~1.78e6
            zamiast ~1.78e12 -> ZERO trafien w merge z ts_ms swiec -> po
            fillna(0) WSZYSTKIE cechy derywatow (funding_*, oi_*,
            taker_buy_ratio) byly ZERAMI w kazdym backtescie i snapshocie,
            mimo ze trening (ml_trainer, pd.to_datetime+searchsorted) widzial
            je poprawnie - czyli modele dostawaly na inferencji rozklad inny
            niz na treningu."""
            s = pd.to_datetime(col)
            return (s.astype("datetime64[ms]").astype("int64")).astype("int64")

        out: Dict[str, np.ndarray] = {}
        _sym_stem = self._sym_to_filename(symbol)
        _wh_deriv = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / "derivatives"
        try:
            _tr_path = _wh_deriv / "taker_ratio" / f"{_sym_stem}.parquet"
            if _tr_path.exists():
                _tr_df = pd.read_parquet(_tr_path)
                _tr_df["ts_ms"] = _ts_col_to_ms(_tr_df["timestamp"])
                out["taker_buy_ratio"] = (
                    _tdf.merge(_tr_df[["ts_ms", "taker_buy_ratio"]], on="ts_ms", how="left")
                    ["taker_buy_ratio"].fillna(0.5).values
                )
        except Exception:
            pass
        try:
            _fr_path = _wh_deriv / "funding_rates" / f"{_sym_stem}.parquet"
            if _fr_path.exists():
                _fr_df = pd.read_parquet(_fr_path).dropna()
                _fc = "funding_rate" if "funding_rate" in _fr_df.columns else "close"
                _fr_df["ts_ms"] = _ts_col_to_ms(_fr_df["timestamp"])
                _fr_arr = (
                    _tdf.merge(_fr_df[["ts_ms", _fc]].assign(_v=_fr_df[_fc] * 100)[["ts_ms", "_v"]],
                               on="ts_ms", how="left")["_v"].values.astype(float)
                )
                _fr_arr = pd.Series(_fr_arr).ffill().fillna(0.0).values
                out["funding_rate"] = _fr_arr
                # funding_change_24h — 24 świece 1h wstecz (spojne z ml_trainer.py)
                _fr_prev = np.roll(_fr_arr, 24)
                _fr_prev[:24] = _fr_arr[:24]
                out["funding_change_24h"] = _fr_arr - _fr_prev
        except Exception:
            pass
        try:
            _oi_path = _wh_deriv / "open_interest" / f"{_sym_stem}.parquet"
            if _oi_path.exists():
                _oi_df = pd.read_parquet(_oi_path).dropna()
                _oi_df["ts_ms"] = _ts_col_to_ms(_oi_df["timestamp"])
                _oi_arr = (
                    _tdf.merge(_oi_df[["ts_ms", "close"]], on="ts_ms", how="left")
                    ["close"].values.astype(float)
                )
                _oi_arr = pd.Series(_oi_arr).ffill().fillna(0.0).values
                _oi_prev = np.roll(_oi_arr, 24); _oi_prev[:24] = _oi_arr[:24]
                out["oi_total_log"]  = np.log1p(_oi_arr)
                out["oi_change_24h"] = np.where(_oi_prev > 0, (_oi_arr - _oi_prev) / _oi_prev, 0.0)
                _roll30 = pd.Series(_oi_arr).rolling(30*24, min_periods=24)
                out["oi_zscore_30d"] = ((_oi_arr - _roll30.mean().values) /
                                        _roll30.std().values.clip(1e-8)).clip(-5, 5)
        except Exception:
            pass
        try:
            # LS RATIO (B2 gen.Dir-v1, 2026-07-19) - z magazynu, 1h; neutralne
            # 1.0 gdy brak danych/symbolu (spojne z ml_trainer).
            _ls_path = _wh_deriv / "ls_ratio" / f"{_sym_stem}.parquet"
            out["ls_ratio"] = np.full(n, 1.0)
            out["ls_ratio_chg_24h"] = np.zeros(n)
            if _ls_path.exists():
                _ls_df = pd.read_parquet(_ls_path).dropna()
                _ls_df["ts_ms"] = _ts_col_to_ms(_ls_df["timestamp"])
                _ls_arr = (
                    _tdf.merge(_ls_df[["ts_ms", "ls_ratio"]], on="ts_ms", how="left")
                    ["ls_ratio"].values.astype(float)
                )
                _ls_arr = pd.Series(_ls_arr).ffill().fillna(1.0).values
                out["ls_ratio"] = _ls_arr
                _ls_prev = np.roll(_ls_arr, 24); _ls_prev[:24] = _ls_arr[:24]
                out["ls_ratio_chg_24h"] = _ls_arr - _ls_prev
        except Exception:
            pass
        return out

    def _build_feat_src(self, ind: Dict, neural: Dict, candles_1h: List[Dict],
                        symbol: str, n: int) -> Dict[str, np.ndarray]:
        """Buduje slownik cecha -> tablica, ktory karmi predykcje w backtescie.

        WYDZIELONE 2026-08-08 z run_simulation_ai. Powod: test parytetu trzech
        sciezek (tools/test_parytet_cech.py) musi sprawdzac TE SAMA sciezke,
        ktorej uzywa symulacja. Gdyby test mial wlasna kopie tej logiki,
        rozjechalby sie z produkcja przy pierwszej zmianie — czyli dokladnie ten
        blad, ktory ten test ma wykrywac.
        """
        feat_src = {**{k: v for k, v in ind.items() if not k.startswith("_")},
                    **neural}

        # Interakcja funding x OI-zscore (B3 gen.Dir-v1, 2026-07-19) - spojne
        # z ml_trainer (record dict). Oba czynniki juz w feat_src (neural).
        _fr_v = feat_src.get("funding_rate")
        _oz_v = feat_src.get("oi_zscore_30d")
        if _fr_v is not None and _oz_v is not None:
            feat_src["funding_x_oizscore"] = np.asarray(_fr_v) * np.asarray(_oz_v)

        # Cechy mapy likwidacji (dist_below_liq/dist_above_liq) - WSPOLNA funkcja z
        # ml_trainer (parzystosc!). Bez tego feat_src.get(f, zeros) wstawilby zera
        # dla modeli wytrenowanych z ta cecha -> scaler OOD -> smieciowy routing
        # (dokladnie bug derywatow). Ledger inkrementalny z okna; brak -> DIST_DEFAULT.
        try:
            from .liqmap_features import compute_liq_dist_from_warehouse
            _lo = pd.DataFrame({
                "timestamp": pd.to_datetime([c["timestamp"] for c in candles_1h], unit="ms"),
                "close": np.array([c["close"] for c in candles_1h], dtype=np.float64),
                "high":  np.array([c["high"]  for c in candles_1h], dtype=np.float64),
                "low":   np.array([c["low"]   for c in candles_1h], dtype=np.float64),
            })
            _db, _da = compute_liq_dist_from_warehouse(_lo, symbol)
            feat_src["dist_below_liq"] = _db
            feat_src["dist_above_liq"] = _da
        except Exception as _liq_e:
            logger.debug(f"liq features skip: {_liq_e}")
            feat_src["dist_below_liq"] = np.full(n, 30.0)
            feat_src["dist_above_liq"] = np.full(n, 30.0)

        # === CECHY x_*/r_*/e_* — parytet trening <-> backtest (2026-08-08) ===
        # DO DZIS backtester ICH NIE LICZYL. Model trenowal sie z x_real_vol_6h czy
        # r_stoch_rsi (dataset je ma, bramka feature_mix przepuszczala), a w symulacji
        # dostawal w tym miejscu ZERA — bo predykcja robi feat_src.get(f, np.zeros(n)).
        # Skaler zamienial te zera w skrajne OOD. Skutek: KAZDY wynik WFV dla configu
        # z warstwa rptr/x_ mierzyl model z polowa wejscia zastapiona zerami, a nie
        # wartosc tych cech. Tlumaczy to, czemu rptr nigdy nie pokazal zysku i czemu
        # ET w kampanii v8 dal ZERO transakcji (jego zestaw byl w wiekszosci nowy).
        #
        # Wolamy TE SAME funkcje co ml_trainer — nie wlasne kopie formul.
        try:
            from . import ml_trainer as _mt
            _c = np.array([c["close"]  for c in candles_1h], dtype=np.float64)
            _h = np.array([c["high"]   for c in candles_1h], dtype=np.float64)
            _l = np.array([c["low"]    for c in candles_1h], dtype=np.float64)
            _v = np.array([c["volume"] for c in candles_1h], dtype=np.float64)
            _extra = {}
            if hasattr(_mt, "calc_x_features"):
                _extra.update(_mt.calc_x_features(_c, _h, _l, _v))
            if hasattr(_mt, "calc_rptr_features"):
                _rsi = np.asarray(ind.get("rsi", np.zeros(n)), dtype=np.float64)
                _atr = np.asarray(ind.get("_atr_abs", np.zeros(n)), dtype=np.float64)
                _extra.update(_mt.calc_rptr_features(_c, _h, _l, _v, _rsi, _atr))
            _kolizje = [k for k in _extra if k in feat_src]
            if _kolizje:
                logger.debug(f"rptr/x_: pomijam kolizje z istniejacymi: {_kolizje}")
            for _k, _val in _extra.items():
                if _k not in feat_src and len(np.asarray(_val)) == n:
                    feat_src[_k] = np.asarray(_val, dtype=np.float64)
        except Exception as _x_e:
            logger.warning(f"cechy x_/rptr nieobliczone ({_x_e}) — modele z tymi "
                           f"cechami dostana ZERA (wynik niewiarygodny)")

        # === ALIASY I INTERAKCJE — parytet z ml_trainer (2026-08-08) ===
        # Wykryte przez tools/test_parytet_cech.py: trening je liczyl, backtest nie,
        # wiec modele dostawaly tu zera. Wzory 1:1 z ml_trainer (numery linii to
        # stan na 2026-08-08): r_htf_trend_4h=trend_4h (1361),
        # e_phantomflow_score=sr_node_strength (1372), e_rsi_x_funding (1373),
        # e_atr_pct_x_vol_zscore (1374), x_weekend (1376).
        def _we(nazwa):
            v = feat_src.get(nazwa)
            return np.asarray(v, dtype=np.float64) if v is not None else None
        _aliasy = {
            "r_htf_trend_4h":         _we("trend_4h"),
            "e_phantomflow_score":    _we("sr_node_strength"),
        }
        _rsi_a, _fund_a = _we("rsi"), _we("funding_rate")
        if _rsi_a is not None and _fund_a is not None:
            _aliasy["e_rsi_x_funding"] = _rsi_a * _fund_a
        _atr_a, _vz_a = _we("atr_pct"), _we("volume_zscore")
        if _atr_a is not None and _vz_a is not None:
            _aliasy["e_atr_pct_x_vol_zscore"] = _atr_a * _vz_a
        _dow_a = _we("day_of_week")
        if _dow_a is not None:
            _aliasy["x_weekend"] = np.where((_dow_a == 5.0) | (_dow_a == 6.0), 1.0, 0.0)
        # x_btc_leadlag = btc_dominance - momentum (ml_trainer:1391). Backtester
        # mapuje ten sam szereg pod nazwa btc_dominance_chg (macro_feats), bo
        # btc_dominance.value w magazynie JEST juz gotowa zmiana, nie poziomem.
        _btc_a, _mom_a = _we("btc_dominance_chg"), _we("momentum")
        if _btc_a is not None and _mom_a is not None:
            _aliasy["x_btc_leadlag"] = _btc_a - _mom_a
        for _k, _v in _aliasy.items():
            if _v is not None and _k not in feat_src and len(_v) == n:
                feat_src[_k] = _v

        # Cechy liczone w ml_trainer osobnymi funkcjami (divergence / S&D /
        # wiek przeciecia EMA). Wolamy TE SAME funkcje, nie kopie formul.
        try:
            from . import ml_trainer as _mt2
            _c2 = np.array([c["close"] for c in candles_1h], dtype=np.float64)
            _h2 = np.array([c["high"]  for c in candles_1h], dtype=np.float64)
            _l2 = np.array([c["low"]   for c in candles_1h], dtype=np.float64)
            _rsi2 = np.asarray(ind.get("rsi", np.zeros(n)), dtype=np.float64)
            _atr2 = np.asarray(ind.get("_atr_abs", np.zeros(n)), dtype=np.float64)
            _poz = {}
            if hasattr(_mt2, "calc_divergence"):
                _poz["div_rsi"] = _mt2.calc_divergence(_c2, _rsi2, 20)
            if hasattr(_mt2, "calc_sd_proximity"):
                _poz["sd_prox"] = _mt2.calc_sd_proximity(_h2, _l2, _c2, _atr2, 50)
            if hasattr(_mt2, "calc_bars_since_cross"):
                _poz["bars_cross"] = _mt2.calc_bars_since_cross(_c2, 20, 50, 100)
            for _k, _v in _poz.items():
                _a = np.asarray(_v, dtype=np.float64)
                if _k not in feat_src and _a.ndim == 1 and len(_a) == n:
                    feat_src[_k] = _a
        except Exception as _p_e:
            logger.warning(f"cechy div/sd/bars nieobliczone ({_p_e}) — modele z nimi "
                           f"dostana ZERA")

        return feat_src

    def _precompute_indicators(
        self,
        candles_1h: List[Dict],
        candles_4h: List[Dict],
        candles_1d: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """Wektoryzuje WSZYSTKIE wskaźniki techniczne na całej serii RAZ.
        W pętli walk-forward → O(1) lookup po indeksie zamiast O(n) per candle.
        """
        c1h = np.array([c["close"]  for c in candles_1h], dtype=np.float64)
        v1h = np.array([c["volume"] for c in candles_1h], dtype=np.float64)
        t1h = np.array([c["timestamp"] for c in candles_1h], dtype=np.int64)

        # ── 1H wskaźniki ──────────────────────────────────────────────
        rsi_1h   = _vec_rsi(c1h, 14)
        ema_9    = _vec_ema(c1h, 9)
        ema_21   = _vec_ema(c1h, 21)
        ema_50   = _vec_ema(c1h, 50)
        atr_14   = _vec_atr(candles_1h, 14)
        roc_10   = _vec_roc(c1h, 10)
        bb_pos, bb_bw_pct = _vec_bb(c1h, 20)
        adx_14   = _vec_adx(c1h, 14)
        stoch_k, stoch_d = _vec_stoch(c1h, 14, 3)

        # Volume ratio (30-period rolling mean) — vectorized via pandas
        _v1h_s  = pd.Series(v1h)
        vol_sum = _v1h_s.rolling(30, min_periods=1).mean().values
        vol_ratio = np.where(vol_sum > 0, v1h / vol_sum, 1.0)

        # EMA ratios
        ema_slow_r = np.where(ema_21 > 0, (c1h / ema_21 - 1) * 100, 0.0)
        ema_mid_r  = np.where(ema_50 > 0, (c1h / ema_50 - 1) * 100, 0.0)
        ema_fast_r = np.where(ema_9 > 0, (c1h / ema_9 - 1) * 100, 0.0)
        atr_pct    = np.where(c1h > 0, atr_14 / c1h * 100, 0.0)

        # trend_1h — ta sama definicja co ml_trainer.precompute_trend_series (EMA9 vs EMA21)
        _diff_pct_1h = np.where(ema_21 != 0, (ema_9 - ema_21) / ema_21 * 100, 0.0)
        trend_1h = np.where(_diff_pct_1h > 0.3, 1.0, np.where(_diff_pct_1h < -0.3, -1.0, 0.0))

        # Nowe cechy (audyt sesji 2026-07-03)
        volume_zscore_arr = _vec_volume_zscore(v1h, 30)
        macd_hist_arr = _vec_macd_hist(c1h)
        sr_dist_arr, sr_strength_arr, sh_level_ff, sl_level_ff = _vec_swing_sr(c1h, np.array([c["high"] for c in candles_1h], dtype=np.float64), np.array([c["low"] for c in candles_1h], dtype=np.float64))
        fib_dist_arr = _vec_fib_dist(c1h, sh_level_ff, sl_level_ff)

        # ── Time features — vectorized via pandas DatetimeIndex ───────
        _t_dt    = pd.to_datetime(t1h, unit='ms', utc=True)
        hours    = _t_dt.hour.values.astype(np.int32)
        dows     = _t_dt.dayofweek.values.astype(np.float64)
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)

        # ── Session — vectorized ──────────────────────────────────────
        # 0=normal, 1=prime, 2=dead
        _DEAD  = np.array([22, 23, 0])
        _PRIME = np.array([7, 8, 13, 14, 15])
        sessions = np.where(np.isin(hours, _DEAD), np.int8(2),
                            np.where(np.isin(hours, _PRIME), np.int8(1), np.int8(0))
                   ).astype(np.int8)

        # ── Neural extra features (22/24-feat models) ─────────────────
        ema_200           = _vec_ema(c1h, 200)
        ema_200_dist_pct  = np.where(ema_200 > 0, (c1h / ema_200 - 1) * 100, 0.0)
        rsi_slope_5h_arr  = np.zeros(len(rsi_1h))
        rsi_slope_5h_arr[5:] = rsi_1h[5:] - rsi_1h[:-5]
        _vol5         = _v1h_s.rolling(5, min_periods=1).mean().values
        vol_trend_arr = np.where(vol_sum > 0, _vol5 / vol_sum, 1.0)
        # vwap_dev — 24h rolling VWAP deviation (v8.4)
        _pv24 = pd.Series(c1h * v1h).rolling(24, min_periods=1).sum().values
        _v24  = pd.Series(v1h).rolling(24, min_periods=1).sum().values
        _vwap24 = np.where(_v24 > 0, _pv24 / _v24, c1h)
        vwap_dev_arr = np.where(_vwap24 > 0, (c1h / _vwap24 - 1) * 100, 0.0)

        # ── 4H features (mapowane na 1h) ──────────────────────────────
        if candles_4h:
            c4h = np.array([c["close"] for c in candles_4h], dtype=np.float64)
            t4h = np.array([c["timestamp"] for c in candles_4h], dtype=np.int64)
            rsi_4h_s = _vec_rsi(c4h, 14)
            # FIX 2026-08-08 (test_parytet_cech): bylo ROC(10) z progiem +-0.1,
            # a trening liczy trend jako EMA(9) vs EMA(21) z progiem +-0.3%
            # (ml_trainer.precompute_trend_series). Dwie ROZNE definicje tej samej
            # cechy: model uczyl sie jednej, w symulacji dostawal druga. Uwaga —
            # rozjazd bywa niewidoczny w pojedynczej swiecy (oba wzory czesto daja
            # ten sam znak), wiec nie da sie go zlapac punktowo; widac go dopiero
            # przy porownaniu FORMUL. trend_1h byl juz liczony poprawnie (EMA/0.3).
            trend_4h_s = _trend_ema_like(c4h)
            rsi_4h   = _map_tf_to_1h(rsi_4h_s, t4h, t1h, 50.0)
            trend_4h = _map_tf_to_1h(trend_4h_s, t4h, t1h, 0.0)
        else:
            rsi_4h   = np.full(len(c1h), 50.0)
            trend_4h = np.zeros(len(c1h))

        # ── 1D features (mapowane na 1h) ──────────────────────────────
        if candles_1d:
            c1d = np.array([c["close"] for c in candles_1d], dtype=np.float64)
            t1d = np.array([c["timestamp"] for c in candles_1d], dtype=np.int64)
            rsi_1d_s = _vec_rsi(c1d, 14)
            trend_1d_s = _trend_ema_like(c1d)   # patrz komentarz przy trend_4h
            rsi_1d   = _map_tf_to_1h(rsi_1d_s, t1d, t1h, 50.0)
            trend_1d = _map_tf_to_1h(trend_1d_s, t1d, t1h, 0.0)
        else:
            rsi_1d   = np.full(len(c1h), 50.0)
            trend_1d = np.zeros(len(c1h))

        # ── Macro extended: Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance
        # (dzienne, mapowane na 1h jak rsi_4h/trend_4h - audyt 2026-07-05) ──
        # nazwa cechy finalnej != nazwa pliku dla us10y_yield -> us10y_chg,
        # spojne z core/ml_trainer.py i core/features.py (MODEL_FEATURES['cat']).
        _macro_feat_name = {"us10y_yield": "us10y_chg"}
        _macro_ext = _load_macro_extended_bt()
        macro_feats = {}
        for _name in _MACRO_EXT_TICKERS:
            _mt = _macro_ext.get(f"{_name}_times", np.array([]))
            _mc = _macro_ext.get(f"{_name}_chg", np.array([]))
            _fname = _macro_feat_name.get(_name, f"{_name}_chg")
            macro_feats[_fname] = _map_tf_to_1h(_mc, _mt, t1h, 0.0)

        return {
            **macro_feats,
            # ATR BEZWZGLEDNY (nie %) — potrzebny dla calc_rptr_features, ktore
            # w ml_trainer dostaje calc_atr(highs, lows, closes), a nie atr_pct.
            # Prefiks "_" wyklucza go z feat_src (patrz filtr not k.startswith("_")),
            # wiec nie trafia do modeli jako cecha — sluzy tylko do liczenia rptr.
            "_atr_abs":           atr_14,
            "rsi":                rsi_1h,
            "rsi_4h":             rsi_4h,
            "rsi_1d":             rsi_1d,
            "ema_slow_r":         ema_slow_r,
            "ema_mid_r":          ema_mid_r,
            "atr_pct":            atr_pct,
            "momentum":           roc_10,
            "trend_4h":           trend_4h,
            "trend_1d":           trend_1d,
            "volume_ratio":       vol_ratio,
            "funding_rate":       np.zeros(len(c1h)),
            "price_position_bb":  bb_pos,
            "bb_bandwidth_pct":   bb_bw_pct,
            "oi_total_log":       np.zeros(len(c1h)),
            "oi_change_24h":      np.zeros(len(c1h)),
            "oi_zscore_30d":      np.zeros(len(c1h)),
            "hour_sin":           hour_sin,
            "hour_cos":           hour_cos,
            "day_of_week":        dows,
            "adx_14":             adx_14,
            "ema_fast_r":         ema_fast_r,
            "trend_1h":           trend_1h,
            # Nowe cechy (audyt sesji 2026-07-03)
            "volume_zscore":      volume_zscore_arr,
            "macd_hist":          macd_hist_arr,
            "sr_dist_pct":        sr_dist_arr,
            "sr_node_strength":   sr_strength_arr,
            "fib_dist_pct":       fib_dist_arr,
            "funding_change_24h": np.zeros(len(c1h)),
            "taker_buy_ratio":    np.full(len(c1h), 0.5),
            # Neural extra (22-feat models)
            "ema_200_dist_pct":   ema_200_dist_pct,
            "rsi_slope_5h":       rsi_slope_5h_arr,
            "vol_trend":          vol_trend_arr,
            # v8.4 neural extra
            "vwap_dev":           vwap_dev_arr,
            # Doctrine arrays (nie są features, używane do filtrów)
            "_atr":               atr_14,
            "_bb_pos":            bb_pos,
            "_bb_bw_pct":         bb_bw_pct,
            "_adx":               adx_14,
            "_stoch_k":           stoch_k,
            "_stoch_d":           stoch_d,
            "_sessions":          sessions,
            "_hours":             hours.astype(np.float64),
        }

    def run_simulation_ai(
        self,
        candles_1h: List[Dict],
        candles_4h: List[Dict],
        candles_1d: List[Dict],
        symbol: str,
        mode: str = "neutral",
        atr_tp: float = 3.5,
        atr_sl: float = 1.5,
        enable_pyramid: bool = False,
        enable_cooldown: bool = True,
        enable_daily_limit: bool = True,
        daily_loss_limit: float = 50.0,
    ) -> List[Dict]:
        """Walk-forward symulacja v9.2 — vectorized indicators + O(1) lookup.

        Zamiast score_symbol() per candle (O(n²)), wszystkie wskaźniki
        liczone raz vectorized, w pętli tylko lookup + ensemble.predict().
        Wywoływana w ThreadPoolExecutor.
        """
        try:
            import torch as _t; _t.set_num_threads(1)  # 4 wątki × 1 thread = brak contention
        except Exception:
            pass
        from .ensemble import ensemble as _ens

        strategy = get_strategy("ai_strategy")
        strategy.set_mode(mode)
        min_h = strategy.min_history
        min_conf = strategy.min_confidence

        # audyt 2026-07-05: wartosci dopasowane do strategies/ai_strategy.py
        # (live) - komentarze TAM dokumentuja ze 0.20/0.08 i regime=0 hard-blok
        # byly swiadomie poluzowane wczesniej ("zbyt restrykcyjne", "3-class
        # modele radza bez hard-bloku") - backtester byl PRZESTARZALY wzgledem
        # live, nie live wzgledem backtestera. Teraz oba uzywaja tych samych
        # progow, wiec wynik bt faktycznie reprezentuje zachowanie live.
        BB_LONG_MAX   = 0.30
        BB_SHORT_MIN  = 0.70
        BB_WIDTH_MAX  = 0.12
        ADX_MIN       = 22
        REGIME_ADJUST = {0: 0.02, 1: 0.00, 2: 0.03}

        n = len(candles_1h)
        if n < min_h + 10:
            return []

        closes_1h = np.array([c["close"]     for c in candles_1h], dtype=np.float64)
        times_1h  = np.array([c["timestamp"] for c in candles_1h], dtype=np.int64)
        hi_arr    = np.array([c["high"]      for c in candles_1h], dtype=np.float64)
        lo_arr    = np.array([c["low"]       for c in candles_1h], dtype=np.float64)
        date_arr  = pd.to_datetime(times_1h, unit='ms', utc=True).strftime('%Y-%m-%d').values
        _tdf      = pd.DataFrame({"ts_ms": times_1h})

        # ── PRE-COMPUTE: wszystkie wskaźniki raz ──────────────────────
        ind = self._precompute_indicators(candles_1h, candles_4h, candles_1d)

        # N-BEATS / Transformer (LAB only) — batch inference z dyskiem cache
        # Cache key = symbol + last 1h timestamp → reuse gdy dane niezmienione
        neural = {"nbeats_pred_return_4h": np.zeros(n),
                  "transformer_pred_return_4h": np.zeros(n),
                  "taker_buy_ratio": np.full(n, 0.5)}

        neural.update(self._load_deriv_arrays(symbol, _tdf, n))
        try:
            from .features import NBEATS_PATH, TRANSFORMER_PATH
            from .nbeats import predict_nbeats_series
            from .transformer import predict_transformer_series

            last_ts_ms = int(times_1h[-1]) if n > 0 else 0
            cached = _neural_cache_load(symbol, last_ts_ms)
            if cached is not None and len(cached["nbeats"]) == n:
                neural["nbeats_pred_return_4h"]      = cached["nbeats"]
                neural["transformer_pred_return_4h"] = cached["transformer"]
            else:
                import torch as _torch
                _torch.set_num_threads(2)
                nb = predict_nbeats_series(closes_1h.astype(np.float32), NBEATS_PATH)
                tr = predict_transformer_series(closes_1h.astype(np.float32), TRANSFORMER_PATH)
                neural["nbeats_pred_return_4h"]      = nb
                neural["transformer_pred_return_4h"] = tr
                _neural_cache_save(symbol, last_ts_ms, nb, tr)
        except (ImportError, Exception):
            pass

        # ── REGIME pre-compute: raz na dobę (nie per-candle) ──────────
        # detect_from_closes buduje features + HMM Viterbi → kosztowne per-candle.
        # Reżim zmienia się powoli → wystarczy raz na 24h świece.
        regime_arr = np.full(n, -1, dtype=np.int8)
        try:
            from .regime_detector import regime_detector as _rd
            if _rd.is_trained:
                vols = np.array([c["volume"] for c in candles_1h], dtype=np.float64)
                # Zrób wpisy co 24 świece, potem forward-fill
                step = 24
                last_reg = -1
                for i in range(min_h, n, step):
                    reg = int(_rd.detect_from_closes(
                        closes_1h[max(0, i-min_h):i].tolist(),
                        vols[max(0, i-min_h):i].tolist()
                    ) or -1)
                    regime_arr[i:min(i+step, n)] = reg
                    last_reg = reg
                # Wypełnij luki przed pierwszym krokiem
                if min_h < n:
                    regime_arr[:min_h] = regime_arr[min_h]
        except Exception:
            pass

        # ── ARRAYS DOKTRYNOWE ─────────────────────────────────────────
        bb_pos_arr  = ind["_bb_pos"]
        bb_bw_arr   = ind["_bb_bw_pct"]
        adx_arr     = ind["_adx"]
        stoch_k_arr = ind["_stoch_k"]
        stoch_d_arr = ind["_stoch_d"]
        sess_arr    = ind["_sessions"]
        atr_arr     = ind["_atr"]
        trend_1d_arr = ind["trend_1d"]  # audyt 2026-07-05, regime-adaptive

        # ── BATCH ENSEMBLE PREDICT (vectorized, 705× szybszy niż per-call) ──
        # 1. Maska doktrynowa — w pełni wektoryzowana (zero Python loops)
        _long_cand  = (bb_pos_arr <= BB_LONG_MAX)
        _short_cand = (bb_pos_arr >= BB_SHORT_MIN)
        # audyt 2026-07-05: regime=0 (trend_following) juz NIE jest twardo
        # blokowany - dopasowane do ai_strategy.py (live), gdzie 3-class
        # modele same rozrozniaja LONG/SHORT/NEUTRAL bez hard-bloku, tylko
        # z podwyzszonym progiem (REGIME_ADJUST[0]=+0.02, ponizej).
        #
        # REGIME-ADAPTIVE (audyt 2026-07-05, "napisz nowa strategie trzymajaca
        # sie doktryn") - gdy globalny toggle _REGIME_ADAPTIVE_MODE wlaczony,
        # w regime=0 (trend_following) zamiast ekstremum BB (mean-reversion)
        # uzywamy strefy KONTYNUACJI zgodnej z trend_1d - patrz
        # strategies/ai_strategy.py._check_doctrine_zone dla tej samej logiki
        # w live. Domyslnie WYLACZONY - zero zmiany istniejacego zachowania.
        if _REGIME_ADAPTIVE_MODE:
            _regime0 = (regime_arr == 0)
            _cont_long = (_regime0 & (trend_1d_arr > 0) &
                          (bb_pos_arr >= CONTINUATION_ZONE_LONG[0]) & (bb_pos_arr <= CONTINUATION_ZONE_LONG[1]))
            _cont_short = (_regime0 & (trend_1d_arr < 0) &
                           (bb_pos_arr >= CONTINUATION_ZONE_SHORT[0]) & (bb_pos_arr <= CONTINUATION_ZONE_SHORT[1]))
            _long_cand  = np.where(_regime0, _cont_long, _long_cand)
            _short_cand = np.where(_regime0, _cont_short, _short_cand)
        _env        = ((bb_bw_arr <= BB_WIDTH_MAX) &
                       (adx_arr   >= ADX_MIN) &
                       (sess_arr  != 2))
        # Doktryna wolna: bez kierunkowego filtra BB (_long_cand|_short_cand),
        # zostaja tylko filtry srodowiska. Model sam decyduje kierunek.
        _base       = _env if _DOCTRINE_FREE else ((_long_cand | _short_cand) & _env)
        doc_mask              = np.zeros(n, dtype=bool)
        doc_mask[min_h:]      = _base[min_h:]
        doc_dir               = np.zeros(n, dtype=np.int8)
        doc_dir[_long_cand]   = np.int8(1)
        doc_dir[_short_cand]  = np.int8(-1)
        doc_dir[~doc_mask]    = np.int8(0)

        mask_idx = np.where(doc_mask)[0]

        # 2. Feature matrix for tree models (N_pass, 19) + raw feat_src for neural
        try:
            from .ensemble import FEATURE_NAMES as _FN
        except ImportError:
            _FN = [k for k in ind if not k.startswith("_")]

        feat_src = self._build_feat_src(ind, neural, candles_1h, symbol, n)
        # Cechy do feature_snapshot (2026-08-06): union tego co modele w ensemble
        # FAKTYCZNIE maja w feature_names + stala lista kontekstu rynkowego.
        # Bez tego snapshot ignorowal feature_mix i zapisywal cechy spoza modelu.
        try:
            _used = set()
            for _mn in getattr(_ens, "models", {}):
                _used.update(_ens.feature_names.get(_mn) or [])
            _SNAP_FEATS = sorted(_used | set(_TOP_FEATURES_SNAPSHOT)) if _used \
                          else list(_TOP_FEATURES_SNAPSHOT)
        except Exception:
            _SNAP_FEATS = list(_TOP_FEATURES_SNAPSHOT)

        X_full = np.stack([feat_src.get(f, np.zeros(n)) for f in _FN], axis=1)
        X_pass = X_full[mask_idx] if len(mask_idx) > 0 else np.zeros((0, len(_FN)))

        # 3. Batch predict — drzewa: 2D+scaler extern; TorchWrapper: własne featy+seq+scaler wewnętrzny
        batch_long  = np.zeros(n)
        batch_short = np.zeros(n)
        # Glos KAZDEGO modelu z osobna (audyt 2026-07-04 - user chcial widziec
        # "kto/czemu/za ile" przy otwarciu pozycji, nie tylko sume ensemble).
        # Puste (0.0) poza mask_idx - wypelniane tylko tam gdzie model realnie
        # policzyl predykcje.
        per_model_lp: Dict[str, np.ndarray] = {}
        per_model_sp: Dict[str, np.ndarray] = {}
        # Wazone (nie surowe) glosy per model - potrzebne dla voting_mode=
        # "winner_take_all" (audyt 2026-07-06), gdzie zamiast SUMY bierzemy MAX.
        per_model_wl: Dict[str, np.ndarray] = {}
        per_model_ws: Dict[str, np.ndarray] = {}

        if len(mask_idx) > 0:
            eff_w = _ens.weights
            # audyt 2026-07-05: regime-blended wagi wpiete tu tez - wczesniej
            # dotyczylo TYLKO live/paper (przez ensemble._regime_blended_weights,
            # wolane z ai_strategy.py), backtester uzywal plaskich wag zawsze -
            # bt teraz faktycznie testuje to co live robi. Mnoznik per-candle
            # (regime sie zmienia w czasie), nie per-symbol jak wagi bazowe.
            from .ensemble import REGIME_WEIGHTS as _RW, SIDE_WEIGHTS as _SW
            _regime_slice = regime_arr[mask_idx]
            for mname, model in _ens.models.items():
                w_base = eff_w.get(mname, 1.0 / max(len(_ens.models), 1))
                _mult = np.ones(len(mask_idx))
                for _reg in (0, 1, 2):
                    _m = _RW.get(_reg, {}).get(mname)
                    if _m is not None:
                        _mult[_regime_slice == _reg] = _m
                w_mdl = w_base * _mult
                # Opcja C (audyt 2026-07-06) - per-strona mnoznik, patrz
                # SIDE_WEIGHTS w ensemble.py. Rozdzielone TU (nie w w_mdl
                # wspolnym) bo LONG/SHORT maja rozne akumulatory ponizej.
                w_mdl_long  = w_mdl * _SW.get(mname, {}).get("long", 1.0)
                w_mdl_short = w_mdl * _SW.get(mname, {}).get("short", 1.0)
                sc    = _ens.scalers.get(mname)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if hasattr(model, 'model_type'):
                            # TorchWrapper — własna macierz feat + seq, skaluje wewnętrznie
                            mfeats = _ens.feature_names.get(mname) or list(model.feature_names)
                            X_m = np.stack([feat_src.get(f, np.zeros(n)) for f in mfeats], axis=1)
                            seq_l = getattr(model, 'seq_len', 0) or 0
                            if seq_l > 0:
                                # Vectorized window building — numpy fancy indexing (zero Python loop)
                                _nf   = len(mfeats)
                                _pad  = np.zeros((seq_l - 1, _nf), dtype=np.float32)
                                _Xp   = np.vstack([_pad, X_m.astype(np.float32)])
                                _aidx = mask_idx + (seq_l - 1)
                                _ridx = _aidx[:, None] - np.arange(seq_l - 1, -1, -1)[None, :]
                                X_input = _Xp[_ridx]  # (N_mask, seq_l, n_feat)
                            else:
                                X_input = X_m[mask_idx].astype(np.float32)
                            proba = model.predict_proba(X_input)
                        else:
                            # Tree model (RF/LGB/XGB) — external scaler, wektor cech per-model
                            # (bez tego wszystkie drzewa dostawały wspólny X_pass zbudowany
                            # z globalnego 15-cechowego FEATURE_NAMES, podczas gdy każdy model
                            # ma swoją własną liczbę/kolejność cech — zawsze padały w backteście)
                            mfeats_t = _ens.feature_names.get(mname, _FN)
                            if mfeats_t != _FN:
                                X_m_t   = np.stack([feat_src.get(f, np.zeros(n)) for f in mfeats_t], axis=1)
                                X_pass_t = X_m_t[mask_idx]
                            else:
                                X_pass_t = X_pass
                            X_sc = sc.transform(X_pass_t) if sc is not None else X_pass_t
                            # MULTI-HORIZON (2026-08-08): cecha 'horizon_hours' nie jest
                            # liczona w feat_src -> model dostawalby ZERA (bug jak rptr/x_).
                            # Sondujemy kazdy horyzont i uodredniamy proba (soft-vote).
                            if 'horizon_hours' in mfeats_t:
                                _hzi = mfeats_t.index('horizon_hours')
                                _acc = None
                                for _hz in MULTI_HORIZONS:
                                    _Xz = X_pass_t.copy()
                                    _Xz[:, _hzi] = _hz
                                    _Xs = sc.transform(_Xz) if sc is not None else _Xz
                                    _p = model.predict_proba(_Xs)
                                    _acc = _p if _acc is None else _acc + _p
                                proba = _acc / len(MULTI_HORIZONS)
                            else:
                                proba = model.predict_proba(X_sc)
                except Exception as _pred_e:
                    logger.debug(f"Predict skip {mname}: {_pred_e}")
                    continue
                lp = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                sp = proba[:, 2] if proba.shape[1] > 2 else (1.0 - lp)
                _wl = np.where(lp > _VOTE_GATE, lp * w_mdl_long, 0.0)
                _ws = np.where(sp > _VOTE_GATE, sp * w_mdl_short, 0.0)
                batch_long[mask_idx]  += _wl
                batch_short[mask_idx] += _ws
                # Zapamietaj surowy (nieprzewazony) glos tego modelu - do
                # atrybucji "kto/czemu" per transakcja (audyt 2026-07-04).
                _lp_full = np.zeros(n); _lp_full[mask_idx] = lp
                _sp_full = np.zeros(n); _sp_full[mask_idx] = sp
                per_model_lp[mname] = _lp_full
                per_model_sp[mname] = _sp_full
                _wl_full = np.zeros(n); _wl_full[mask_idx] = _wl
                _ws_full = np.zeros(n); _ws_full[mask_idx] = _ws
                per_model_wl[mname] = _wl_full
                per_model_ws[mname] = _ws_full

        # GLOSOWANIE WIEKSZOSCIOWE (audyt 2026-07-06, "testy mechanizmow
        # glosowania") - zamiast wazonej sumy prawdopodobienstw, kazdy model
        # oddaje 1 glos (LONG jesli lp>0.52, SHORT jesli sp>0.52, inaczej
        # wstrzymuje sie) - batch_long/batch_short reinterpretowane jako
        # FRAKCJA glosow (0..1) zamiast sumy wazonej. Reszta pipeline'u
        # (prog 0.40, REGIME_ADJUST, session penalty, stochastic boost)
        # dziala bez zmian na tych frakcjach.
        if _VOTING_MODE == "majority" and len(mask_idx) > 0 and per_model_lp:
            _n_models = len(per_model_lp)
            _vote_long  = np.zeros(n)
            _vote_short = np.zeros(n)
            for mname in per_model_lp:
                _vote_long  += (per_model_lp[mname] > 0.52).astype(float)
                _vote_short += (per_model_sp[mname] > 0.52).astype(float)
            batch_long  = _vote_long / _n_models
            batch_short = _vote_short / _n_models

        # WINNER-TAKE-ALL (audyt 2026-07-06, "kolejna metoda votingu") - zamiast
        # sumowania wazonych glosow wszystkich modeli, decyzje podejmuje
        # WYLACZNIE model z najwyzszym wazonym glosem w danej swiecy (MAX
        # zamiast SUM) - pozostale modele sa ignorowane dla tej swiecy.
        # Genuinie inny mechanizm niz "weighted" (suma) i "majority" (liczba
        # glosow) - tu liczy sie POJEDYNCZY najsilniejszy ekspert, nie komitet.
        if _VOTING_MODE == "winner_take_all" and len(mask_idx) > 0 and per_model_wl:
            batch_long  = np.maximum.reduce(list(per_model_wl.values()))
            batch_short = np.maximum.reduce(list(per_model_ws.values()))

        # 4. Wyznacz batch_action (int8): 1=LONG, -1=SHORT, 0=NEUTRAL — wektoryzowane
        # Prog 0.40 sparametryzowany (audyt 2026-07-07, "5. mechanizm: threshold
        # sweep") - _DECISION_THRESHOLD, domyslnie 0.40 (bez zmian), nadpisywalny
        # przez endpoint. Nigdy wczesniej nie testowany na innej wartosci.
        batch_conf   = np.maximum(batch_long, batch_short)
        batch_action = np.zeros(n, dtype=np.int8)
        if len(mask_idx) > 0:
            _ls = batch_long[mask_idx]
            _ss = batch_short[mask_idx]
            # Progi asymetryczne (audyt 2026-07-07) - LONG i SHORT moga miec
            # OSOBNE progi. Fallback na wspolny _DECISION_THRESHOLD gdy None.
            _th_l = _THRESHOLD_LONG if _THRESHOLD_LONG is not None else _DECISION_THRESHOLD
            _th_s = _THRESHOLD_SHORT if _THRESHOLD_SHORT is not None else _DECISION_THRESHOLD
            batch_action[mask_idx] = np.where(
                (_ls > _th_l) & (_ls > _ss), np.int8(1),
                np.where((_ss > _th_s) & (_ss > _ls), np.int8(-1), np.int8(0))
            )

        # KONTRARIAŃSKIE bramki abstencji (of_cvd / BOS / FVG), env HAI_*_GATE=1.
        # Weto kierunkow modelu zgodnych z fadeowanym sygnalem. of_cvd=BTC+pula(of_*),
        # BOS/FVG z samych swiec (portfelowo). Patrz flow_gate.apply_gate.
        if (os.environ.get("HAI_CVD_GATE") == "1" or os.environ.get("HAI_BOS_GATE") == "1"
                or os.environ.get("HAI_FVG_GATE") == "1") and len(mask_idx) > 0:
            try:
                from .flow_gate import apply_gate
                batch_action, _nveto = apply_gate(batch_action, candles_1h, symbol)
                if _nveto:
                    logger.debug(f"cvd_gate: zawetowano {_nveto} sygnalow {symbol}")
            except Exception as _ge:
                logger.debug(f"cvd_gate skip: {_ge}")

        # 5. Pre-compute score/action arrays — eliminuje _score() per-candle call
        # Doktryna wolna: dropujemy wymog zgodnosci z kierunkiem doktryny BB -
        # model trend-following gra swoim kierunkiem (batch_action).
        _valid  = doc_mask & (batch_action != 0)
        if not _DOCTRINE_FREE:
            _valid = _valid & (doc_dir == batch_action)

        # META-LABELING (audyt 2026-07-05) - post-hoc filtr, config-gated,
        # domyslnie wylaczony. WCZESNIEJ tylko core/ensemble.py mial ten hook,
        # ale backtester ma WLASNA, zwektoryzowana sciezke ktora ensemble.predict()
        # calkowicie omija (jak regime-blending) - bez tego wpiecia backtest
        # nigdy by nie odzwierciedlal wlaczenia meta-labelingu w AI Control.
        try:
            from .config import config as _cfg_ml
            if _cfg_ml.META_LABEL_ENABLED:
                from .meta_label import load_meta_label
                _meta = load_meta_label()
                if _meta is not None:
                    valid_idx = np.where(_valid)[0]
                    if len(valid_idx) > 0:
                        _side_is_long = (batch_action[valid_idx] == 1).astype(float)
                        votes_list = [
                            np.where(_side_is_long.astype(bool), per_model_lp[m][valid_idx], per_model_sp[m][valid_idx])
                            for m in per_model_lp
                        ]
                        if votes_list:
                            _va = np.stack(votes_list, axis=1)
                            vote_mean, vote_std = _va.mean(axis=1), _va.std(axis=1)
                            vote_max, vote_min = _va.max(axis=1), _va.min(axis=1)
                        else:
                            vote_mean = vote_std = vote_max = vote_min = np.zeros(len(valid_idx))
                        _zeros = np.zeros(n)
                        meta_X = np.column_stack([
                            batch_conf[valid_idx], vote_mean, vote_std, vote_max, vote_min,
                            bb_pos_arr[valid_idx], regime_arr[valid_idx].astype(float),
                            feat_src.get("adx_14", _zeros)[valid_idx],
                            feat_src.get("atr_pct", _zeros)[valid_idx],
                            feat_src.get("sr_node_strength", _zeros)[valid_idx],
                            feat_src.get("rsi_4h", _zeros)[valid_idx],
                            _side_is_long,
                        ])
                        meta_proba = _meta["model"].predict_proba(_meta["scaler"].transform(meta_X))[:, 1]
                        _meta_block = np.zeros(n, dtype=bool)
                        _meta_block[valid_idx] = meta_proba < _cfg_ml.META_LABEL_THRESHOLD
                        _valid = _valid & ~_meta_block
        except Exception as _meta_e:
            logger.debug(f"meta-label backtester hook skip: {_meta_e}")

        # CONSENSUS-DEPTH GATING (audyt 2026-07-07, K1 z NewHorizonts) - trade
        # wazny tylko gdy >= _CONSENSUS_MIN modeli zgadza sie z KIERUNKIEM decyzji
        # (glos >0.52). Liczymy zgodnych osobno dla LONG i SHORT, wybieramy wg
        # batch_action. Aplikowane PO meta-labelu -> oba filtry lacza sie (AND).
        # Glebokosc konsensu liczona gdy potrzebna do BRAMKI lub do SIZINGU.
        _consensus_depth = None
        _n_cons_models = max(1, len(per_model_lp))
        if (_CONSENSUS_MIN > 0 or _CONSENSUS_SIZING) and per_model_lp:
            _depth_long  = np.zeros(n)
            _depth_short = np.zeros(n)
            for _mn in per_model_lp:
                _depth_long  += (per_model_lp[_mn] > 0.52).astype(float)
                _depth_short += (per_model_sp[_mn] > 0.52).astype(float)
            _consensus_depth = np.where(batch_action == 1, _depth_long,
                                        np.where(batch_action == -1, _depth_short, 0.0))
            if _CONSENSUS_MIN > 0:
                _valid = _valid & (_consensus_depth >= _CONSENSUS_MIN)

        _radj   = np.array([REGIME_ADJUST.get(int(r), 0.0) or 0.0 for r in regime_arr], dtype=np.float64)
        _cpre   = np.clip(batch_conf + _radj, 0.0, 0.99)
        _cpre[sess_arr == 1] = np.clip(_cpre[sess_arr == 1] - 0.03, 0.0, 0.99)
        _sb = ((batch_action == 1) & (stoch_k_arr < 25) & (stoch_d_arr < 25)) | \
              ((batch_action == -1) & (stoch_k_arr > 75) & (stoch_d_arr > 75))
        _cpre[_sb] = np.clip(_cpre[_sb] * 1.08, 0.0, 0.99)
        score_arr      = np.where(_valid, np.round(_cpre * 100, 1), 0.0)
        precomp_action = np.where(_valid, batch_action, np.int8(0)).astype(np.int8)

        trades: List[Dict]    = []
        open_pos              = None
        pyramid_pos           = None
        consecutive_losses    = 0
        pyramid_blocked       = False
        symbol_cooldown_until = None
        daily_pnl             = 0.0
        daily_reset_day       = ""

        for i in range(min_h, n):
            cur     = closes_1h[i]
            ts_ms   = int(times_1h[i])  # np.int64 -> python int (audyt 2026-07-04,
                                         # inaczej trade_log psul json.dumps bez default=str)
            cur_day = date_arr[i]
            hi      = hi_arr[i]
            lo      = lo_arr[i]

            if cur_day != daily_reset_day:
                daily_pnl = 0.0
                daily_reset_day = cur_day

            # ── ZAMKNIĘCIE ────────────────────────────────────────────
            # 2026-08-04: partial-TP + trailing (RAPORT_EDGE.md §12, koncept Hauzera).
            # Do tej pory backtester mial TYLKO stary binarny hit_tp/hit_sl - honest
            # WFV (GitHub/Kaggle/VPS) testowal modele pod STARYM systemem wyjscia,
            # mimo ze engine.py (live/paper) ma juz partial-TP+trailing od 12:04-12:05
            # tego samego dnia. Ten blok domyka ta rozbieznosc.
            if open_pos is not None:
                entry = open_pos["entry"]
                side  = open_pos["side"]
                atr_e = open_pos["atr"]
                tp_p  = entry + atr_e * atr_tp if side == "LONG" else entry - atr_e * atr_tp
                sl_p  = entry - atr_e * atr_sl if side == "LONG" else entry + atr_e * atr_sl

                def _finalize_close(exit_p, result_label, is_sl):
                    nonlocal open_pos, pyramid_pos, consecutive_losses, pyramid_blocked, \
                             symbol_cooldown_until, daily_pnl
                    slip_exit = self.slippage_sl if is_sl else 0.0
                    if side == "LONG":
                        exit_eff = exit_p * (1 - slip_exit)
                        pnl_pct  = (exit_eff - open_pos["entry_eff"]) / open_pos["entry_eff"] * 100
                    else:
                        exit_eff = exit_p * (1 + slip_exit)
                        pnl_pct  = (open_pos["entry_eff"] - exit_eff) / open_pos["entry_eff"] * 100

                    hours_held = (ts_ms - open_pos["open_ts"]) / 3_600_000
                    fee_total  = self.fee_taker * 2 * 100
                    funding    = self.funding_daily_rate / 24 * hours_held * 100
                    pnl_net    = pnl_pct - fee_total - funding
                    # size_usdt tu jest RESZTKĄ (po ew. partial close) - pnl_usdt liczony
                    # tylko na tym co jeszcze zostalo w pozycji.
                    pnl_usdt   = open_pos["size_usdt"] * pnl_net / 100 * self.base_leverage
                    # Dolicz zbankowany partial-close (jesli byl) do finalnego USDT-pnl.
                    # pnl_pct/pnl_net ZOSTAJĄ czyste (ruch ceny resztki) - ta sama zasada
                    # co w state.py::close_position (2026-08-04): pnl_pct to niezalezna
                    # od sizingu miara ruchu ceny, nie miksujemy jej z blended USDT.
                    pnl_usdt  += open_pos.get("realized_partial_pnl", 0.0)

                    pyramid_pnl = 0.0
                    had_pyramid = pyramid_pos is not None
                    if had_pyramid:
                        if side == "LONG":
                            pp = (exit_eff - pyramid_pos["entry_eff"]) / pyramid_pos["entry_eff"] * 100
                        else:
                            pp = (pyramid_pos["entry_eff"] - exit_eff) / pyramid_pos["entry_eff"] * 100
                        pyr_h = (ts_ms - pyramid_pos["open_ts"]) / 3_600_000
                        pp   -= self.fee_taker * 2 * 100 + self.funding_daily_rate / 24 * pyr_h * 100
                        pyramid_pnl = pyramid_pos["size_usdt"] * pp / 100 * self.base_leverage

                    total_pnl = round(pnl_usdt + pyramid_pnl, 2)
                    self.capital += total_pnl
                    daily_pnl   += total_pnl

                    # 2026-08-05 (zlapane przez Hauzera): sizing po stracie (SIZE_SCALE_BY_LOSSES)
                    # MUSI patrzec na faktyczny wynik finansowy (total_pnl), nie na to KTORY branch
                    # zamknal pozycje (is_sl). Przy partial-TP trade moze zamknac sie jako "SL" (SL
                    # trafil na resztce) a mimo to byc NETTO zyskowny (partial zbankowal wiecej niz
                    # reszta stracila - patrz §9.J/§11: 36.5%/103 takich przypadkow w smoke teście).
                    # Stary kod (`if not is_sl`) traktowal taki trade jako strate -> kurczyl pozycje
                    # mimo ze konto realnie zarobilo. is_sl nadal steruje slippage/cooldown (to o
                    # MECHANICE ceny - SL faktycznie zostal dotkniety), ale NIE o sizing.
                    if total_pnl > 0:
                        consecutive_losses = 0
                        pyramid_blocked    = False
                    else:
                        consecutive_losses = min(consecutive_losses + 1, 1)
                        if had_pyramid:
                            pyramid_blocked = True
                    if is_sl and enable_cooldown:
                        symbol_cooldown_until = ts_ms + SL_COOLDOWN_MINUTES * 60_000

                    trades.append({
                        "symbol":      symbol,
                        "side":        side,
                        "entry":       round(entry, 6),
                        "exit":        round(exit_p, 6),
                        "pnl_pct":     round(pnl_pct, 3),
                        "pnl_net":     round(pnl_net, 3),
                        "pnl_usdt":    total_pnl,
                        "result":      result_label,
                        "open_ts":     open_pos["open_ts"],
                        "close_ts":    ts_ms,
                        "atr":         round(atr_e, 6),
                        # size_usdt = ORYGINALNY rozmiar pozycji (przed ew. partial-close),
                        # zeby "size_usdt" mialo ten sam sens (rozmiar wejscia) dla
                        # wszystkich trade'ow, niezaleznie czy byl partial czy nie.
                        "size_usdt":   open_pos.get("original_size_usdt") or open_pos["size_usdt"],
                        "hours_held":  round(hours_held, 1),
                        "had_pyramid": had_pyramid,
                        "pyramid_pnl": round(pyramid_pnl, 2),
                        "bb_pos":      open_pos.get("bb_pos"),
                        "regime":      open_pos.get("regime"),
                        "session":     open_pos.get("session"),
                        "confidence":  open_pos.get("confidence"),
                        "model_votes":      open_pos.get("model_votes"),
                        "feature_snapshot": open_pos.get("feature_snapshot"),
                        "dominant_model":   open_pos.get("dominant_model"),
                    })
                    open_pos    = None
                    pyramid_pos = None

                # === OBRONA POZYCJI: zamek zysku (2026-08-08, koncept Hauzera) =====
                # Gdy pozycja osiagnie +TRIGGER% zwrotu, przesun SL na +LOCK%.
                # Od tego momentu pozycja nie moze wyjsc na minus.
                #
                # UWAGA na dzwignie: "+25% na pozycji" to zwrot NA KAPITALE, a ceny
                # ruszaja sie 5x mniej (base_leverage=5). Stad dzielenie przez
                # dzwignie — bez tego szukalibysmy 25% ruchu ceny, czyli poziomu,
                # ktorego prawie nigdy nie ma i mechanizm nie odpalilby sie wcale.
                # Fee+funding celowo POMINIETE w progu: to ma byc prosty, czytelny
                # poziom cenowy, a nie ruchomy cel zalezny od czasu trzymania.
                #
                # WYLACZONE domyslnie (HAI_SL_LOCK=1 wlacza) — inaczej zmienialoby
                # wyniki wszystkich dotychczasowych przebiegow i nie dalo sie
                # porownac A/B.
                if _SL_LOCK and not open_pos.get("sl_locked"):
                    _lev = self.base_leverage or 1.0
                    _trig_move = _SL_LOCK_TRIGGER / 100.0 / _lev
                    _lock_move = _SL_LOCK_AT / 100.0 / _lev
                    _e = open_pos["entry_eff"]
                    if side == "LONG":
                        if hi >= _e * (1 + _trig_move):
                            sl_p = max(sl_p, _e * (1 + _lock_move))
                            open_pos["sl_locked"] = True
                    else:
                        if lo <= _e * (1 - _trig_move):
                            sl_p = min(sl_p, _e * (1 - _lock_move))
                            open_pos["sl_locked"] = True
                elif _SL_LOCK and open_pos.get("sl_locked"):
                    # poziom raz zablokowany obowiazuje do konca zycia pozycji
                    _e = open_pos["entry_eff"]
                    _lock_move = _SL_LOCK_AT / 100.0 / (self.base_leverage or 1.0)
                    sl_p = (max(sl_p, _e * (1 + _lock_move)) if side == "LONG"
                            else min(sl_p, _e * (1 - _lock_move)))

                if not open_pos.get("partial_closed"):
                    # FAZA 1: pelna pozycja. 50% do TP -> partial close 75% wielkosci.
                    # SL nadal chroni CALA pozycje (jak w oryginale).
                    tp50_p = entry + (tp_p - entry) * 0.5 if side == "LONG" else entry - (entry - tp_p) * 0.5
                    hit_tp50 = (hi >= tp50_p) if side == "LONG" else (lo <= tp50_p)
                    hit_sl   = (lo <= sl_p) if side == "LONG" else (hi >= sl_p)
                    if hit_tp50 and hit_sl:
                        hit_tp50 = False  # niejednoznaczny bar -> konserwatywnie SL (jak oryginal)

                    if hit_sl:
                        _finalize_close(sl_p, "SL", is_sl=True)
                        continue
                    elif hit_tp50:
                        close_frac = 0.75
                        if side == "LONG":
                            p_pct = (tp50_p - open_pos["entry_eff"]) / open_pos["entry_eff"] * 100
                        else:
                            p_pct = (open_pos["entry_eff"] - tp50_p) / open_pos["entry_eff"] * 100
                        p_hours = (ts_ms - open_pos["open_ts"]) / 3_600_000
                        p_net   = p_pct - self.fee_taker * 2 * 100 - self.funding_daily_rate / 24 * p_hours * 100
                        close_usdt = open_pos["size_usdt"] * close_frac
                        p_pnl = close_usdt * p_net / 100 * self.base_leverage

                        open_pos["original_size_usdt"]  = open_pos["size_usdt"]
                        open_pos["size_usdt"]           = round(open_pos["size_usdt"] * (1 - close_frac), 2)
                        open_pos["realized_partial_pnl"] = p_pnl
                        open_pos["partial_closed"]       = True
                        continue

                elif not open_pos.get("trailing_active"):
                    # FAZA 2: partial zrobiony. 75% do TP -> aktywuj trailing na resztce.
                    # SL (oryginalny, pelny) nadal chroni resztke do tego momentu.
                    tp75_p = entry + (tp_p - entry) * 0.75 if side == "LONG" else entry - (entry - tp_p) * 0.75
                    hit_tp75 = (hi >= tp75_p) if side == "LONG" else (lo <= tp75_p)
                    hit_sl   = (lo <= sl_p) if side == "LONG" else (hi >= sl_p)
                    if hit_tp75 and hit_sl:
                        hit_tp75 = False

                    if hit_sl:
                        _finalize_close(sl_p, "SL", is_sl=True)
                        continue
                    elif hit_tp75:
                        open_pos["trailing_active"] = True
                        open_pos["peak_price"] = hi if side == "LONG" else lo
                        continue

                else:
                    # FAZA 3: trailing aktywny na resztce (25%). Cel = 150% oryg. TP.
                    # Trailing-stop = 85% szczytowego pnl% od entry (ratchet, nigdy w dol).
                    peak = open_pos["peak_price"]
                    peak = max(peak, hi) if side == "LONG" else min(peak, lo)
                    open_pos["peak_price"] = peak

                    if side == "LONG":
                        peak_pct = (peak - entry) / entry * 100
                        tp_pct   = (tp_p - entry) / entry * 100
                    else:
                        peak_pct = (entry - peak) / entry * 100
                        tp_pct   = (entry - tp_p) / entry * 100
                    trail_stop_pct  = peak_pct * 0.85
                    full_target_pct = tp_pct * 1.5
                    trail_stop_p  = entry * (1 + trail_stop_pct / 100) if side == "LONG" else entry * (1 - trail_stop_pct / 100)
                    full_target_p = entry * (1 + full_target_pct / 100) if side == "LONG" else entry * (1 - full_target_pct / 100)

                    hit_target = (hi >= full_target_p) if side == "LONG" else (lo <= full_target_p)
                    hit_trail  = (lo <= trail_stop_p) if side == "LONG" else (hi >= trail_stop_p)

                    if hit_target:
                        _finalize_close(full_target_p, "TP150", is_sl=False)
                        continue
                    elif hit_trail:
                        _finalize_close(trail_stop_p, "TRAIL", is_sl=False)
                        continue

            # ── PYRAMID LAYER ─────────────────────────────────────────
            if (open_pos is not None and enable_pyramid
                    and pyramid_pos is None and not pyramid_blocked):
                _pa = precomp_action[i]
                if _pa != 0 and score_arr[i] >= 70.0:
                    act_pyr = "LONG" if _pa == 1 else "SHORT"
                    if act_pyr == open_pos["side"]:
                        scale_pyr   = SIZE_SCALE_BY_LOSSES.get(min(consecutive_losses, 1), 0.1)
                        size_pyr    = round(self.order_size * scale_pyr * 0.5, 2)
                        slip        = self.slippage_entry
                        e_eff       = cur * (1 + slip) if act_pyr == "LONG" else cur * (1 - slip)
                        pyramid_pos = {"entry_eff": e_eff, "size_usdt": size_pyr, "open_ts": ts_ms}

            # ── OTWARCIE ──────────────────────────────────────────────
            if open_pos is not None:
                continue
            if enable_cooldown and symbol_cooldown_until and ts_ms < symbol_cooldown_until:
                continue
            if enable_daily_limit and daily_pnl <= -daily_loss_limit:
                continue

            _act = precomp_action[i]
            if _act == 0:
                continue
            score = score_arr[i]
            if score < min_conf * 100:
                continue
            action = "LONG" if _act == 1 else "SHORT"

            atr_e = atr_arr[i]
            if atr_e <= 0 or np.isnan(atr_e):
                continue

            scale     = SIZE_SCALE_BY_LOSSES.get(min(consecutive_losses, 1), 0.1)
            if consecutive_losses > 0 and pyramid_blocked:
                scale = PYRAMID_SL_SCALE
            if _CONF_SIZING_ENABLED:
                _margin = max(0.0, min(1.0, (score / 100 - min_conf) / max(1e-6, 1.0 - min_conf)))
                scale *= (0.5 + _margin)
            if _CONSENSUS_SIZING and _consensus_depth is not None:
                scale *= (0.5 + _consensus_depth[i] / _n_cons_models)
            size_usdt = round(self.order_size * scale, 2)
            slip      = self.slippage_entry
            entry_eff = cur * (1 + slip) if action == "LONG" else cur * (1 - slip)

            reg = int(regime_arr[i])
            sess_name = ["normal", "prime", "dead"][int(sess_arr[i])]

            # "Kto/czemu/za ile" (audyt 2026-07-04) - glos kazdego modelu z
            # osobna + wartosci kluczowych cech w momencie otwarcia, zeby
            # widziec NA CZYM konkretnie model sie oparl, nie tylko finalny
            # score ensemble.
            model_votes = {
                mname: {"long": round(float(per_model_lp[mname][i]), 3),
                        "short": round(float(per_model_sp[mname][i]), 3)}
                for mname in per_model_lp
            }
            # Ktory model byl NAJBARDZIEJ przekonany w kierunku faktycznie
            # otwartej pozycji - odpowiedz na "kto zlapal ta transakcje"
            # (audyt 2026-07-04).
            _side_key = "long" if action == "LONG" else "short"
            dominant_model = (
                max(model_votes, key=lambda m: model_votes[m][_side_key])
                if model_votes else None
            )
            # FIX 2026-08-06: bylo TYLKO `_TOP_FEATURES_SNAPSHOT` - sztywna lista 10
            # cech, niezalezna od tego czego model NAPRAWDE uzywa. Skutek: przy
            # configach z feature_mix (RPGC v7: 18 cech/model, w tym r_*/e_*/x_*)
            # snapshot zapisywal cechy, ktorych model w ogole nie widzial, a jego
            # wlasnych - nie. Analiza sygnal/szum po cechach byla przez to niemozliwa.
            # Teraz: cechy FAKTYCZNIE uzywane przez modele w ensemble (union), plus
            # stala lista jako kontekst rynkowy. Fallback na stara liste gdy ensemble
            # nie wystawia feature_names.
            feature_snapshot = {
                f: round(float(feat_src[f][i]), 4)
                for f in _SNAP_FEATS if f in feat_src
            }

            open_pos = {
                "side":       action,
                "entry":      cur,
                "entry_eff":  entry_eff,
                "atr":        atr_e,
                "open_ts":    ts_ms,
                "size_usdt":  size_usdt,
                "bb_pos":     float(bb_pos_arr[i]),
                "regime":     reg if reg >= 0 else None,
                "session":    sess_name,
                "confidence": round(score / 100, 3),
                "model_votes":       model_votes,
                "feature_snapshot":  feature_snapshot,
                "dominant_model":    dominant_model,
            }

        return trades

    # ─────────────────────────────────────────────────────────────────────
    # ASYNC RUNNER — Semaphore + ThreadPoolExecutor
    # ─────────────────────────────────────────────────────────────────────

    async def _run_symbol_async(
        self, sym: str, days: int, mode: str,
        semaphore: asyncio.Semaphore,
        executor: ThreadPoolExecutor,
    ) -> List[Dict]:
        """Odpala run_simulation_ai w thread pool — GIL zwalniany przez numpy/sklearn/xgb."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            try:
                c1h = self.load_candles_from_warehouse(sym, "1h", days)
                c4h = self.load_candles_from_warehouse(sym, "4h", days)
                c1d = self.load_candles_from_warehouse(sym, "1d", days)
                if not c1h:
                    return []
                # run_simulation_ai jest CPU-bound ale numpy/sklearn zwalniają GIL
                # → faktyczny parallelizm w ThreadPoolExecutor
                trades = await loop.run_in_executor(
                    executor,
                    lambda: self.run_simulation_ai(c1h, c4h, c1d, sym, mode)
                )
                logger.debug(f"{sym}: {len(trades)} trades")
                return trades
            except Exception as e:
                logger.error(f"{sym}: {e}")
                return []
            finally:
                # Progres widoczny w logu (audyt 2026-07-04 - user chcial widziec
                # ile z 106 symboli juz policzone). self._progress ustawiane w
                # run_full_ai/run_wfv przed odpaleniem gather().
                prog = getattr(self, "_progress", None)
                if prog is not None:
                    prog["done"] += 1
                    if prog["done"] % 10 == 0 or prog["done"] == prog["total"]:
                        logger.info(f"Progres: {prog['done']}/{prog['total']} symboli")
                    if isinstance(self.stats, dict):
                        self.stats["progress"] = f"{prog['done']}/{prog['total']}"
                    # Co ~25% wpisujemy tez do AI Log (nie tylko surowy log
                    # Pythona) - widoczne w dashboardzie jako historia postepu
                    step = max(1, prog["total"] // 4)
                    if prog["done"] % step == 0 or prog["done"] == prog["total"]:
                        try:
                            from .state import state
                            state.add_log("ai", "INFO", event="BACKTEST",
                                          message=f"Postep: {prog['done']}/{prog['total']} symboli")
                        except Exception:
                            pass

    async def run_full_ai(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 90,
        mode: str = "neutral",
    ) -> Dict:
        """Pełny backtest — MAX_CONCURRENT_SYMBOLS symboli równolegle."""
        if symbols is None:
            symbols = self._list_symbols()

        from .ensemble import ensemble
        if not ensemble.active:
            ensemble.load_models()

        self.capital = 1000.0
        semaphore    = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
        self._progress = {"done": 0, "total": len(symbols)}
        # Status ustawiany TUTAJ (nie tylko przez wrapper /backtest/full w
        # trading.py) - audyt 2026-07-04, bez tego kazdy inny caller run_full_ai
        # (np. standalone skrypty ktore i tak dziela ta sama baze ai_logs)
        # zostawial self.stats na "idle" mimo aktywnego liczenia, myle frontend.
        if isinstance(self.stats, dict):
            self.stats["status"] = "running"
        logger.info(f"run_full_ai start: {len(symbols)} symboli, {days}d, mode={mode}")

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYMBOLS) as executor:
            tasks    = [self._run_symbol_async(s, days, mode, semaphore, executor)
                        for s in symbols]
            results  = await asyncio.gather(*tasks)

        all_trades = [t for trades in results for t in trades]
        out = {"status": "ok", "mode": mode, "days": days,
               **self._calc_stats(all_trades, days, include_trade_log=True)}
        if isinstance(self.stats, dict):
            self.stats["status"] = "ok"
        return out

    async def run_screening(
        self,
        symbols: Optional[List[str]] = None,
        mode: str = "neutral",
    ) -> Dict:
        """Screening: 3 okresy (90/180/365d) → {SYM: {d90, d180, d365, score}}."""
        if symbols is None:
            symbols = self._list_symbols()

        from .ensemble import ensemble
        if not ensemble.active:
            ensemble.load_models()

        per_sym: Dict[str, Dict] = {s: {} for s in symbols}

        for days in [90, 180, 365]:
            self.capital = 1000.0
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYMBOLS) as executor:
                tasks   = [self._run_symbol_async(s, days, mode, semaphore, executor) for s in symbols]
                results = await asyncio.gather(*tasks)
            for sym, trades in zip(symbols, results):
                pnl = sum(t["pnl_usdt"] for t in trades)
                per_sym[sym][f"d{days}"] = round(pnl, 2)

        for sym, data in per_sym.items():
            data["score"] = sum(1 for k in ["d90", "d180", "d365"] if data.get(k, 0) > 0)

        return per_sym

    async def run_walk_forward(
        self,
        symbols: Optional[List[str]] = None,
        window_days: int = 30,
        n_windows:   int = 4,
        mode:        str = "neutral",
    ) -> Dict:
        """Walk-forward: N kolejnych okien po window_days dni (równolegle per symbol)."""
        if symbols is None:
            symbols = self._list_symbols()

        from .ensemble import ensemble
        if not ensemble.active:
            ensemble.load_models()

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
        windows_results = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYMBOLS) as executor:
            for w in range(n_windows):
                offset_end   = (n_windows - 1 - w) * window_days
                offset_start = offset_end + window_days
                label        = f"W{w+1} (-{offset_start}d → -{offset_end}d)"

                self.capital = 1000.0

                async def _run_window_sym(sym, os=offset_start, oe=offset_end):
                    async with semaphore:
                        loop = asyncio.get_event_loop()
                        c1h  = self._load_window(sym, "1h", os, oe)
                        c4h  = self._load_window(sym, "4h", os, oe)
                        c1d  = self._load_window(sym, "1d", os, oe)
                        if not c1h:
                            return []
                        return await loop.run_in_executor(
                            executor,
                            lambda: self.run_simulation_ai(c1h, c4h, c1d, sym, mode)
                        )

                results    = await asyncio.gather(*[_run_window_sym(s) for s in symbols])
                all_trades = [t for trades in results for t in trades]
                windows_results.append({"window": label, **self._calc_stats(all_trades)})

        return {
            "status": "ok", "mode": mode,
            "window_days": window_days, "n_windows": n_windows,
            "windows": windows_results,
        }

    # ─────────────────────────────────────────────────────────────────────
    # EXPANDING WALK-FORWARD VALIDATION (WFV)
    # ─────────────────────────────────────────────────────────────────────

    def _wfv_window_stats(self, trades: List[Dict]) -> Dict:
        """Rozszerzone statystyki dla okna WFV: _calc_stats + Sharpe + syms_pf_pct."""
        base = self._calc_stats(trades, include_trade_log=_INCLUDE_TRADE_LOG)

        # Sharpe ratio (annualizowany, na dziennych PnL)
        daily: Dict[str, float] = {}
        for t in trades:
            day = datetime.fromtimestamp(t["close_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + t["pnl_usdt"]
        if len(daily) >= 5:
            pnls = list(daily.values())
            mu   = sum(pnls) / len(pnls)
            std  = (sum((x - mu) ** 2 for x in pnls) / len(pnls)) ** 0.5
            sharpe = round(mu / std * (252 ** 0.5), 2) if std > 0 else 0.0
        else:
            sharpe = 0.0

        # % symboli z PF >= 1.5
        by_sym: Dict[str, list] = {}
        for t in trades:
            by_sym.setdefault(t["symbol"], []).append(t)
        pf_ok = 0
        for st in by_sym.values():
            gw = sum(t["pnl_usdt"] for t in st if t["pnl_usdt"] > 0)
            gl = abs(sum(t["pnl_usdt"] for t in st if t["pnl_usdt"] < 0))
            if gw / gl >= 1.5 if gl > 0 else False:
                pf_ok += 1
        syms_pf_pct = round(pf_ok / len(by_sym) * 100, 1) if by_sym else 0.0

        base["sharpe_ratio"]       = sharpe
        base["syms_pf_gt15_pct"]   = syms_pf_pct
        base["syms_tested"]        = len(by_sym)
        return base

    @staticmethod
    def _wfv_verdict(windows: List[Dict]) -> Dict:
        """Go / Warning / No-Go na podstawie wyników okien."""
        pfs   = [w["profit_factor"] for w in windows if w["total_trades"] > 0]
        dds   = [w["max_drawdown_pct"] for w in windows if w["total_trades"] > 0]
        wrs   = [w["win_rate"] for w in windows if w["total_trades"] > 0]
        trades = [w["total_trades"] for w in windows]

        if not pfs:
            return {"decision": "NO_DATA", "reason": "Brak transakcji w oknach"}

        # float()/int() jawnie - audyt 2026-07-06, sum()/round() na wartosciach
        # numpy (z pandas/numpy obliczen w _wfv_window_stats) zwraca np.float64,
        # ktorego porownania (>=, <) daja numpy.bool_ - NIE Python bool - co
        # wywala json.dumps() ("Object of type bool is not JSON serializable")
        # przy pierwszym W6/6 ktore faktycznie doszlo do konca bez restartu.
        avg_pf  = float(round(sum(pfs) / len(pfs), 3))   # macro-srednia, TYLKO do raportu
        max_dd  = float(round(max(dds), 1))
        min_pf  = float(round(min(pfs), 3))
        weak_w  = int(sum(1 for p in pfs if p < 1.20))
        avg_tr  = float(round(sum(trades) / len(trades), 0))

        # === POOLED (2026-08-06) — to decyduje o GO/NO_GO ===========================
        # Bylo: decyzja na `avg_pf` = SREDNIA z PF poszczegolnych okien. To srednia
        # ILORAZOW, ktora systematycznie ZAWYZA (nierownosc Jensena): pojedyncze okno
        # z malymi stratami daje PF=20+ i samo podbija cala srednia. Zmierzone na
        # CAT-sniper-rptr@0.75: okna [1.83, 6.33, 6.20, 1.68, 2.33, 1.83, 2.02,
        # 19.96, 2.70, 2.49, 1.89] -> srednia 4.478 vs POOLED 2.660 (+68% zawyzenia).
        # Teraz decyzja idzie z POOLED = iloraz SUM (wszystkie trady traktowane rowno,
        # okno z 3 tradami nie wazy tyle co okno z 300). To samo dla WR.
        _gw = sum(float(w.get("gross_win") or 0.0) for w in windows if w["total_trades"] > 0)
        _gl = sum(float(w.get("gross_loss") or 0.0) for w in windows if w["total_trades"] > 0)
        _has_gross = (_gw > 0 or _gl > 0)
        if _has_gross:
            pooled_pf = float(round(_gw / _gl, 3)) if _gl > 0 else 999.0
        else:
            # Fallback dla starych okien bez gross_win/gross_loss (dane sprzed
            # 2026-08-06): wazenie PF liczba tradow - blizej pooled niz zwykla
            # srednia, ale NIE identyczne. Oznaczone ponizej flaga pf_basis.
            _tw = sum(w["total_trades"] for w in windows if w["total_trades"] > 0) or 1
            pooled_pf = float(round(
                sum(w["profit_factor"] * w["total_trades"]
                    for w in windows if w["total_trades"] > 0) / _tw, 3))

        _wins = sum(int(w.get("wins") or 0) for w in windows if w["total_trades"] > 0)
        _tr_ok = sum(w["total_trades"] for w in windows if w["total_trades"] > 0)
        pooled_wr = float(round(_wins / _tr_ok * 100, 1)) if _tr_ok else 0.0
        # avg_wr zwracane dalej pod ta sama nazwa (kompatybilnosc bazy/dashboardu),
        # ale liczone jako POOLED - bo tez bylo macro-srednia i tez zawyzalo.
        avg_wr = pooled_wr if _tr_ok else float(round(sum(wrs) / len(wrs), 1))

        if pooled_pf >= 1.45 and max_dd < 18.0 and weak_w == 0:
            decision = "GO"
            color    = "green"
        elif pooled_pf >= 1.35 and max_dd < 22.0 and weak_w <= 1:
            decision = "WARNING"
            color    = "yellow"
        else:
            decision = "NO_GO"
            color    = "red"

        return {
            "decision":     decision,
            "color":        color,
            "avg_pf":       pooled_pf,        # <- POOLED (to co decyduje); nazwa dla zgodnosci
            "avg_pf_macro": avg_pf,           # <- stara srednia-z-okien, tylko informacyjnie
            "pf_basis":     "pooled" if _has_gross else "trade_weighted_fallback",
            "min_pf":       min_pf,
            "avg_wr":       avg_wr,           # <- POOLED
            "max_dd":       max_dd,
            "weak_windows": weak_w,
            "avg_trades":   avg_tr,
            "criteria": {
                "pf_ok":  pooled_pf >= 1.45,
                "dd_ok":  max_dd < 18.0,
                "weak_ok": weak_w == 0,
            },
        }

    async def run_wfv(
        self,
        n_windows:    int = 6,
        window_days:  int = 90,
        embargo_days: int = 7,
        mode:         str = "neutral",
        symbols:      Optional[List[str]] = None,
    ) -> Dict:
        """
        Expanding Walk-Forward Validation:
          - n_windows okien testowych po window_days dni
          - embargo_days przerwy między oknem treningowym a testowym
          - Okna od najstarszego (W1) do najnowszego (W{n})
          - Raportuje: PF, WR, MaxDD, Sharpe, trades, regime, go/no-go
        """
        if symbols is None:
            symbols = self._list_symbols()

        from .ensemble import ensemble
        if not ensemble.active:
            ensemble.load_models()

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
        windows_out = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYMBOLS) as executor:
            # W1 = najstarsze, W{n} = najnowsze
            # offset_end  = ile dni temu kończy się okno testowe
            # offset_start = offset_end + window_days
            for w in range(n_windows):
                # Okna od najstarszego do najnowszego
                idx = n_windows - 1 - w   # 0 = najnowsze
                # Embargo realnie przesuwa granice okna (nie tylko etykieta -
                # audyt 2026-07-04): idx*(window_days+embargo_days) daje
                # skumulowany embargo_days-dniowy odstep miedzy KAZDA para
                # sasiadujacych okien (zweryfikowane numerycznie), zamiast
                # poprzedniego kosmetycznego dopisku do etykiety bez realnego
                # wplywu na granice.
                offset_end   = idx * (window_days + embargo_days)
                offset_start = offset_end + window_days
                label = (
                    f"W{w+1} | -{offset_start}d → -{offset_end}d"
                    + (f" [embargo {embargo_days}d]" if embargo_days else "")
                )
                self.wfv_progress = f"W{w+1}/{n_windows}"

                _win_progress = {"done": 0, "total": len(symbols)}

                async def _sym(sym, os=offset_start, oe=offset_end):
                    async with semaphore:
                        loop = asyncio.get_event_loop()
                        c1h = self._load_window(sym, "1h", os, oe)
                        c4h = self._load_window(sym, "4h", os, oe)
                        c1d = self._load_window(sym, "1d", os, oe)
                        if not c1h:
                            return []
                        try:
                            return await loop.run_in_executor(
                                executor,
                                lambda c1=c1h, c4=c4h, c1d_=c1d: self.run_simulation_ai(c1, c4, c1d_, sym, mode)
                            )
                        finally:
                            # Progres widoczny w logu (audyt 2026-07-04)
                            _win_progress["done"] += 1
                            if _win_progress["done"] % 10 == 0 or _win_progress["done"] == _win_progress["total"]:
                                logger.info(f"WFV {label}: {_win_progress['done']}/{_win_progress['total']} symboli")
                            if _win_progress["done"] == _win_progress["total"]:
                                try:
                                    from .state import state
                                    state.add_log("ai", "INFO", event="WFV",
                                                  message=f"WFV {label}: okno ukonczone ({_win_progress['total']} symboli)")
                                except Exception:
                                    pass

                results    = await asyncio.gather(*[_sym(s) for s in symbols])
                all_trades = [t for r in results for t in r]
                stats      = self._wfv_window_stats(all_trades)
                windows_out.append({"window": label, "offset_end": offset_end,
                                    "offset_start": offset_start, **stats})

        verdict = self._wfv_verdict(windows_out)

        return {
            "status":       "ok",
            "mode":         mode,
            "n_windows":    n_windows,
            "window_days":  window_days,
            "embargo_days": embargo_days,
            "symbols":      len(symbols),
            "verdict":      verdict,
            "windows":      windows_out,
        }

    # ─────────────────────────────────────────────────────────────────────
    # METRYKI — pełna analiza doktrynowa
    # ─────────────────────────────────────────────────────────────────────

    def _calc_stats(self, trades: List[Dict], days: int = 90, include_trade_log: bool = False) -> Dict:
        total = len(trades)
        if total == 0:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl_usdt": 0, "profit_factor": 0,
                "gross_win": 0.0, "gross_loss": 0.0,   # 2026-08-06: do POOLED PF w _wfv_verdict
                "max_drawdown_pct": 0, "avg_hold_hours": 0,
                "longs": 0, "shorts": 0, "circuit_breaker_days": 0,
                "doctrine": {}, "pyramiding": {}, "top_symbols": [], "trade_log": [],
                "model_attribution": {},
            }

        # 2026-08-05: win = wynik finansowy (pnl_usdt>0), NIE etykieta "result".
        # Partial-TP (2026-08-04) zmienil etykiety na SL/TP150/TRAIL - literalne
        # "TP" juz nigdy nie powstaje, wiec kazdy check == "TP" dawal zawsze 0.
        wins   = sum(1 for t in trades if t["pnl_usdt"] > 0)
        losses = total - wins
        wr     = wins / total * 100
        pnl    = sum(t["pnl_usdt"] for t in trades)
        pf_win = sum(t["pnl_usdt"] for t in trades if t["pnl_usdt"] > 0)
        pf_los = abs(sum(t["pnl_usdt"] for t in trades if t["pnl_usdt"] < 0))
        pf     = round(pf_win / pf_los, 2) if pf_los > 0 else 999.0
        avg_h  = round(sum(t.get("hours_held", 0) for t in trades) / total, 1)

        # Equity curve + max drawdown + circuit breaker days
        equity    = 1000.0
        peak      = equity
        max_dd    = 0.0
        daily_pnl: Dict[str, float] = {}
        for t in sorted(trades, key=lambda x: x["close_ts"]):
            equity += t["pnl_usdt"]
            peak    = max(peak, equity)
            dd      = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd  = max(max_dd, dd)
            day     = datetime.fromtimestamp(t["close_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0) + t["pnl_usdt"]
        circuit_breaks = sum(1 for v in daily_pnl.values() if v <= -50.0)

        longs  = [t for t in trades if t["side"] == "LONG"]
        shorts = [t for t in trades if t["side"] == "SHORT"]

        def _side(ts, is_long):
            if not ts:
                return {"count": 0, "win_rate": 0, "pnl": 0, "pf": 0,
                        "bb_quality_pct": 0, "bb_lt25_pct": 0, "bb_gt75_pct": 0}
            w    = sum(1 for t in ts if t["pnl_usdt"] > 0)
            spnl = sum(t["pnl_usdt"] for t in ts)
            gw   = sum(t["pnl_usdt"] for t in ts if t["pnl_usdt"] > 0)
            gl   = abs(sum(t["pnl_usdt"] for t in ts if t["pnl_usdt"] < 0))
            bb   = [t for t in ts if t.get("bb_pos") is not None]
            bb_ok   = sum(1 for t in bb if (is_long and t["bb_pos"] < 0.20) or
                                           (not is_long and t["bb_pos"] > 0.80))
            bb_lt25 = sum(1 for t in bb if t["bb_pos"] < 0.25)
            bb_gt75 = sum(1 for t in bb if t["bb_pos"] > 0.75)
            return {
                "count":          len(ts),
                "win_rate":       round(w / len(ts) * 100, 1),
                "pnl":            round(spnl, 2),
                "pf":             round(gw / gl, 2) if gl > 0 else 0.0,
                "bb_quality_pct": round(bb_ok / len(bb) * 100, 1) if bb else 0,
                "bb_lt25_pct":    round(bb_lt25 / len(bb) * 100, 1) if bb else 0,
                "bb_gt75_pct":    round(bb_gt75 / len(bb) * 100, 1) if bb else 0,
            }

        def _regime(rid):
            rt   = [t for t in trades if t.get("regime") == rid]
            if not rt:
                return {"count": 0, "win_rate": 0, "pf": 0, "pnl": 0}
            rw   = sum(1 for t in rt if t["pnl_usdt"] > 0)
            rpnl = sum(t["pnl_usdt"] for t in rt)
            rgw  = sum(t["pnl_usdt"] for t in rt if t["pnl_usdt"] > 0)
            rgl  = abs(sum(t["pnl_usdt"] for t in rt if t["pnl_usdt"] < 0))
            return {
                "count":    len(rt),
                "win_rate": round(rw / len(rt) * 100, 1),
                "pf":       round(rgw / rgl, 2) if rgl > 0 else 0.0,
                "pnl":      round(rpnl, 2),
            }

        def _sess(name):
            st = [t for t in trades if t.get("session") == name]
            if not st:
                return {"count": 0, "win_rate": 0}
            sw = sum(1 for t in st if t["pnl_usdt"] > 0)
            return {"count": len(st), "win_rate": round(sw / len(st) * 100, 1)}

        # Pyramiding
        pyr  = [t for t in trades if t.get("had_pyramid")]
        base = [t for t in trades if not t.get("had_pyramid")]
        pw   = sum(1 for t in pyr if t["pnl_usdt"] > 0)
        ppnl = sum(t.get("pyramid_pnl", 0) for t in pyr)
        pgw  = sum(t["pnl_usdt"] for t in pyr if t["pnl_usdt"] > 0)
        pgl  = abs(sum(t["pnl_usdt"] for t in pyr if t["pnl_usdt"] < 0))
        bw   = sum(1 for t in base if t["pnl_usdt"] > 0)
        bgw  = sum(t["pnl_usdt"] for t in base if t["pnl_usdt"] > 0)
        bgl  = abs(sum(t["pnl_usdt"] for t in base if t["pnl_usdt"] < 0))

        # Top symbole
        by_sym: Dict[str, list] = {}
        for t in trades:
            by_sym.setdefault(t["symbol"], []).append(t)
        top = sorted([
            {
                "symbol": s,
                "trades": len(st),
                "pf":     round(sum(t["pnl_usdt"] for t in st if t["pnl_usdt"] > 0) /
                                max(abs(sum(t["pnl_usdt"] for t in st if t["pnl_usdt"] < 0)), 0.01), 2),
                "pnl":    round(sum(t["pnl_usdt"] for t in st), 2),
                "wr":     round(sum(1 for t in st if t["pnl_usdt"] > 0) / len(st) * 100, 1),
            }
            for s, st in by_sym.items()
        ], key=lambda x: x["pf"], reverse=True)[:5]

        # "Kto zlapal ile transakcji i z jakim skutkiem" - agregat po
        # dominant_model (audyt 2026-07-04).
        by_model: Dict[str, list] = {}
        for t in trades:
            dm = t.get("dominant_model")
            if dm:
                by_model.setdefault(dm, []).append(t)
        model_attribution = {
            m: {
                "count":    len(mt),
                "win_rate": round(float(sum(1 for t in mt if t["pnl_usdt"] > 0) / len(mt) * 100), 1),
                "pf":       round(float(sum(t["pnl_usdt"] for t in mt if t["pnl_usdt"] > 0) /
                                  max(abs(sum(t["pnl_usdt"] for t in mt if t["pnl_usdt"] < 0)), 0.01)), 2),
                "pnl":      round(float(sum(t["pnl_usdt"] for t in mt)), 2),
            }
            for m, mt in by_model.items()
        }

        # AGREGACJA CECH per WYGRANA/PRZEGRANA (audyt 2026-07-07, "kod") - lekka,
        # bez zapisywania surowych trade_logow. Dla kazdej cechy ze snapshotu:
        # srednia wartosc w WYGRANYCH (TP) vs PRZEGRANYCH + delta (im wieksza
        # |delta|, tym cecha bardziej rozroznia wygrane od przegranych).
        _fw: Dict[str, list] = {}
        _fl: Dict[str, list] = {}
        for _t in trades:
            _fs = _t.get("feature_snapshot") or {}
            _bucket = _fw if _t["pnl_usdt"] > 0 else _fl
            for _f, _v in _fs.items():
                if _v is not None:
                    _bucket.setdefault(_f, []).append(_v)
        feature_attribution = {}
        for _f in set(list(_fw) + list(_fl)):
            _w, _l = _fw.get(_f, []), _fl.get(_f, [])
            _wm = round(sum(_w) / len(_w), 4) if _w else None
            _lm = round(sum(_l) / len(_l), 4) if _l else None
            feature_attribution[_f] = {
                "win_mean": _wm, "loss_mean": _lm,
                "n_wins": len(_w), "n_losses": len(_l),
                "delta": round(_wm - _lm, 4) if (_wm is not None and _lm is not None) else None,
            }

        bb_all = [t for t in trades if t.get("bb_pos") is not None]
        bb_ok_all = sum(
            1 for t in bb_all
            if (t["side"] == "LONG" and t["bb_pos"] < 0.20) or
               (t["side"] == "SHORT" and t["bb_pos"] > 0.80)
        )

        return {
            "total_trades":         total,
            "wins":                 wins,
            "losses":               losses,
            "win_rate":             round(wr, 1),
            "total_pnl_usdt":       round(pnl, 2),
            "profit_factor":        pf,
            # 2026-08-06: surowe sumy zysk/strata per okno - potrzebne zeby
            # _wfv_verdict mogl policzyc POOLED PF (iloraz sum), zamiast
            # sredniej z ilorazow ktora systematycznie ZAWYZA (nierownosc
            # Jensena: jedno okno z PF=19.96 podbijalo srednia o +68%).
            "gross_win":            round(pf_win, 4),
            "gross_loss":           round(pf_los, 4),
            "max_drawdown_pct":     round(max_dd, 1),
            "avg_hold_hours":       avg_h,
            "longs":                len(longs),
            "shorts":               len(shorts),
            "circuit_breaker_days": circuit_breaks,
            "doctrine": {
                "bb_quality_all_pct": round(bb_ok_all / len(bb_all) * 100, 1) if bb_all else 0,
                "longs":   _side(longs, True),
                "shorts":  _side(shorts, False),
                "regime":  {
                    "trend":    _regime(0),
                    "mean_rev": _regime(1),
                    "high_vol": _regime(2),
                    "unknown":  _regime(None),
                },
                "session": {
                    "prime":  _sess("prime"),
                    "normal": _sess("normal"),
                    "dead":   _sess("dead"),
                },
            },
            "trade_log": (
                sorted(trades, key=lambda x: x["close_ts"]) if include_trade_log else []
            ),  # pelna lista transakcji z model_votes/feature_snapshot (audyt 2026-07-04)
            "model_attribution": model_attribution,  # kto zlapal ile i z jakim skutkiem
            "feature_attribution": feature_attribution,  # srednia cech w win vs loss + delta
            "pyramiding": {
                "count":          len(pyr),
                "pct_of_trades":  round(len(pyr) / total * 100, 1) if total else 0,
                "win_rate":       round(pw / len(pyr) * 100, 1) if pyr else 0,
                "sl_count":       sum(1 for t in pyr if t["result"] == "SL"),
                "pnl_from_layer": round(ppnl, 2),
                "pf":             round(pgw / pgl, 2) if pgl > 0 else 0.0,
                "base_win_rate":  round(bw / len(base) * 100, 1) if base else 0,
                "base_pf":        round(bgw / bgl, 2) if bgl > 0 else 0.0,
            },
            "top_symbols": top,
        }

    # ─────────────────────────────────────────────────────────────────────
    # INDICATORS
    # ─────────────────────────────────────────────────────────────────────

    def _calc_atr(self, candles: List[Dict], i: int, period: int = 14) -> float:
        if i < period + 1:
            return 0.0
        window = candles[max(0, i - period - 1): i]
        trs = []
        for j in range(1, len(window)):
            h, l, pc = window[j]["high"], window[j]["low"], window[j-1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs[-period:]) / period if trs else 0.0

backtester = Backtester()

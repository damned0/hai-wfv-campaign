# ===========================================
# HAI_EPV Engine ver.10 Final — core/ml_trainer.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - build_features_for_symbol()/build_dataset() - cechy TA + Triple Barrier
#   labeling (3-class: NEUTRAL/LONG/SHORT), rownolegle po symbolach
# - triple_barrier_label() - reuzywalna funkcja labelowania (dowolny horyzont/TP)
# - MODEL_FEATURES/MODEL_LABEL_COLUMN - specjalizacja cech i labelu per model
# - HORIZON_CHOICES - rejestr wariantow horyzontu do treningu niestandardowego
# - train_models()/train_and_save() - trening 5 algorytmow (LGB/RF/XGB/CAT/HistGB)
#   + specjalisci (fast24h/sharp6x/itd.), zapis .pkl (staging _NEW)
# ===========================================
"""
ML Trainer ver.10 - zoptymalizowany pod multi-core.

Triple Barrier (Marcos Lopez de Prado):
  Dla kazdej swiecy patrzy w przyszlosc 48 swiec (2 dni 1h).
  TP = +4.0 * ATR, SL = -1.0 * ATR (TP_ATR_MULT/SL_ATR_MULT ponizej)
  Label = 1 jezeli hit TP przed SL, 0 inaczej.

Trening: chronologiczny train/val split (NO shuffle - time series!).
Modele zapisywane jako _NEW.pkl - atomic swap przez endpoint.
"""
import gc
import json
import joblib
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Sciezki
WH_BASE = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse")) / 'ohlcv' / 'binance'
MODELS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Konfiguracja
# v7 (30maja): auto-detekcja z magazynu (jak LAB/LIV) - zawsze zgodne z warehouse
SYMBOLS = sorted([f.stem for f in (WH_BASE / '1h').glob('*.parquet')])
# Opcjonalny whitelist przez HAI_SYMBOLS (CSV) — do lekkiego build_dataset/WFV A/B
# na podzbiorze coinow. Puste = wszystkie (stare zachowanie).
_SYM_WL = os.getenv("HAI_SYMBOLS", "").strip()
if _SYM_WL:
    _wl = {s.strip().upper() for s in _SYM_WL.split(",") if s.strip()}
    SYMBOLS = [s for s in SYMBOLS if s.upper() in _wl]

# Hyperparams Triple Barrier
LOOKAHEAD_BARS = 48      # 48 swiec 1h = 2 dni
TP_ATR_MULT = 4.0        # zysk = 4.0 * ATR (R/R 4:1 → PF ~1.7 przy WR 32%, spójne ze starym backtestem)
SL_ATR_MULT = 1.0        # strata = 1.0 * ATR
MIN_HISTORY = 80         # min swiec przed liczeniem featurow

# Wagi feature engineering
RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
EMA_MID = 50

# Train/Val split
VAL_RATIO = 0.20  # ostatnie 20% jako validation

# Parallel workers dla build_dataset
# Zaszyte 6 = liczba rdzeni VPS. Na RunPodzie (256 rdzeni) budowa datasetu
# szla na 6 workerach — reszta maszyny stala. HAI_N_PARALLEL nadpisuje.
N_PARALLEL_SYMBOLS = int(os.environ.get("HAI_N_PARALLEL", "6"))

FEATURE_NAMES = [
    # Przyciete z 29 do 19 na bazie feature_importance (audyt 2026-07-03) —
    # usuniete: taker_buy_ratio, momentum, volume_ratio, volume_zscore,
    # ema_fast_r, funding_change_24h, oi_total_log, trend_1h, sr_dist_pct,
    # fib_dist_pct (wszystkie ponizej ~0.4 sredniego percentyla waznosci
    # w RF/XGB/CAT). price_position_bb zostaje mimo niskiego wyniku modelu
    # — jest filtrem doktrynalnym (BB_LONG_MAX/BB_SHORT_MIN), nie tylko cecha.
    'rsi', 'rsi_4h', 'rsi_1d',
    'ema_slow_r', 'ema_mid_r',
    'atr_pct',
    'trend_4h', 'trend_1d',
    'funding_rate',
    'oi_change_24h', 'oi_zscore_30d',
    'price_position_bb', 'bb_bandwidth_pct',
    'hour_sin', 'hour_cos', 'day_of_week',
    'adx_14',
    'macd_hist', 'sr_node_strength',
    # Fear&Greed + BTC context testowane 2026-07-04 na wszystkich 5 modelach -
    # pomogly TYLKO RF (ktory ma je osobno w _CORE_FEATURES_RF), HistGB
    # wraca do wersji bez nich (regresja: acc 0.567->0.553, L:prec 0.533->0.454).
]

# === PER-MODEL FEATURE SPECIALIZATION (audyt 2026-07-04) ===
# Core = 12 cech uniwersalnie silnych (argmax-owo prawie zawsze w top dla
# kazdego modelu). Specjalistyczne = cechy, w ktorych DANY model realnie
# === "Strategia cechowania" v2 (audyt 2026-07-05, na podstawie raportu
# uzytkownika + weryfikacji korelacji) === Core zredukowany do 10 cech
# uniwersalnych + kazdy model ma WYRAZNA specjalizacje/"perspektywe"
# (LGB=sentyment/OI, XGB=trend/volume, RF=mean-reversion, CAT_SHARP6X=
# doktryna, LGB_FAST24H=szybki, HISTGB=uniwersalny). DWIE POPRAWKI wzgledem
# oryginalnego raportu (zweryfikowane empirycznie PRZED wdrozeniem):
# 1. XGB mial dostac RAZEM ema_slow_r+ema_fast_r - korelacja 0.917 (!),
#    łamie WLASNA zasade raportu "korelacja<0.7, usun jedna" - zostawiono
#    tylko ema_slow_r (dluzszy termin, pasuje do roli "trend").
# 2. RF mial dostac bb_bandwidth_pct jako "specjalistyczna" - to JUZ jest
#    w core (duplikat), usuniete. BTC+Fear&Greed ZACHOWANE w RF mimo ze
#    raport o tym nie wspomnial - to POTWIERDZONA empirycznie poprawka
#    (audyt 2026-07-04: acc 0.564->0.596), szkoda by bylo ja stracic.
# STARY core (12 cech, z oi_change_24h+sr_node_strength) zachowany ponizej
# jako _CORE_FEATURES_V1 - latwy rollback gdyby v2 wypadl gorzej w WFV.
_CORE_FEATURES_V1 = [
    'oi_change_24h', 'rsi_1d', 'rsi_4h', 'ema_mid_r', 'atr_pct',
    'trend_1d', 'trend_4h', 'sr_node_strength', 'adx_14',
    'day_of_week', 'hour_sin', 'bb_bandwidth_pct',
]
_CORE_FEATURES = [
    'rsi_4h', 'rsi_1d', 'ema_mid_r', 'atr_pct', 'trend_1d', 'trend_4h',
    'adx_14', 'day_of_week', 'hour_sin', 'bb_bandwidth_pct',
]
# Fear&Greed + BTC context (audyt 2026-07-04) - test pelnego retreningu
# pokazal ze pomaga TYLKO RF (acc 0.564->0.596, L:prec 0.577->0.572,
# S:prec 0.332->0.356), a szkodzi LGB/XGB/CAT/HistGB (wszystkie gorsze
# na precision_long). Zostaje wiec WYLACZNIE w zestawie RF, nie w core.
_CORE_FEATURES_RF = _CORE_FEATURES + [
    'fear_greed', 'btc_trend_1h', 'btc_trend_4h', 'btc_trend_1d', 'btc_rsi_4h',
]
# A/B dywergencji jako cecha (env HAI_FEAT_DIV=1). Kolumna 'div_rsi' zawsze
# jest w rekordzie; ten toggle decyduje czy modele NA NIEJ trenuja. Konsultacja
# bytow 2026-07-26 (DeepSeek v4-pro/flash: jednogl. #1). Domyslnie OFF = baseline.
import os as _os_div
if _os_div.getenv("HAI_FEAT_DIV", "0") == "1":
    for _cf in (_CORE_FEATURES, _CORE_FEATURES_RF):
        if "div_rsi" not in _cf:
            _cf.append("div_rsi")
if _os_div.getenv("HAI_FEAT_SD", "0") == "1":
    for _cf in (_CORE_FEATURES, _CORE_FEATURES_RF):
        if "sd_prox" not in _cf:
            _cf.append("sd_prox")
if _os_div.getenv("HAI_FEAT_TAGE", "0") == "1":
    for _cf in (_CORE_FEATURES, _CORE_FEATURES_RF):
        if "bars_cross" not in _cf:
            _cf.append("bars_cross")
MODEL_FEATURES = {
    # LGB - "Analityk sentymentu i OI": 10 core + 8 = 18 cech
    'lgb': _CORE_FEATURES + [
        'oi_change_24h', 'oi_zscore_30d', 'oi_total_log', 'funding_rate',
        'funding_change_24h', 'hour_cos', 'sr_node_strength', 'macd_hist',
        'dist_below_liq', 'dist_above_liq',  # mapa likwidacji (gen.Core, +0.100 wal.)
    ],
    # XGB - "Analityk trendu i volume": 10 core + 7 = 17 cech (usuniety
    # ema_fast_r wzgledem oryginalnego raportu, patrz poprawka #1 wyzej)
    'xgb': _CORE_FEATURES + [
        'ema_slow_r', 'momentum', 'trend_1h', 'volume_ratio',
        'volume_zscore', 'price_position_bb', 'fib_dist_pct',
        'dist_below_liq', 'dist_above_liq',  # mapa likwidacji (gen.Core, +0.100 wal.)
    ],
    # RF - "Analityk mean-reversion": 10 core + BTC/F&G(5) + 5 = 20 cech
    # (usuniety duplikat bb_bandwidth_pct wzgledem raportu, patrz poprawka #2)
    'rf': _CORE_FEATURES_RF + [
        'rsi', 'fib_dist_pct', 'sr_dist_pct', 'taker_buy_ratio', 'sr_node_strength',
        'dist_below_liq', 'dist_above_liq',  # mapa likwidacji (gen.Core, +0.100 wal.)
    ] + (['of_cvd_chg_24h', 'cvd_x_adx'] if os.environ.get("HAI_LEJEK_CVD") == "1" else []),
    # CAT - "Analityk makro/kontekstu globalnego": 10 core + 7 = 17 cech
    # (audyt 2026-07-05 - CAT dotad bez specjalizacji, a byl zidentyfikowany
    # jako "fabryka falszywych sygnalow" - duzo transakcji, slaby PF. Test:
    # czy kontekst makro (Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance) pomaga
    # mu byc bardziej selektywnym. User: "te dane tez tylko szumialy ale
    # dobra daj je" - eksperyment, nie pewnik. USUNIETE 2026-08-06: potwierdzony
    # szum (gold/oil/sp500/vix/us10y/dxy), zero wartosci predykcyjnej - zostaje
    # tylko btc_dominance_chg (odrebny, bardziej bezposredni sygnal).)
    'cat': _CORE_FEATURES + [
        'btc_dominance_chg',
        'dist_below_liq', 'dist_above_liq',  # mapa likwidacji (gen.Core, +0.100 wal.)
    ],
    # HISTGB - "uniwersalny": 10 core + 9 = 19 cech (bylo: pelny FEATURE_NAMES
    # niezmieniony, teraz tez ma wyrazna specjalizacje wg "strategii cechowania" v2)
    'histgb': _CORE_FEATURES + [
        'rsi', 'ema_slow_r', 'funding_rate', 'oi_zscore_30d', 'price_position_bb',
        'hour_cos', 'macd_hist', 'fib_dist_pct', 'sr_dist_pct',
        'dist_below_liq', 'dist_above_liq',  # mapa likwidacji (gen.Core, +0.100 wal.)
    ],
    # NOWE ALGORYTMY (audyt 2026-07-05, na wyrazna prosbe "co jeszcze mamy z
    # drzew do wyboru" + "dolacz do biblioteki wszystkie trzy") - odkrycie z
    # korelacji glosow: LGB/XGB/HistGB/LGB_FAST24H glosuja niemal identycznie
    # (r=0.994-0.997) - potrzeba WIECEJ BAGGING (rodzina RF), nie kolejnego
    # gradient boostingu.
    # ET (Extra Trees) - TA SAMA rola co RF (mean-reversion, bagging), zeby
    # sprawdzic czy inny mechanizm splitow (losowy prog, nie tylko losowa
    # cecha) da GENUINE druga "rodzine bagging" zamiast piatej odmiany GB.
    'et': _CORE_FEATURES_RF + [
        'rsi', 'fib_dist_pct', 'sr_dist_pct', 'taker_buy_ratio', 'sr_node_strength',
    ] + (['of_cvd_chg_24h', 'cvd_x_adx'] if os.environ.get("HAI_LEJEK_CVD") == "1" else []),
    # GB (sklearn GradientBoostingClassifier) - core only na start (neutralny),
    # kolejna odmiana gradient boostingu - ryzyko dolaczenia do juz
    # "zblokowanej" rodziny LGB/XGB/HistGB, do zweryfikowania przez korelacje.
    'gb': _CORE_FEATURES,
    # ADA (AdaBoost) - core only, genuinie INNY mechanizm (wazenie probek po
    # bledzie, nie gradienty) - potencjalnie najbardziej odrebny wybor.
    'ada': _CORE_FEATURES,
    # LONG SPECIALIST (audyt 2026-07-04) - dedykowany booster, nie zastepuje
    # glownego ensemble. Cechy wybrane na podstawie binarnej analizy IS_LONG
    # vs IS_SHORT (osobne RF na pelnych 29 cechach): tylko te, ktore realnie
    # faworyzuja LONG (roznica >0.4pp), bez cech ktore w danych wychodza
    # SHORT-owe (np. rsi_1d, hour_sin, ema_slow_r/mid_r - mimo ze "brzmialy"
    # long-owo w intuicji, dane pokazaly odwrotnie).
    # v1 (tylko 9 "czysto long-owych" cech + waga 4.0) dala KATASTROFALNY
    # wynik: acc=0.38 (ponizej trywialnego baseline ~61% zawsze-NEUTRAL),
    # prec_long SPADLA z 0.536 do 0.256. Zbyt agresywna waga + zbyt waskie
    # cechy = model "trigger-happy" tracacy precyzje na WSZYSTKICH klasach.
    # v2: core-12 (moc dyskryminacyjna 3-klasowa) + tylko NOWE long-owe
    # dodatki (fib_dist_pct/oi_total_log/oi_zscore_30d/volume_zscore -
    # sr_node_strength/atr_pct/adx_14/bb_bandwidth_pct/trend_4h juz w core,
    # bez duplikatow), waga zlagodzona do 3.0 (audyt 2026-07-04).
    'long_spec': _CORE_FEATURES + [
        'fib_dist_pct', 'oi_total_log', 'oi_zscore_30d', 'volume_zscore',
    ],
    # DWA SPECJALISCI (audyt 2026-07-04/05, ten sam algorytm co glowny
    # model, INNY label + WLASNA specjalizacja cech wg "strategii cechowania" v2).
    # LGB_FAST24H - "szybki horyzont": 10 core + 7 = 17 cech.
    'lgb_fast24h': _CORE_FEATURES + [
        'oi_change_24h', 'momentum', 'trend_1h', 'volume_ratio',
        'rsi', 'price_position_bb', 'funding_rate',
    ],
    # CAT_SHARP6X - "doktryna wzmocniona": 10 core + 6 = 16 cech.
    'cat_sharp6x': _CORE_FEATURES + [
        'sr_node_strength', 'funding_rate', 'oi_zscore_30d',
        'ema_slow_r', 'rsi', 'price_position_bb',
    ],
    # Test systematyczny (audyt 2026-07-04) - czy zmiana horyzontu/progu
    # pomaga TEZ pozostalym 3 algorytmom (RF/XGB/HistGB), nie tylko LGB/CAT.
    # TE SAME cechy co rodzic, tylko inny label.
    'rf_fast24h':     _CORE_FEATURES_RF + ['rsi', 'fib_dist_pct', 'sr_dist_pct', 'taker_buy_ratio', 'sr_node_strength']
                     + (['of_cvd_chg_24h', 'cvd_x_adx'] if os.environ.get("HAI_LEJEK_CVD") == "1" else []),
    'rf_sharp6x':     _CORE_FEATURES_RF + ['rsi', 'fib_dist_pct', 'sr_dist_pct', 'taker_buy_ratio', 'sr_node_strength'],
    'xgb_fast24h':    _CORE_FEATURES + ['ema_slow_r', 'momentum', 'trend_1h', 'volume_ratio', 'volume_zscore', 'price_position_bb', 'fib_dist_pct'],
    'xgb_sharp6x':    _CORE_FEATURES + ['ema_slow_r', 'momentum', 'trend_1h', 'volume_ratio', 'volume_zscore', 'price_position_bb', 'fib_dist_pct'],
    'histgb_fast24h': _CORE_FEATURES + ['rsi', 'ema_slow_r', 'funding_rate', 'oi_zscore_30d', 'price_position_bb', 'hour_cos', 'macd_hist', 'fib_dist_pct', 'sr_dist_pct'],
    'histgb_sharp6x': _CORE_FEATURES + ['rsi', 'ema_slow_r', 'funding_rate', 'oi_zscore_30d', 'price_position_bb', 'hour_cos', 'macd_hist', 'fib_dist_pct', 'sr_dist_pct'],
}
# Ktory model uzywa jakiego labelu (domyslnie label_long jesli brak wpisu)
MODEL_LABEL_COLUMN = {
    'lgb_fast24h': 'label_fast24h',  # 24h lookahead zamiast 48h - szybsze ruchy
    'cat_sharp6x': 'label_sharp6x',  # TP=6xATR zamiast 4x - tylko zdecydowane ruchy
    'rf_fast24h': 'label_fast24h', 'rf_sharp6x': 'label_sharp6x',
    'xgb_fast24h': 'label_fast24h', 'xgb_sharp6x': 'label_sharp6x',
    'histgb_fast24h': 'label_fast24h', 'histgb_sharp6x': 'label_sharp6x',
}
# Skrajnosci horyzontu - fast6h (dolny praktyczny limit) i wide96h (gorny
# praktyczny limit) na WSZYSTKICH 5 algorytmach (audyt 2026-07-04, na
# wyrazna prosbe "ile w dol/gore mozemy zejsc"). TE SAME cechy co rodzic,
# tylko inny label - generyczna petla (patrz train_models) zamiast kopiowania
# kolejnych prawie-identycznych blokow treningowych.
for _algo in ('lgb', 'rf', 'xgb', 'cat', 'histgb'):
    MODEL_FEATURES[f'{_algo}_fast6h'] = MODEL_FEATURES[_algo]
    MODEL_FEATURES[f'{_algo}_wide96h'] = MODEL_FEATURES[_algo]
    MODEL_FEATURES[f'{_algo}_h72'] = MODEL_FEATURES[_algo]
    MODEL_LABEL_COLUMN[f'{_algo}_fast6h'] = 'label_fast6h'
    MODEL_LABEL_COLUMN[f'{_algo}_wide96h'] = 'label_wide96h'
    MODEL_LABEL_COLUMN[f'{_algo}_h72'] = 'label_h72'
HORIZON_SWEEP_NAMES = [f'{a}_fast6h' for a in ('lgb', 'rf', 'xgb', 'cat', 'histgb')] + \
                      [f'{a}_wide96h' for a in ('lgb', 'rf', 'xgb', 'cat', 'histgb')]

# Rejestr wyborow horyzontu do UI "Trening niestandardowy" (audyt 2026-07-05)
# - human-readable etykieta -> sufiks nazwy modelu ('' = glowny label_long).
HORIZON_CHOICES = {
    'main':    {'label': 'Główny (48h, TP=4×ATR)', 'suffix': ''},
    'fast24h': {'label': '24h lookahead',           'suffix': '_fast24h'},
    'sharp6x': {'label': '48h, TP=6×ATR',            'suffix': '_sharp6x'},
    'fast6h':  {'label': '6h lookahead',             'suffix': '_fast6h'},
    'h72':     {'label': '72h lookahead',            'suffix': '_h72'},
    'wide96h': {'label': '96h lookahead',            'suffix': '_wide96h'},
}
# Waga klasy LONG dla long_spec - v1 (4.0, tylko 9 cech) dala katastrofalny
# wynik (acc=0.38, prec_long spadla do 0.256) - zlagodzone do 3.0 + wiecej cech
_CW3_LONG_SPEC = {0: 1.0, 1: 3.0, 2: 2.5}


def calc_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI Wilder's smoothing."""
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 50
    rsi[period] = 100 - 100 / (1 + rs) if down != 0 else 50

    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 100
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi


def calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """EMA - exponential moving average."""
    k = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR - Average True Range."""
    tr = np.zeros_like(closes)
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.zeros_like(closes)
    atr[:period] = tr[:period].mean()
    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_divergence(closes: np.ndarray, rsi_arr: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Regularna dywergencja cena vs RSI w oknie `lookback` (jedna liczba/bar).
    +1 = byczy rozjazd (cena nizej, RSI wyzej -> wyczerpanie spadku),
    -1 = niedzwiedzi (cena wyzej, RSI nizej), 0 = brak. Wektorowo, zgodne 1:1
    z build_features_live (parytet train/inference). Konsultacja bytow 2026-07-26."""
    n = len(closes)
    div = np.zeros(n, dtype=np.float64)
    if n <= lookback:
        return div
    dprice = closes[lookback:] - closes[:-lookback]
    drsi = rsi_arr[lookback:] - rsi_arr[:-lookback]
    bull = (dprice < 0) & (drsi > 0)
    bear = (dprice > 0) & (drsi < 0)
    div[lookback:] = bull.astype(np.float64) - bear.astype(np.float64)
    return div


def _ffill_nan(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN w 1D (ostatnia znana wartosc)."""
    mask = ~np.isnan(a)
    idx = np.where(mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def calc_sd_proximity(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      atr_arr: np.ndarray, window: int = 50) -> np.ndarray:
    """Proximity do stref popytu/podazy (Supply&Demand), jedna liczba/bar.
    Strefa POPYTU = swing-low (min z okna) na barze wzrostowym (close>close[-1]).
    Strefa PODAZY = swing-high (max z okna) na barze spadkowym. Poziomy ffill-owane.
    sd = (close-demand)/atr - (supply-close)/atr:  +  = blisko PODAZY (kontrarianski
    short), -  = blisko POPYTU (kontrarianski long). Reversal z samych closes ->
    parytet 1:1 z build_features_live (bez potrzeby opens). Konsultacja bytow 2026-07-26."""
    n = len(closes)
    sd = np.zeros(n, dtype=np.float64)
    if n < window + 2:
        return sd
    up = np.zeros(n, dtype=bool);   up[1:] = closes[1:] > closes[:-1]
    down = np.zeros(n, dtype=bool); down[1:] = closes[1:] < closes[:-1]
    demand = np.full(n, np.nan); supply = np.full(n, np.nan)
    for i in range(window - 1, n):
        if up[i] and lows[i] == lows[i - window + 1:i + 1].min():
            demand[i] = lows[i]
        if down[i] and highs[i] == highs[i - window + 1:i + 1].max():
            supply[i] = highs[i]
    demand = _ffill_nan(demand); supply = _ffill_nan(supply)
    atr_safe = np.where(atr_arr > 0, atr_arr, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = (closes - demand) / atr_safe - (supply - closes) / atr_safe
    val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(val, -15.0, 15.0)


def calc_bars_since_cross(closes: np.ndarray, fast: int = 20, slow: int = 50, cap: int = 100) -> np.ndarray:
    """Wiek trendu: ile barow od ostatniego przeciecia EMA20/EMA50, ze znakiem
    kierunku (+ = uptrend/EMA20 nad EMA50, - = downtrend). 0 na barze przeciecia.
    Kontrarianski sens: stary trend = kandydat do fade. Insight DeepSeek v4-pro.
    Parytet 1:1 z build_features_live (ta sama calc_ema/_ema_series_live)."""
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    if n < slow + 2:
        return out
    ef = calc_ema(closes, fast)
    es = calc_ema(closes, slow)
    spread = ef - es
    sg = np.sign(spread)
    bars = 0
    for i in range(1, n):
        if sg[i] != sg[i - 1] and sg[i] != 0:
            bars = 0
        else:
            bars += 1
        out[i] = bars * (1.0 if spread[i] > 0 else (-1.0 if spread[i] < 0 else 0.0))
    return np.clip(out, -cap, cap)


def detect_trend_scalar(prices: np.ndarray, fast: int = 9, slow: int = 21) -> int:
    """1=UP, -1=DOWN, 0=SIDEWAYS - na podstawie EMA."""
    if len(prices) < slow:
        return 0
    ema_f = calc_ema(prices[-slow:], fast)[-1]
    ema_s = calc_ema(prices[-slow:], slow)[-1]
    diff_pct = (ema_f - ema_s) / ema_s * 100
    if diff_pct > 0.3:
        return 1
    elif diff_pct < -0.3:
        return -1
    return 0


def precompute_trend_series(closes: np.ndarray, fast: int = 9, slow: int = 21) -> np.ndarray:
    """NEW v2.0: Pre-compute trend dla calego szeregu, vectorized."""
    if len(closes) < slow:
        return np.zeros(len(closes), dtype=np.int8)
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    diff_pct = np.where(ema_s != 0, (ema_f - ema_s) / ema_s * 100, 0)
    trend = np.zeros(len(closes), dtype=np.int8)
    trend[diff_pct > 0.3] = 1
    trend[diff_pct < -0.3] = -1
    return trend


def calc_macd_hist(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram (linia MACD - linia sygnalu), znormalizowany do % ceny."""
    ema_f = calc_ema(prices, fast)
    ema_s = calc_ema(prices, slow)
    macd_line = ema_f - ema_s
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return np.where(prices > 0, hist / prices * 100, 0.0)


def calc_volume_zscore(volumes: np.ndarray, period: int = 30) -> np.ndarray:
    """Z-score wolumenu wzgledem rolling mean/std — inna informacja niz volume_ratio (mean-only)."""
    vs = pd.Series(volumes)
    rmean = vs.rolling(period, min_periods=1).mean()
    rstd = vs.rolling(period, min_periods=1).std().replace(0, 1.0).fillna(1.0)
    return ((vs - rmean) / rstd).fillna(0.0).values


def calc_swing_sr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                   lookback: int = 5, tolerance: float = 0.003,
                   strength_window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """PhantomFlow (reaktywacja) — uproszczona, wektoryzowana wersja.

    Swing high/low: rolling max/min z centrowanym oknem (2*lookback+1) —
    O(n) bez zagniezdzonej petli. WAZNE (no-lookahead): centrowane okno
    "wie" ze idx to swing dopiero majac dane z idx+lookback — punkt jest
    wiec oznaczany jako znany DOPIERO od confirm_idx=idx+lookback, nie od
    idx. Sila wezla (liczba dotkniec) liczona tez tylko z danych <= confirm_idx,
    nigdy z przyszlosci wzgledem momentu w ktorym feature staje sie widoczny.
    Zwraca (sr_dist_pct, sr_node_strength) forward-filled na kazdy wiersz:
    dodatnie = blisko wsparcia, ujemne = blisko oporu.
    """
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
        confirm_idx = idx + lookback  # dopiero teraz wiadomo ze idx byl swing high
        if confirm_idx >= n:
            continue
        level = highs[idx]
        band = level * tolerance
        w0, w1 = max(0, idx - strength_window), confirm_idx + 1  # zero danych z przyszlosci
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


def calc_fib_dist(closes: np.ndarray, sh_level_ff: np.ndarray, sl_level_ff: np.ndarray) -> np.ndarray:
    """Odleglosc% do najblizszego poziomu zniesienia Fibonacciego (23.6/38.2/50/61.8/78.6%)
    liczonego miedzy ostatnim znanym swing high i swing low (impuls). Reuzywa poziomy
    z calc_swing_sr — brak dodatkowego kosztu obliczeniowego.
    """
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


def _eq_hl_window_count(window: np.ndarray, tolerance: float = 0.0015) -> float:
    """Pomocnicza do rolling().apply: liczba prawie-rownych par w oknie (O(window^2),
    window mala stala=20 wiec akceptowalne w apply)."""
    m = len(window)
    cnt = 0
    for ii in range(m):
        for jj in range(ii + 1, m):
            if abs(window[ii] - window[jj]) <= tolerance * max(abs(window[ii]), 1e-8):
                cnt += 1
    return float(cnt)


def _ob_strength_window(window: np.ndarray) -> float:
    """Pomocnicza do rolling().apply: sila+wiek najwiekszego impulsu w oknie closes."""
    rets = np.diff(window) / window[:-1]
    if len(rets) == 0:
        return 0.0
    idx = int(np.argmax(np.abs(rets)))
    age = len(rets) - idx
    return float(abs(rets[idx]) * 100 / max(age, 1))


def calc_rptr_features(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                        volumes: np.ndarray, rsi_arr: np.ndarray, atr_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Wektoryzowane RF/ET specjalistyczne cechy wg rptr (external feature-eng
    review, 2026-08-05) - parytet formul z hai_common.features (live). Trailing
    rolling (min_periods=window, bez center=True poza structure_bias ktore uzywa
    tego samego confirm_idx no-lookahead patternu co calc_swing_sr). basis_zscore
    i funding_zscore_8h/24h pominiete (brak danych/historii - jak w wersji live)."""
    n = len(closes)
    c = pd.Series(closes); h = pd.Series(highs); l = pd.Series(lows); v = pd.Series(volumes)
    rsi_s = pd.Series(rsi_arr); atr_s = pd.Series(atr_arr)
    out: Dict[str, np.ndarray] = {}

    # --- RF specialist ---
    r_min = rsi_s.rolling(14, min_periods=14).min()
    r_max = rsi_s.rolling(14, min_periods=14).max()
    out['r_stoch_rsi'] = ((rsi_s - r_min) / (r_max - r_min).replace(0, np.nan) * 100).fillna(50.0).values

    tp = (h + l + c) / 3.0
    sma20 = tp.rolling(20, min_periods=20).mean()
    mad20 = (tp - sma20).abs().rolling(20, min_periods=20).mean()
    out['r_cci_20'] = ((tp - sma20) / (0.015 * mad20.replace(0, np.nan))).fillna(0.0).values

    up_move = h.diff()
    down_move = -l.diff()
    dm_plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    dm_minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = dm_plus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w.replace(0, np.nan) * 100
    mdi = dm_minus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w.replace(0, np.nan) * 100
    out['r_di_spread'] = (pdi - mdi).fillna(0.0).values

    ema12 = calc_ema(closes, 12); ema26 = calc_ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    out['r_macd_signal_dist'] = np.where(closes > 0, (macd_line - signal_line) / closes * 100, 0.0)

    out['r_momentum_20'] = ((c / c.shift(20) - 1) * 100).fillna(0.0).values

    atr_mean20 = atr_s.rolling(20, min_periods=20).mean()
    atr_std20 = atr_s.rolling(20, min_periods=20).std().replace(0, np.nan)
    out['r_atr_zscore_20'] = ((atr_s - atr_mean20) / atr_std20).fillna(0.0).values

    v_mean50 = v.rolling(50, min_periods=50).mean()
    v_std50 = v.rolling(50, min_periods=50).std().replace(0, np.nan)
    out['r_volume_zscore_50'] = ((v - v_mean50) / v_std50).fillna(0.0).values

    short_range = h.rolling(6, min_periods=6).max() - l.rolling(6, min_periods=6).min()
    long_range = (h.rolling(24, min_periods=24).max() - l.rolling(24, min_periods=24).min()).replace(0, np.nan)
    out['r_range_compression'] = (short_range / long_range).fillna(1.0).values

    tvol24 = v.rolling(24, min_periods=24).sum().replace(0, np.nan)
    vwap24 = (tp * v).rolling(24, min_periods=24).sum() / tvol24
    out['r_vwap_distance'] = ((c / vwap24.replace(0, np.nan) - 1) * 100).fillna(0.0).values

    # structure_bias: HH/HL(1) vs LH/LL(-1) na potwierdzonych swingach, sam
    # no-lookahead confirm_idx pattern co calc_swing_sr (idx+lookback zanim znany).
    lookback = 5
    win = 2 * lookback + 1
    roll_max = h.rolling(win, center=True, min_periods=win).max()
    roll_min = l.rolling(win, center=True, min_periods=win).min()
    is_sh = (h == roll_max).fillna(False).values
    is_sl = (l == roll_min).fillna(False).values
    sh_seq = sorted((idx + lookback, highs[idx]) for idx in np.where(is_sh)[0] if idx + lookback < n)
    sl_seq = sorted((idx + lookback, lows[idx]) for idx in np.where(is_sl)[0] if idx + lookback < n)
    structure_bias = np.zeros(n)
    si = li = 0
    ph1 = ph2 = pl1 = pl2 = None
    cur_bias = 0.0
    for idx in range(n):
        while si < len(sh_seq) and sh_seq[si][0] == idx:
            ph2 = ph1; ph1 = sh_seq[si][1]; si += 1
        while li < len(sl_seq) and sl_seq[li][0] == idx:
            pl2 = pl1; pl1 = sl_seq[li][1]; li += 1
        if ph1 is not None and ph2 is not None and pl1 is not None and pl2 is not None:
            hh = ph1 > ph2; hl = pl1 > pl2
            cur_bias = 1.0 if (hh and hl) else (-1.0 if (not hh and not hl) else 0.0)
        structure_bias[idx] = cur_bias
    out['r_structure_bias'] = structure_bias

    # --- ET specialist (SMC + Ichimoku) ---
    prior_high20 = h.shift(1).rolling(20, min_periods=20).max()
    prior_low20 = l.shift(1).rolling(20, min_periods=20).min()
    out['e_liquidity_sweep_high'] = ((h > prior_high20) & (c < prior_high20)).fillna(False).astype(float).values
    out['e_liquidity_sweep_low'] = ((l < prior_low20) & (c > prior_low20)).fillna(False).astype(float).values

    h_2 = h.shift(2); l_2 = l.shift(2)
    bull_gap = l - h_2
    bear_gap = l_2 - h
    fvg_size = np.where(bull_gap > 0, bull_gap / c.replace(0, np.nan) * 100,
                np.where(bear_gap > 0, bear_gap / c.replace(0, np.nan) * 100, 0.0))
    fvg_dir = np.where(bull_gap > 0, 1.0, np.where(bear_gap > 0, -1.0, 0.0))
    out['e_fvg_size'] = np.nan_to_num(fvg_size)
    out['e_fvg_direction'] = np.nan_to_num(fvg_dir)

    cur_h10 = h.rolling(10, min_periods=10).max()
    cur_l10 = l.rolling(10, min_periods=10).min()
    prior_h10 = cur_h10.shift(10); prior_l10 = cur_l10.shift(10)
    bos = np.where(cur_h10 > prior_h10, 1.0, np.where(cur_l10 < prior_l10, -1.0, 0.0))
    out['e_bos_choch'] = np.nan_to_num(bos)

    eq_h = h.rolling(20, min_periods=20).apply(_eq_hl_window_count, raw=True)
    eq_l = l.rolling(20, min_periods=20).apply(_eq_hl_window_count, raw=True)
    out['e_equal_highs_lows_count'] = (eq_h.fillna(0) + eq_l.fillna(0)).values

    tenkan = (h.rolling(9, min_periods=9).max() + l.rolling(9, min_periods=9).min()) / 2
    kijun = (h.rolling(26, min_periods=26).max() + l.rolling(26, min_periods=26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (h.rolling(52, min_periods=52).max() + l.rolling(52, min_periods=52).min()) / 2
    thickness = (senkou_a - senkou_b).abs() / c.replace(0, np.nan) * 100
    tk_cross = np.where(tenkan > kijun, 1.0, np.where(tenkan < kijun, -1.0, 0.0))
    out['e_ichimoku_cloud_thickness'] = thickness.fillna(0.0).values
    out['e_tk_cross'] = np.nan_to_num(tk_cross)

    rets_sign = np.sign(c.diff())
    signed_vol = v * rets_sign
    net20 = signed_vol.rolling(20, min_periods=20).sum()
    total20 = v.rolling(20, min_periods=20).sum().replace(0, np.nan)
    out['e_volume_delta_imbalance'] = (net20 / total20).fillna(0.0).values

    ob = c.rolling(21, min_periods=21).apply(_ob_strength_window, raw=True)
    out['e_orderblock_strength'] = ob.fillna(0.0).values

    # x_obv_slope_24h (dormant x_* z features.py, 2026-08-05 - parytet z live:
    # ten sam wzor co e_volume_delta_imbalance, tylko okno 24 zamiast 20)
    net24 = signed_vol.rolling(24, min_periods=24).sum()
    total24 = v.rolling(24, min_periods=24).sum().replace(0, np.nan)
    out['x_obv_slope_24h'] = (net24 / total24).fillna(0.0).values

    return out


def calc_x_features(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                    volumes: np.ndarray) -> Dict[str, np.ndarray]:
    """Wektoryzowane EKSPERYMENTALNE cechy x_* (parytet z hai_common.features,
    audyt 2026-08-05) + vwap_dev. Do cache WFV, zeby configi divcore (i inne)
    mogly ich uzywac przez feature_mix. Trailing rolling, min_periods=window."""
    n = len(closes)
    c = pd.Series(closes); h = pd.Series(highs); l = pd.Series(lows); v = pd.Series(volumes)
    out: Dict[str, np.ndarray] = {}

    rets = c.pct_change()
    out['x_real_vol_6h'] = rets.rolling(6, min_periods=6).std().fillna(0.0).values * 100
    out['x_real_vol_24h'] = rets.rolling(24, min_periods=24).std().fillna(0.0).values * 100

    hl = (h / l).clip(lower=1e-9).apply(np.log)
    out['x_parkinson_24h'] = (np.sqrt((1.0 / (4 * np.log(2))) * (hl**2).rolling(25, min_periods=25).mean())).fillna(0.0).values * 100

    out['x_rsi_7'] = calc_rsi(closes, 7)
    out['x_rsi_21'] = calc_rsi(closes, 21)

    for _bars in (4, 24, 168):
        out[f'x_roc_{_bars}'] = ((c / c.shift(_bars) - 1) * 100).fillna(0.0).values

    for _f, _s in ((9, 21), (21, 55), (50, 200)):
        _ef = calc_ema(closes, _f); _es = calc_ema(closes, _s)
        out[f'x_ema{_f}_over_{_s}'] = np.where(_es > 0, _ef / np.where(_es == 0, np.nan, _es) - 1, 0.0)

    tp = (h + l + c) / 3.0
    tvol = v.rolling(24, min_periods=24).sum()
    vwap = (tp * v).rolling(24, min_periods=24).sum() / tvol.replace(0, np.nan)
    out['vwap_dev'] = np.where(vwap.notna() & (vwap > 0), (c / vwap - 1) * 100, 0.0)

    return out


def triple_barrier_label(cur: float, atr: float, highs: np.ndarray, lows: np.ndarray,
                          i: int, n: int, lookahead: int, tp_mult: float, sl_mult: float) -> int:
    """Triple Barrier 3-klasowy (0=NEUTRAL,1=LONG,2=SHORT), parametryzowany po
    lookahead/TP/SL (audyt 2026-07-04) - pozwala liczyc RAZEM z glownym labelem
    dodatkowe warianty (np. krotszy lookahead, wyzszy TP) bez powtarzania calego
    liczenia cech. LONG i SHORT sledzone jako dwie niezalezne symetryczne
    "transakcje" w tym samym oknie (patrz komentarz w build_features_for_symbol)."""
    tp_long  = cur + tp_mult * atr
    sl_long  = cur - sl_mult * atr
    tp_short = cur - tp_mult * atr
    sl_short = cur + sl_mult * atr

    long_win = short_win = False
    long_done = short_done = False
    first_hit = None
    for j in range(i + 1, min(i + 1 + lookahead, n)):
        if not long_done:
            if highs[j] >= tp_long:
                long_win = True; long_done = True
                if first_hit is None: first_hit = 'long'
            elif lows[j] <= sl_long:
                long_done = True
        if not short_done:
            if lows[j] <= tp_short:
                short_win = True; short_done = True
                if first_hit is None: first_hit = 'short'
            elif highs[j] >= sl_short:
                short_done = True
        if long_done and short_done:
            break

    if long_win and short_win:
        return 1 if first_hit == 'long' else 2
    elif long_win:
        return 1
    elif short_win:
        return 2
    return 0


def calc_adx_closes_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX proxy z samych close dla całej serii (rolling window 50)."""
    n = len(closes)
    result = np.zeros(n, dtype=np.float32)
    min_len = period * 2 + 2
    for i in range(min_len, n):
        window = closes[max(0, i - 50):i + 1].tolist()
        if len(window) < min_len:
            continue
        dm_plus, dm_minus, trs = [], [], []
        for k in range(1, len(window)):
            diff = window[k] - window[k - 1]
            dm_plus.append(max(diff, 0.0))
            dm_minus.append(max(-diff, 0.0))
            trs.append(abs(diff))
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
        result[i] = round(sum(dx_vals[-period:]) / period, 1) if dx_vals else 0.0
    return result


_FEAR_GREED_CACHE = None


def _load_fear_greed() -> Optional[pd.DataFrame]:
    """Fear & Greed Index - dane globalne (nie per-symbol), ladowane raz i
    cache'owane w module (audyt 2026-07-04, pelna historia 2018-dzis)."""
    global _FEAR_GREED_CACHE
    if _FEAR_GREED_CACHE is not None:
        return _FEAR_GREED_CACHE
    try:
        p = WH_BASE.parent.parent / 'macro' / 'fear_greed.parquet'
        df = pd.read_parquet(p)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        _FEAR_GREED_CACHE = df
        return df
    except Exception as e:
        logger.warning(f'Fear&Greed load error: {e}')
        _FEAR_GREED_CACHE = pd.DataFrame()
        return _FEAR_GREED_CACHE


_MACRO_EXT_CACHE = None
_MACRO_EXT_TICKERS = ['gold', 'oil_wti', 'sp500', 'vix', 'us10y_yield', 'dxy', 'btc_dominance']


def _load_macro_extended() -> Dict:
    """7 dodatkowych serii makro (audyt 2026-07-05, na wyrazna prosbe -
    'te dane tez tylko szumialy ale dobra daj je'): Gold/Oil WTI/S&P500/VIX/
    US10Y (yfinance, OHLC dzienne, kolumna 'close') + DXY/BTC dominance
    (CoinGecko/yfinance, kolumna 'value'). Ladowane raz, cache'owane w module
    - jak _load_fear_greed()/_load_btc_context(). Kazda seria daje 1 cecha:
    zmiana % dzien-do-dnia (nie poziom absolutny - poziom gold w USD
    nie niesie wprost sygnalu bez kontekstu, zmiana % nadaje sie do
    porownania z funding_change_24h ktore juz istnieje)."""
    global _MACRO_EXT_CACHE
    if _MACRO_EXT_CACHE is not None:
        return _MACRO_EXT_CACHE
    out = {}
    macro_dir = WH_BASE.parent.parent / 'macro'
    for name in _MACRO_EXT_TICKERS:
        try:
            df = pd.read_parquet(macro_dir / f'{name}.parquet')
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            df = df.sort_values('timestamp').reset_index(drop=True)
            val_col = 'close' if 'close' in df.columns else 'value'
            out[f'{name}_times'] = df['timestamp'].values
            out[f'{name}_vals'] = df[val_col].values.astype(np.float64)
        except Exception as e:
            logger.warning(f'macro_extended {name} load error: {e}')
            out[f'{name}_times'] = None
            out[f'{name}_vals'] = None
    _MACRO_EXT_CACHE = out
    return out


_BTC_CONTEXT_CACHE = None


def _load_btc_context() -> Optional[Dict]:
    """BTC trend/rsi jako kontekst dla WSZYSTKICH symboli (audyt 2026-07-04) -
    altcoiny sa silnie skorelowane z BTC (0.6-0.85, beta 1.1-1.8 zweryfikowane
    empirycznie), ale kazdy symbol dotad widzial tylko swoje wlasne dane,
    slepy na to co robi lider rynku. Ladowane raz, cache'owane w module."""
    global _BTC_CONTEXT_CACHE
    if _BTC_CONTEXT_CACHE is not None:
        return _BTC_CONTEXT_CACHE
    try:
        out = {}
        for tf in ['1h', '4h', '1d']:
            p = WH_BASE / tf / 'BTC.parquet'
            df = pd.read_parquet(p)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            closes = df['close'].values.astype(np.float64)
            times = df['timestamp'].values
            if tf == '1h':
                trend = precompute_trend_series(closes, EMA_FAST, EMA_SLOW)
                out['1h_times'] = times
                out['1h_trend'] = trend
            else:
                if len(closes) >= RSI_PERIOD + 1:
                    rsi = calc_rsi(closes, RSI_PERIOD)
                else:
                    rsi = np.full(len(closes), 50.0)
                trend = precompute_trend_series(closes, EMA_FAST, EMA_SLOW)
                out[f'{tf}_times'] = times
                out[f'{tf}_rsi'] = rsi
                out[f'{tf}_trend'] = trend
        _BTC_CONTEXT_CACHE = out
        return out
    except Exception as e:
        logger.warning(f'BTC context load error: {e}')
        _BTC_CONTEXT_CACHE = {}
        return _BTC_CONTEXT_CACHE


def load_symbol_data(symbol: str) -> Optional[Dict]:
    """Zwraca dict z 1h/4h/1d DataFrames + funding dla symbolu."""
    try:
        out = {}
        for tf in ['1h', '4h', '1d']:
            p = WH_BASE / tf / f'{symbol}.parquet'
            if not p.exists():
                logger.warning(f'{symbol} {tf}: brak pliku')
                return None
            df = pd.read_parquet(p)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            out[tf] = df

        # Funding (opcjonalne - jak brak, dajemy zera)
        # FIX 30maja: sciezka derivatives/funding_rates + kolumna close->funding_rate
        fp = WH_BASE.parent.parent / 'derivatives' / 'funding_rates' / f'{symbol}.parquet'
        if fp.exists():
            f_df = pd.read_parquet(fp)
            f_df['timestamp'] = pd.to_datetime(f_df['timestamp'])
            if 'funding_rate' not in f_df.columns and 'close' in f_df.columns:
                f_df['funding_rate'] = f_df['close']
            f_df = f_df.sort_values('timestamp').reset_index(drop=True)
            out['funding'] = f_df
        else:
            out['funding'] = None

        # OI (open_interest) - analogicznie do funding
        oip = WH_BASE.parent.parent / 'derivatives' / 'open_interest' / f'{symbol}.parquet'
        if oip.exists():
            oi_df = pd.read_parquet(oip)
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
            oi_df = oi_df.sort_values('timestamp').reset_index(drop=True)
            # oi_total_log, oi_change_24h, oi_zscore_30d (jak LAB)
            import numpy as _np
            oi_close = oi_df['close'] if 'close' in oi_df.columns else oi_df.iloc[:, -1]
            oi_df['oi_total_log'] = _np.log1p(oi_close.clip(lower=0))
            oi_df['oi_change_24h'] = oi_close.pct_change(1).fillna(0.0)
            rmean = oi_close.rolling(30, min_periods=1).mean()
            rstd = oi_close.rolling(30, min_periods=1).std().replace(0, 1.0).fillna(1.0)
            oi_df['oi_zscore_30d'] = ((oi_close - rmean) / rstd).fillna(0.0)
            out['oi'] = oi_df
        else:
            out['oi'] = None

        # Taker buy ratio (agresorzy kupno/sprzedaz) - juz uzywane w backtesterze,
        # dotad nigdy niepodlaczone do treningu drzew
        trp = WH_BASE.parent.parent / 'derivatives' / 'taker_ratio' / f'{symbol}.parquet'
        if trp.exists():
            tr_df = pd.read_parquet(trp)
            tr_df['timestamp'] = pd.to_datetime(tr_df['timestamp'])
            tr_df = tr_df.sort_values('timestamp').reset_index(drop=True)
            out['taker'] = tr_df
        else:
            out['taker'] = None

        # Long/Short ratio (B2 gen.Dir-v1, 2026-07-19) - lezal odlogiem w
        # magazynie od 2026-07-07 (kolektor cron, 102 symbole, 1h), nigdy
        # nie byl cecha zadnego modelu. Kontrarianskie ekstrema pozycjonowania.
        lsp = WH_BASE.parent.parent / 'derivatives' / 'ls_ratio' / f'{symbol}.parquet'
        if lsp.exists():
            ls_df = pd.read_parquet(lsp).dropna()
            ls_df['timestamp'] = pd.to_datetime(ls_df['timestamp'])
            ls_df = ls_df.sort_values('timestamp').reset_index(drop=True)
            out['ls'] = ls_df
        else:
            out['ls'] = None

        # KORELACJA Z BTC (2026-08-21) — per symbol, godzinowa, liczona wstecznie
        # przez data_warehouse/licz_korelacje_btc.py. Magazyn mial cechy opisujace
        # SAM BTC (btc_trend_*, btc_rsi_4h), ale ani jednej opisujacej ZWIAZEK
        # alta z BTC. Brak pliku = cechy neutralne (0.0), nie blad — nie kazdy
        # symbol musi je miec, a BTC nie ma ich z definicji.
        try:
            _kp = WH_BASE.parent.parent / 'korelacje_btc' / f'{symbol}.parquet'
            if _kp.exists():
                _kdf = pd.read_parquet(_kp)
                _kdf['timestamp'] = pd.to_datetime(_kdf['timestamp'])
                out['korelacje'] = _kdf.sort_values('timestamp').reset_index(drop=True)
            else:
                out['korelacje'] = None
        except Exception as _ke:
            logger.warning(f'{symbol}: korelacje_btc nie wczytane ({_ke}) — cechy beda 0.0')
            out['korelacje'] = None

        # CECHY CVD (2026-08-24) — order flow, znormalizowany PER SYMBOL.
        # Magazyn mial te dane od zawsze (orderflow/binance/, 128 symboli,
        # godzinowe), ale ml_trainer nigdy po nie nie siegal: cvd_x_adx bylo
        # przypisane ZEREM na sztywno. Normalizacja jest tu kluczowa — surowe
        # of_cvd_chg_24h daje 0.527 na samym BTC, ale tylko 0.017 na calym
        # uniwersum, bo CVD jest w jednostkach bezwzglednych i skale sie nie
        # zgadzaja miedzy coinami. Po z-score: 0.351.
        try:
            _cp = WH_BASE.parent.parent / 'cechy_cvd' / f'{symbol}.parquet'
            if _cp.exists():
                _cdf = pd.read_parquet(_cp)
                _cdf['timestamp'] = pd.to_datetime(_cdf['timestamp'])
                out['cvd'] = _cdf.sort_values('timestamp').reset_index(drop=True)
            else:
                out['cvd'] = None
        except Exception as _ce:
            logger.warning(f'{symbol}: cechy_cvd nie wczytane ({_ce}) — beda neutralne')
            out['cvd'] = None

        # CECHY PRZEKROJOWE (2026-08-23) — zdarzenia i pozycja na tle rynku,
        # zamiast rolling korelacji (ta okazala sie bezuzyteczna, patrz
        # raporty/korebtc.txt i kampania P5). Liczone przez
        # data_warehouse/licz_cechy_przekrojowe.py. Brak pliku = wartosci
        # neutralne, nie blad.
        try:
            _pp = WH_BASE.parent.parent / 'cechy_przekrojowe' / f'{symbol}.parquet'
            if _pp.exists():
                _pdf = pd.read_parquet(_pp)
                _pdf['timestamp'] = pd.to_datetime(_pdf['timestamp'])
                out['przekrojowe'] = _pdf.sort_values('timestamp').reset_index(drop=True)
            else:
                out['przekrojowe'] = None
        except Exception as _pe:
            logger.warning(f'{symbol}: cechy_przekrojowe nie wczytane ({_pe}) — beda neutralne')
            out['przekrojowe'] = None

        # Fear & Greed Index (globalny, cache'owany raz - audyt 2026-07-04)
        out['fear_greed'] = _load_fear_greed()
        # BTC context (globalny, cache'owany raz - audyt 2026-07-04)
        out['btc_context'] = _load_btc_context()
        # Macro extended: Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance
        # (globalny, cache'owany raz - audyt 2026-07-05)
        out['macro_ext'] = _load_macro_extended()

        return out
    except Exception as e:
        logger.error(f'{symbol}: load error {e}')
        return None


def build_features_for_symbol(data: Dict, symbol: str, extra_horizons: list = None) -> pd.DataFrame:
    """Buduje DataFrame z features + labels dla 1 symbolu.

    v2.0: pre-compute RSI/trend dla 4h+1d RAZ, funding searchsorted O(log n).
    """
    df_1h = data['1h']
    df_4h = data['4h']
    df_1d = data['1d']
    df_fund = data.get('funding')
    df_taker = data.get('taker')

    if len(df_1h) < MIN_HISTORY + LOOKAHEAD_BARS:
        return pd.DataFrame()

    # === DANE 1H ===
    closes = df_1h['close'].values.astype(np.float64)
    highs = df_1h['high'].values.astype(np.float64)
    lows = df_1h['low'].values.astype(np.float64)
    volumes = df_1h['volume'].values.astype(np.float64)
    times = df_1h['timestamp'].values  # numpy datetime64[ns]
    n = len(closes)

    # === PRE-COMPUTE 1H (raz dla calego szeregu) ===
    rsi_arr = calc_rsi(closes, RSI_PERIOD)
    ema_slow_arr = calc_ema(closes, EMA_SLOW)
    ema_mid_arr = calc_ema(closes, EMA_MID)
    atr_arr = calc_atr(highs, lows, closes, ATR_PERIOD)
    adx_arr = calc_adx_closes_series(closes)

    # === NOWE CECHY (dołożone po audycie sesji) ===
    ema_fast_arr = calc_ema(closes, EMA_FAST)
    trend_1h_arr = precompute_trend_series(closes, EMA_FAST, EMA_SLOW)
    volume_zscore_arr = calc_volume_zscore(volumes, 30)
    macd_hist_arr = calc_macd_hist(closes)
    div_rsi_arr = calc_divergence(closes, rsi_arr, 20)  # dywergencja cena vs RSI (A/B)
    sd_prox_arr = calc_sd_proximity(highs, lows, closes, atr_arr, 50)  # Supply&Demand proximity (A/B)
    bars_cross_arr = calc_bars_since_cross(closes, 20, 50, 100)  # wiek trendu EMA20/50 (A/B)
    sr_dist_arr, sr_strength_arr, sh_level_ff, sl_level_ff = calc_swing_sr(closes, highs, lows)
    fib_dist_arr = calc_fib_dist(closes, sh_level_ff, sl_level_ff)
    rptr_arr = calc_rptr_features(closes, highs, lows, volumes, rsi_arr, atr_arr)
    x_arr = calc_x_features(closes, highs, lows, volumes)

    # === NEW v2.0: PRE-COMPUTE 4H (raz dla calego szeregu 4h) ===
    closes_4h_all = df_4h['close'].values.astype(np.float64)
    times_4h = df_4h['timestamp'].values
    if len(closes_4h_all) >= RSI_PERIOD + 1:
        rsi_4h_arr = calc_rsi(closes_4h_all, RSI_PERIOD)
        trend_4h_arr = precompute_trend_series(closes_4h_all, EMA_FAST, EMA_SLOW)
    else:
        rsi_4h_arr = np.full(len(closes_4h_all), 50.0)
        trend_4h_arr = np.zeros(len(closes_4h_all), dtype=np.int8)

    # === NEW v2.0: PRE-COMPUTE 1D (raz dla calego szeregu 1d) ===
    closes_1d_all = df_1d['close'].values.astype(np.float64)
    times_1d = df_1d['timestamp'].values
    if len(closes_1d_all) >= RSI_PERIOD + 1:
        rsi_1d_arr = calc_rsi(closes_1d_all, RSI_PERIOD)
        trend_1d_arr = precompute_trend_series(closes_1d_all, EMA_FAST, EMA_SLOW)
    else:
        rsi_1d_arr = np.full(len(closes_1d_all), 50.0)
        trend_1d_arr = np.zeros(len(closes_1d_all), dtype=np.int8)

    # === NEW v2.0: PRE-COMPUTE FUNDING (sort raz + searchsorted) ===
    if df_fund is not None and len(df_fund) > 0:
        funding_times = df_fund['timestamp'].values  # juz posortowane przy load
        funding_rates = df_fund['funding_rate'].values.astype(np.float64)
        has_funding = True
    else:
        funding_times = None
        funding_rates = None
        has_funding = False

    df_oi = data.get('oi')
    if df_oi is not None and len(df_oi) > 0:
        oi_times = df_oi['timestamp'].values
        oi_log_arr = df_oi['oi_total_log'].values.astype(np.float64)
        oi_chg_arr = df_oi['oi_change_24h'].values.astype(np.float64)
        oi_z_arr = df_oi['oi_zscore_30d'].values.astype(np.float64)
        has_oi = True
    else:
        oi_times = None
        oi_log_arr = oi_chg_arr = oi_z_arr = None
        has_oi = False

    if df_taker is not None and len(df_taker) > 0:
        taker_times = df_taker['timestamp'].values
        taker_ratio_arr = df_taker['taker_buy_ratio'].values.astype(np.float64)
        has_taker = True
    else:
        taker_times = None
        taker_ratio_arr = None
        has_taker = False

    CVD_KOL = ['cvd_z6', 'cvd_z24', 'delta_pct_ma6']
    df_cvd = data.get('cvd')
    if df_cvd is not None and len(df_cvd) > 0:
        cvd_times = df_cvd['timestamp'].values
        cvd_arr = {k: df_cvd[k].values.astype(np.float64) for k in CVD_KOL if k in df_cvd.columns}
        has_cvd = len(cvd_arr) == len(CVD_KOL)
    else:
        cvd_times, cvd_arr, has_cvd = None, {}, False

    PRZEKR_KOL = ['resid_ret_4h', 'resid_ret_24h', 'rank_mom_24h',
                  'rank_vol_chg', 'event_dekorelacji']
    df_prz = data.get('przekrojowe')
    if df_prz is not None and len(df_prz) > 0:
        prz_times = df_prz['timestamp'].values
        prz_arr = {k: df_prz[k].values.astype(np.float64) for k in PRZEKR_KOL if k in df_prz.columns}
        has_prz = len(prz_arr) == len(PRZEKR_KOL)
    else:
        prz_times, prz_arr, has_prz = None, {}, False

    KOREL_KOL = ['btc_corr_24h', 'btc_corr_72h', 'btc_corr_change',
                 'btc_beta_72h', 'rel_strength_btc', 'corr_breakdown']
    df_kor = data.get('korelacje')
    if df_kor is not None and len(df_kor) > 0:
        kor_times = df_kor['timestamp'].values
        kor_arr = {k: df_kor[k].values.astype(np.float64) for k in KOREL_KOL if k in df_kor.columns}
        has_kor = len(kor_arr) == len(KOREL_KOL)
    else:
        kor_times, kor_arr, has_kor = None, {}, False

    df_ls = data.get('ls')
    if df_ls is not None and len(df_ls) > 0:
        ls_times = df_ls['timestamp'].values
        ls_arr = df_ls['ls_ratio'].values.astype(np.float64)
        # zmiana 24h w samych danych ls (1h granulacja -> shift 24 wierszy)
        _ls_prev = np.roll(ls_arr, 24); _ls_prev[:24] = ls_arr[:24]
        ls_chg_arr = ls_arr - _ls_prev
        has_ls = True
    else:
        ls_times = None
        ls_arr = ls_chg_arr = None
        has_ls = False

    # Fear & Greed Index (dzienny, searchsorted jak funding/OI - audyt 2026-07-04)
    df_fg = data.get('fear_greed')
    if df_fg is not None and len(df_fg) > 0:
        fg_times = df_fg['timestamp'].values
        fg_vals = df_fg['value'].values.astype(np.float64)
        has_fg = True
    else:
        fg_times = None
        fg_vals = None
        has_fg = False

    # BTC context (searchsorted per-timeframe - audyt 2026-07-04)
    btc_ctx = data.get('btc_context') or {}
    has_btc = bool(btc_ctx)

    # Macro extended (searchsorted per-seria, dzienne - audyt 2026-07-05)
    macro_ext = data.get('macro_ext') or {}

    # Pre-compute 24h ago timestamps (vectorized)
    times_24h_ago = times - np.timedelta64(24, 'h')

    records = []

    for i in range(MIN_HISTORY, n - LOOKAHEAD_BARS):
        ts = times[i]
        cur = closes[i]

        # === FEATURES Z 1H (z pre-computed arrays) ===
        rsi = rsi_arr[i]
        ema_s = ema_slow_arr[i]
        ema_m = ema_mid_arr[i]
        atr = atr_arr[i]
        atr_pct = atr / cur * 100 if cur > 0 else 0

        # Momentum (10-bar RoC)
        momentum = (cur / closes[i - 10] - 1) * 100 if i >= 10 else 0

        # CVD x ADX — wartosc nadawana nizej, w bloku CECHY CVD (potrzebuje
        # adx_14, liczonego dopiero w tej petli). Do 2026-08-24 zostawala
        # ZEREM na sztywno: HAI_LEJEK_CVD dodawal ja do list cech RF/CAT/MIX,
        # wiec modele dostawaly kolumne stalej zerowej i nie mogl tego nikt
        # zauwazyc — zerowa wariancja nie rzuca bledem, po prostu nic nie wnosi.
        cvd_x_adx = 0.0

        # === VOLUME FEATURES ===
        vol_window = volumes[max(0, i - 30):i]
        if len(vol_window) > 0:
            avg_vol = vol_window.mean()
            volume_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0

        # === BOLLINGER POSITION + BANDWIDTH (20-period, 2 std) ===
        if i >= 20:
            bb_window = closes[i - 20:i]
            bb_mean = bb_window.mean()
            bb_std = bb_window.std()
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            bb_width = bb_upper - bb_lower
            price_position_bb = (cur - bb_lower) / bb_width if bb_width > 0 else 0.5
            bb_bandwidth_pct = (4 * bb_std / bb_mean) if bb_mean > 0 else 0.0
        else:
            price_position_bb = 0.5
            bb_bandwidth_pct = 0.0

        adx_14 = float(adx_arr[i])

        # === FEATURES Z 4H (NEW v2.0: searchsorted lookup) ===
        # Znajdz ostatnia swieczke 4h <= ts (binarne wyszukiwanie)
        idx_4h = np.searchsorted(times_4h, ts, side='right') - 1
        if idx_4h >= 30:
            rsi_4h = rsi_4h_arr[idx_4h]
            trend_4h = trend_4h_arr[idx_4h]
        else:
            rsi_4h = 50.0
            trend_4h = 0

        # === FEATURES Z 1D (NEW v2.0: searchsorted lookup) ===
        idx_1d = np.searchsorted(times_1d, ts, side='right') - 1
        if idx_1d >= 30:
            rsi_1d = rsi_1d_arr[idx_1d]
            trend_1d = trend_1d_arr[idx_1d]
        else:
            rsi_1d = 50.0
            trend_1d = 0

        # === FUNDING (NEW v2.0: searchsorted O(log n) zamiast sort O(n log n)) ===
        if has_funding:
            # Ostatni funding <= ts
            f_idx = np.searchsorted(funding_times, ts, side='right') - 1
            funding_rate = float(funding_rates[f_idx]) if f_idx >= 0 else 0.0
            # Funding 24h temu
            ts_24h_ago = times_24h_ago[i]
            f_idx_24h = np.searchsorted(funding_times, ts_24h_ago, side='right') - 1
            funding_24h_ago = float(funding_rates[f_idx_24h]) if f_idx_24h >= 0 else 0.0
            funding_change_24h = funding_rate - funding_24h_ago
        else:
            funding_rate = 0.0
            funding_change_24h = 0.0

        # === OI (searchsorted jak funding) ===
        if has_oi:
            o_idx = np.searchsorted(oi_times, ts, side='right') - 1
            if o_idx >= 0:
                oi_total_log = float(oi_log_arr[o_idx])
                oi_change_24h = float(oi_chg_arr[o_idx])
                oi_zscore_30d = float(oi_z_arr[o_idx])
            else:
                oi_total_log = oi_change_24h = oi_zscore_30d = 0.0
        else:
            oi_total_log = oi_change_24h = oi_zscore_30d = 0.0

        # === TAKER BUY RATIO (searchsorted jak funding/OI) ===
        if has_taker:
            tk_idx = np.searchsorted(taker_times, ts, side='right') - 1
            taker_buy_ratio = float(taker_ratio_arr[tk_idx]) if tk_idx >= 0 else 0.5
        else:
            taker_buy_ratio = 0.5

        # === LS RATIO (searchsorted jak taker; neutralne 1.0 gdy brak) ===
        if has_ls:
            ls_idx = np.searchsorted(ls_times, ts, side='right') - 1
            ls_ratio = float(ls_arr[ls_idx]) if ls_idx >= 0 else 1.0
            ls_ratio_chg_24h = float(ls_chg_arr[ls_idx]) if ls_idx >= 0 else 0.0
        else:
            ls_ratio = 1.0
            ls_ratio_chg_24h = 0.0

        # === CECHY PRZEKROJOWE (searchsorted; godzinowe, bez ryzyka stempla).
        # Neutralne przy braku: residualy 0 (brak wlasnego momentum),
        # percentyle 0.5 (srodek stawki), zdarzenie 0 (nie wystapilo). ===
        if has_prz:
            _pi = np.searchsorted(prz_times, ts, side='right') - 1
            if _pi >= 0:
                def _pv(nazwa, dom):
                    v = prz_arr[nazwa][_pi]
                    return float(v) if np.isfinite(v) else dom
                resid_ret_4h = _pv('resid_ret_4h', 0.0)
                resid_ret_24h = _pv('resid_ret_24h', 0.0)
                rank_mom_24h = _pv('rank_mom_24h', 0.5)
                rank_vol_chg = _pv('rank_vol_chg', 0.5)
                event_dekorelacji = _pv('event_dekorelacji', 0.0)
            else:
                resid_ret_4h = resid_ret_24h = 0.0
                rank_mom_24h = rank_vol_chg = 0.5
                event_dekorelacji = 0.0
        else:
            resid_ret_4h = resid_ret_24h = 0.0
            rank_mom_24h = rank_vol_chg = 0.5
            event_dekorelacji = 0.0

        # === CECHY CVD (searchsorted; dane godzinowe, bez ryzyka stempla).
        # Neutralne przy braku: z-score 0 (przeplyw jak zwykle), delta_pct 0
        # (rownowaga kupno/sprzedaz). Backfill orderflow siega 2026-07-27,
        # wiec najswiezsze okno moze miec luke — wtedy wpada neutralne 0. ===
        if has_cvd:
            _ci = np.searchsorted(cvd_times, ts, side='right') - 1
            if _ci >= 0:
                def _cv(nazwa):
                    v = cvd_arr[nazwa][_ci]
                    return float(v) if np.isfinite(v) else 0.0
                cvd_z6 = _cv('cvd_z6')
                cvd_z24 = _cv('cvd_z24')
                delta_pct_ma6 = _cv('delta_pct_ma6')
            else:
                cvd_z6 = cvd_z24 = delta_pct_ma6 = 0.0
        else:
            cvd_z6 = cvd_z24 = delta_pct_ma6 = 0.0
        # przeplyw wazony sila trendu: napor kupujacych liczy sie inaczej
        # w trendzie niz w konsolidacji. Skala ADX/25 ~ 1.0 przy progu bramki.
        cvd_x_adx = cvd_z24 * (adx_14 / 25.0)

        # === KORELACJA Z BTC (searchsorted jak ls/taker; dane GODZINOWE, wiec
        # bez ryzyka stempla wstecznego — w odroznieniu od macro/fear_greed/OI,
        # ktore byly dzienne i wszystkie trzy mialy przeciek).
        # Neutralne wartosci gdy brak: korelacja 0 (brak sprzezenia), beta 1
        # (porusza sie jak BTC), sila relatywna 0, flaga 0. ===
        if has_kor:
            _ki = np.searchsorted(kor_times, ts, side='right') - 1
            if _ki >= 0:
                def _kv(nazwa, dom):
                    v = kor_arr[nazwa][_ki]
                    return float(v) if np.isfinite(v) else dom
                btc_corr_24h = _kv('btc_corr_24h', 0.0)
                btc_corr_72h = _kv('btc_corr_72h', 0.0)
                btc_corr_change = _kv('btc_corr_change', 0.0)
                btc_beta_72h = _kv('btc_beta_72h', 1.0)
                rel_strength_btc = _kv('rel_strength_btc', 0.0)
                corr_breakdown = _kv('corr_breakdown', 0.0)
            else:
                btc_corr_24h = btc_corr_72h = btc_corr_change = 0.0
                btc_beta_72h = 1.0
                rel_strength_btc = corr_breakdown = 0.0
        else:
            btc_corr_24h = btc_corr_72h = btc_corr_change = 0.0
            btc_beta_72h = 1.0
            rel_strength_btc = corr_breakdown = 0.0

        # === FEAR & GREED INDEX (searchsorted, dzienny - audyt 2026-07-04) ===
        if has_fg:
            fg_idx = np.searchsorted(fg_times, ts, side='right') - 1
            fear_greed = float(fg_vals[fg_idx]) if fg_idx >= 0 else 50.0
        else:
            fear_greed = 50.0

        # === BTC CONTEXT (searchsorted per-timeframe - audyt 2026-07-04) ===
        if has_btc:
            b1_idx = np.searchsorted(btc_ctx['1h_times'], ts, side='right') - 1
            btc_trend_1h = float(btc_ctx['1h_trend'][b1_idx]) if b1_idx >= 0 else 0.0
            b4_idx = np.searchsorted(btc_ctx['4h_times'], ts, side='right') - 1
            if b4_idx >= 0:
                btc_trend_4h = float(btc_ctx['4h_trend'][b4_idx])
                btc_rsi_4h = float(btc_ctx['4h_rsi'][b4_idx])
            else:
                btc_trend_4h = 0.0
                btc_rsi_4h = 50.0
            bd_idx = np.searchsorted(btc_ctx['1d_times'], ts, side='right') - 1
            btc_trend_1d = float(btc_ctx['1d_trend'][bd_idx]) if bd_idx >= 0 else 0.0
        else:
            btc_trend_1h = btc_trend_4h = btc_trend_1d = 0.0
            btc_rsi_4h = 50.0

        # === MACRO EXTENDED: Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance
        # (zmiana % dzien-do-dnia, searchsorted - audyt 2026-07-05).
        # WYJATEK btc_dominance: kolumna 'value' w btc_dominance.parquet JUZ
        # jest gotowa 7d% zmiana BTC mcap (patrz fetch_macro.py), NIE poziomem -
        # liczenie kolejnej % zmiany na tym dawalo bezsensowna "zmiane zmiany"
        # (znaleziono empirycznie: 38% "dzienna zmiana" - artefakt dzielenia
        # blisko zera). Dla btc_dominance bierzemy wartosc WPROST.
        macro_vals = {}
        for _name in _MACRO_EXT_TICKERS:
            _times = macro_ext.get(f'{_name}_times')
            _vals = macro_ext.get(f'{_name}_vals')
            if _times is not None and _vals is not None and len(_vals) > 1:
                _idx = np.searchsorted(_times, ts, side='right') - 1
                if _name == 'btc_dominance':
                    macro_vals[_name] = float(_vals[_idx]) if _idx >= 0 else 0.0
                elif _idx >= 1:
                    _cur_v = _vals[_idx]
                    _prev_v = _vals[_idx - 1]
                    macro_vals[_name] = float((_cur_v / _prev_v - 1) * 100) if _prev_v != 0 else 0.0
                else:
                    macro_vals[_name] = 0.0
            else:
                macro_vals[_name] = 0.0

        # === TRIPLE BARRIER LABEL — 3-klasowy (audyt 2026-07-03/04) ===
        # Glowny label (48h, TP=4xATR/SL=1xATR) + dwa dodatkowe warianty
        # liczone RAZEM (te same highs/lows/atr, zero dodatkowego kosztu
        # budowania cech): label_fast24h (krotszy lookahead 24h - szybsze
        # ruchy) i label_sharp6x (ten sam 48h lookahead, ale TP=6xATR -
        # tylko bardzo zdecydowane ruchy, do wariantu CAT "wyostrzonego").
        label_long = triple_barrier_label(cur, atr, highs, lows, i, n,
                                           LOOKAHEAD_BARS, TP_ATR_MULT, SL_ATR_MULT)
        label_fast24h = triple_barrier_label(cur, atr, highs, lows, i, n,
                                              24, TP_ATR_MULT, SL_ATR_MULT)
        label_sharp6x = triple_barrier_label(cur, atr, highs, lows, i, n,
                                              LOOKAHEAD_BARS, 6.0, SL_ATR_MULT)
        # Skrajnosci horyzontu (audyt 2026-07-04, na wyrazna prosbe "ile w
        # dol/gore mozemy zejsc") - fast6h (dolny praktyczny limit, ponizej
        # tego TP/SL to glownie szum) i wide96h (gorny praktyczny limit,
        # powyzej tego cechy tracace moc predykcyjna wzgledem dryfu rynku).
        label_fast6h = triple_barrier_label(cur, atr, highs, lows, i, n,
                                             6, TP_ATR_MULT, SL_ATR_MULT)
        label_wide96h = triple_barrier_label(cur, atr, highs, lows, i, n,
                                              96, TP_ATR_MULT, SL_ATR_MULT)
        # h72 (audyt 2026-07-05) - punkt posredni miedzy 48h (dobry) a 96h
        # (potwierdzone gorszy) - pokazuje ze degradacja jest STOPNIOWA,
        # zaczyna sie zaraz po 48h (LGB_H72 acc=0.532 < LGB 48h acc=0.569).
        label_h72 = triple_barrier_label(cur, atr, highs, lows, i, n,
                                          72, TP_ATR_MULT, SL_ATR_MULT)
        # Dowolne dodatkowe horyzonty na zadanie (audyt 2026-07-04) - do
        # eksperymentu "horyzont jako cecha": jeden model uczony na
        # POLACZONYM (stacked) zbiorze wielu horyzontow, z horizon_hours
        # jako dodatkowa cecha wejsciowa zamiast osobnego modelu per horyzont.
        extra_labels = {}
        if extra_horizons:
            for _h in extra_horizons:
                extra_labels[_h] = triple_barrier_label(cur, atr, highs, lows, i, n,
                                                          _h, TP_ATR_MULT, SL_ATR_MULT)

        # Intraday seasonality features (cykliczne kodowanie czasu)
        ts_dt = pd.Timestamp(ts).tz_localize('UTC') if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)
        hour_val = ts_dt.hour + ts_dt.minute / 60.0
        hour_sin = float(np.sin(2 * np.pi * hour_val / 24))
        hour_cos = float(np.cos(2 * np.pi * hour_val / 24))
        dow = float(ts_dt.dayofweek)  # 0=pon, 6=niedz

        record = {
            'symbol': symbol,
            'timestamp': ts,
            # Features v8.2 (19)
            'rsi': float(rsi),
            'rsi_4h': float(rsi_4h),
            'rsi_1d': float(rsi_1d),
            'ema_slow_r': float((cur / ema_s - 1) * 100) if ema_s > 0 else 0,
            'ema_mid_r':  float((cur / ema_m - 1) * 100) if ema_m > 0 else 0,
            'atr_pct': float(atr_pct),
            'momentum': float(momentum),
            'trend_4h': float(trend_4h),
            'trend_1d': float(trend_1d),
            'volume_ratio': float(volume_ratio),
            'funding_rate': float(funding_rate),
            'price_position_bb': float(price_position_bb),
            'bb_bandwidth_pct': float(bb_bandwidth_pct),
            'oi_total_log': float(oi_total_log),
            'oi_change_24h': float(oi_change_24h),
            'oi_zscore_30d': float(oi_zscore_30d),
            # Interakcja funding x OI-zscore (B3 gen.Dir-v1, 2026-07-19,
            # wskazowka Grok): sam funding i sam oi_zscore dyskryminuja slabo-
            # srednio osobno (z~0.36/0.42), ale ich ILOCZYN lapie rezim
            # "przegrzania" - dodatni funding (longi placa) PRZY wysokim
            # oi_zscore (skrajny wolumen pozycji) = kontrarianski sygnal
            # odwrocenia. Cecha DEDYKOWANA specjaliscie-lgb, zerowy koszt
            # (oba czynniki juz policzone per-row).
            'funding_x_oizscore': float(funding_rate) * float(oi_zscore_30d),
            'hour_sin': hour_sin,
            'day_of_week': dow,
            'adx_14': adx_14,
            # Cechy dolozone dzis (byly liczone jako zmienne posrednie, nigdy
            # nie trafialy do record — LGB/XGB dostawaly za nie zera w backteście)
            'ema_fast_r': float((cur / ema_fast_arr[i] - 1) * 100) if ema_fast_arr[i] > 0 else 0.0,
            'trend_1h': float(trend_1h_arr[i]),
            'volume_zscore': float(volume_zscore_arr[i]),
            'hour_cos': hour_cos,
            'funding_change_24h': float(funding_change_24h),
            # Nowe cechy (audyt sesji 2026-07-03)
            'macd_hist': float(macd_hist_arr[i]),
            'div_rsi': float(div_rsi_arr[i]),
            'sd_prox': float(sd_prox_arr[i]),
            'bars_cross': float(bars_cross_arr[i]),
            'taker_buy_ratio': float(taker_buy_ratio),
            'ls_ratio': float(ls_ratio),
            'ls_ratio_chg_24h': float(ls_ratio_chg_24h),
            'sr_dist_pct': float(sr_dist_arr[i]),
            'sr_node_strength': float(sr_strength_arr[i]),
            'fib_dist_pct': float(fib_dist_arr[i]),
            # Cechy RPTR: RF/ET specjalistyczne (2026-08-05) - parytet z live
            # (hai_common.features rptr_feats). Aliasy (htf_trend_4h/phantomflow)
            # i interakcje (O(1), zero dodatkowego kosztu) liczone tu inline.
            'r_stoch_rsi': float(rptr_arr['r_stoch_rsi'][i]),
            'r_cci_20': float(rptr_arr['r_cci_20'][i]),
            'r_di_spread': float(rptr_arr['r_di_spread'][i]),
            'r_macd_signal_dist': float(rptr_arr['r_macd_signal_dist'][i]),
            'r_momentum_20': float(rptr_arr['r_momentum_20'][i]),
            'r_atr_zscore_20': float(rptr_arr['r_atr_zscore_20'][i]),
            'r_volume_zscore_50': float(rptr_arr['r_volume_zscore_50'][i]),
            'r_range_compression': float(rptr_arr['r_range_compression'][i]),
            'r_vwap_distance': float(rptr_arr['r_vwap_distance'][i]),
            'r_structure_bias': float(rptr_arr['r_structure_bias'][i]),
            'r_htf_trend_4h': float(trend_4h),
            'e_liquidity_sweep_high': float(rptr_arr['e_liquidity_sweep_high'][i]),
            'e_liquidity_sweep_low': float(rptr_arr['e_liquidity_sweep_low'][i]),
            'e_fvg_size': float(rptr_arr['e_fvg_size'][i]),
            'e_fvg_direction': float(rptr_arr['e_fvg_direction'][i]),
            'e_bos_choch': float(rptr_arr['e_bos_choch'][i]),
            'e_equal_highs_lows_count': float(rptr_arr['e_equal_highs_lows_count'][i]),
            'e_ichimoku_cloud_thickness': float(rptr_arr['e_ichimoku_cloud_thickness'][i]),
            'e_tk_cross': float(rptr_arr['e_tk_cross'][i]),
            'e_volume_delta_imbalance': float(rptr_arr['e_volume_delta_imbalance'][i]),
            'e_orderblock_strength': float(rptr_arr['e_orderblock_strength'][i]),
            'e_phantomflow_score': float(sr_strength_arr[i]),
            'e_rsi_x_funding': float(rsi) * float(funding_rate),
            'e_atr_pct_x_vol_zscore': float(atr_pct) * float(volume_zscore_arr[i]),
             'x_obv_slope_24h': float(rptr_arr['x_obv_slope_24h'][i]),
             'x_weekend': 1.0 if dow in (5.0, 6.0) else 0.0,
             # EKSPERYMENTALNE x_* (2026-08-06): dodane do cache WFV z calc_x_features
             # (parytet hai_common.features) - divcore configi ich uzywaja.
             'x_real_vol_6h': float(x_arr['x_real_vol_6h'][i]),
             'x_real_vol_24h': float(x_arr['x_real_vol_24h'][i]),
             'x_parkinson_24h': float(x_arr['x_parkinson_24h'][i]),
             'x_rsi_7': float(x_arr['x_rsi_7'][i]),
             'x_rsi_21': float(x_arr['x_rsi_21'][i]),
             'x_roc_4': float(x_arr['x_roc_4'][i]),
             'x_roc_24': float(x_arr['x_roc_24'][i]),
             'x_roc_168': float(x_arr['x_roc_168'][i]),
             'x_ema9_over_21': float(x_arr['x_ema9_over_21'][i]),
             'x_ema21_over_55': float(x_arr['x_ema21_over_55'][i]),
             'x_ema50_over_200': float(x_arr['x_ema50_over_200'][i]),
             'vwap_dev': float(x_arr['vwap_dev'][i]),
             'x_btc_leadlag': float(macro_vals['btc_dominance']) - float(momentum),
             'cvd_x_adx': float(cvd_x_adx),
            # Fear & Greed Index (audyt 2026-07-04, pelna historia 2018-dzis)
            'cvd_z6': cvd_z6,
            'cvd_z24': cvd_z24,
            'delta_pct_ma6': delta_pct_ma6,
            'resid_ret_4h': resid_ret_4h,
            'resid_ret_24h': resid_ret_24h,
            'rank_mom_24h': rank_mom_24h,
            'rank_vol_chg': rank_vol_chg,
            'event_dekorelacji': event_dekorelacji,
            'btc_corr_24h': btc_corr_24h,
            'btc_corr_72h': btc_corr_72h,
            'btc_corr_change': btc_corr_change,
            'btc_beta_72h': btc_beta_72h,
            'rel_strength_btc': rel_strength_btc,
            'corr_breakdown': corr_breakdown,
            'fear_greed': float(fear_greed),
            # BTC context (audyt 2026-07-04, korelacja altcoinow z BTC 0.6-0.85)
            'btc_trend_1h': float(btc_trend_1h),
            'btc_trend_4h': float(btc_trend_4h),
            'btc_trend_1d': float(btc_trend_1d),
            'btc_rsi_4h': float(btc_rsi_4h),
            # Macro extended: Gold/Oil/SP500/VIX/US10Y/DXY/BTC dominance
            # (zmiana % dzien-do-dnia, audyt 2026-07-05 - "te dane tez tylko
            # szumialy ale dobra daj je")
            'gold_chg': macro_vals['gold'],
            'oil_wti_chg': macro_vals['oil_wti'],
            'sp500_chg': macro_vals['sp500'],
            'vix_chg': macro_vals['vix'],
            'us10y_chg': macro_vals['us10y_yield'],
            'dxy_chg': macro_vals['dxy'],
            'btc_dominance_chg': macro_vals['btc_dominance'],
            # Label (glowny + warianty specjalistow, audyt 2026-07-04)
            'label_long': int(label_long),
            'label_fast24h': int(label_fast24h),
            'label_sharp6x': int(label_sharp6x),
            'label_fast6h': int(label_fast6h),
            'label_wide96h': int(label_wide96h),
            'label_h72': int(label_h72),
            # Binarny wariant (audyt 2026-07-05, panel 'Trening niestandardowy' -
            # opcja schematu labelu). Pochodny z label_long: 1=LONG, 0=NEUTRAL+SHORT
            # razem. UWAGA: to STARY schemat (jak DEV/LAB przed kalka EPV) -
            # test 2026-07-04 pokazal ze 3-class daje REALNIE lepszy PF (1.13->1.33
            # po przejsciu z binarnego na 3-class) mimo nizszej "ladniejszej"
            # accuracy walidacyjnej binarnego. Dostepne do eksperymentu, NIE
            # domyslne.
            'label_binary': int(1 if label_long == 1 else 0),
        }
        for _h, _lbl in extra_labels.items():
            record[f'label_h{_h}'] = int(_lbl)
        records.append(record)

    return pd.DataFrame(records)


def _build_one_symbol(sym: str, extra_horizons: list = None) -> Optional[pd.DataFrame]:
    """Worker dla joblib.Parallel - laduje + buduje features dla 1 symbolu."""
    t0 = time.time()
    data = load_symbol_data(sym)
    if data is None:
        logger.warning(f'{sym}: brak danych')
        return None
    df = build_features_for_symbol(data, sym, extra_horizons=extra_horizons)
    if df.empty:
        logger.warning(f'{sym}: pusty DataFrame')
        return None
    # Cechy mapy likwidacji (dist_below_liq/dist_above_liq) - WSPOLNA funkcja z
    # backtesterem (parzystosc!). Zweryfikowane +0.100 precyzji_LONG na 2 latach
    # (genflow_liqmap_deep). Liczone z pelnej siatki 1h + dzienne OI + gleboki
    # ls_ratio, merge po timestamp. Brak pokrycia -> DIST_DEFAULT (nie 0!).
    try:
        from .liqmap_features import compute_liq_dist_from_warehouse, DIST_DEFAULT
        _o = data['1h'][['timestamp', 'close', 'high', 'low']].copy()
        _o['timestamp'] = pd.to_datetime(_o['timestamp'], utc=True).dt.tz_localize(None)
        _db, _da = compute_liq_dist_from_warehouse(_o, sym)
        _ld = pd.DataFrame({'timestamp': _o['timestamp'].values,
                            'dist_below_liq': _db, 'dist_above_liq': _da})
        df = df.merge(_ld, on='timestamp', how='left')
        df['dist_below_liq'] = df['dist_below_liq'].fillna(DIST_DEFAULT)
        df['dist_above_liq'] = df['dist_above_liq'].fillna(DIST_DEFAULT)
    except Exception as _liq_e:
        logger.warning(f'{sym}: liq features skip ({_liq_e}) -> DIST_DEFAULT')
        df['dist_below_liq'] = 30.0
        df['dist_above_liq'] = 30.0
    elapsed = time.time() - t0
    logger.info(f'  {sym}: {len(df)} samples ({elapsed:.1f}s)')
    return df


def build_dataset(extra_horizons: list = None) -> pd.DataFrame:
    """Buduje pelny dataset dla wszystkich symboli.

    v2.0: joblib.Parallel(n_jobs=6) - 6 symboli rownolegle.
    `extra_horizons` (2026-07-13): dodatkowe etykiety label_h{H} obok
    standardowych (6/24/48/72/96). Potrzebne do odtworzenia
    lgb_multi_horizon (trenowany na [4,6,8,16,24,36,48,72] ze stackowanego
    zbioru, z horizon_hours jako CECHA — f1=0.597, najlepszy w calej puli).
    build_features_for_symbol umial to od dawna, tylko build_dataset nie
    przekazywal parametru dalej.
    """
    from joblib import Parallel, delayed

    logger.info(f'Building features for {len(SYMBOLS)} symbols (parallel n_jobs={N_PARALLEL_SYMBOLS})'
                f'{f" + extra horyzonty {extra_horizons}" if extra_horizons else ""}...')
    t0 = time.time()

    results = Parallel(n_jobs=N_PARALLEL_SYMBOLS, backend='loky', verbose=0)(
        delayed(_build_one_symbol)(sym, extra_horizons) for sym in SYMBOLS
    )

    elapsed = time.time() - t0
    all_dfs = [df for df in results if df is not None and not df.empty]

    if not all_dfs:
        raise RuntimeError('Brak danych do treningu')

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values('timestamp').reset_index(drop=True)
    # Data-hygiene (audyt 2026-07-29): usun wiersze adx_14==0 — to zombie-dane
    # coinow delistowanych/przemianowanych z Binance (FTM->S, OCEAN->FET, OMG,
    # WAVES, MKR) + swiezy glitch (TON): perp trwa near-frozen po delistingu ->
    # atr_pct≈0 -> ADX kolapsuje do 0, triple-barrier na plaskiej cenie
    # degeneruje do ~96% label==1 (71k wierszy) i uczy modele falszywego "always
    # LONG". adx==0 w plynnym rynku praktycznie nie wystepuje -> filtr bezpieczny,
    # przyczynowo-agnostyczny (lapie tez przyszle zombie), zachowuje dobra historie.
    if 'adx_14' in full.columns:
        _before = len(full)
        full = full[full['adx_14'] > 0].reset_index(drop=True)
        _dropped = _before - len(full)
        if _dropped:
            logger.info(f'Data-hygiene: usunieto {_dropped:,} wierszy adx_14==0 (zombie/frozen)')
    logger.info(f'Dataset built in {elapsed:.1f}s ({elapsed/60:.1f} min) | total: {len(full):,} samples')
    return full


def _safe_feats(name: str, df_columns: set) -> list:
    """MODEL_FEATURES[name] przefiltrowane do kolumn istniejacych w dataframe.
    Pozwala dodawac eksperymentalne cechy (np. of_cvd_chg) bez psucia produkcji."""
    return [f for f in MODEL_FEATURES.get(name, []) if f in df_columns]


def train_models(df: pd.DataFrame, only: "list[str] | None" = None,
                  class_weights: "dict | None" = None) -> Dict:
    """Trenuje modele drzew na danych z df, kazdy na WLASNYM zestawie cech
    (MODEL_FEATURES, audyt 2026-07-04 - specjalizacja per model wg argmax
    feature_importance). `only` ogranicza trening do podanych nazw (np.
    ['lgb']) - do szybkich testow jednego modelu naraz."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from .walk_forward import walk_forward_eval, select_best_model

    _all_names = ['lgb', 'xgb', 'rf', 'cat', 'histgb']
    # only moze zawierac nazwy spoza _all_names (np. 'long_spec' - dedykowany
    # booster, nie czesc glownych 5) - wtedy trenujemy DOKLADNIE to co podano.
    _train_names = list(only) if only is not None else _all_names

    # === WALK-FORWARD ocena (expanding 5-fold) na ostatnich 1M sample ===
    # Subsample żeby WFV nie trwało >30 min (pełny dataset 3.7M jest za duży dla RF WF)
    #
    # HAI_SKIP_WF=1 wylacza te ocene (2026-07-13). Powod: hai_wfv.py robi
    # walk-forward NA POZIOMIE OKIEN (trening per okno na danych sprzed cutoffu),
    # wiec wewnetrzna 3-foldowa walidacja jest tu zbedna — a kosztuje 3 DODATKOWE
    # treningi na kazdy model. Zmierzone: 1 model = 5-6 min zamiast ~1.5 min;
    # przy 66 modelach x 12 okien to ~18h zamiast ~4h. Przy treningu PRODUKCYJNYM
    # (hai_train.py) ocena zostaje wlaczona — tam metryki wf_* sa uzyteczne.
    _skip_wf = os.environ.get("HAI_SKIP_WF") == "1"
    WF_MAX_SAMPLES = 1_000_000
    if _skip_wf:
        logger.info('Walk-forward eval POMINIETA (HAI_SKIP_WF=1)')
        df_wf = df.iloc[:0]
    else:
        logger.info(f'=== Walk-forward evaluation (3-fold expanding, max {WF_MAX_SAMPLES:,} samples) ===')
    if not _skip_wf and len(df) > WF_MAX_SAMPLES:
        df_wf = df.iloc[-WF_MAX_SAMPLES:].copy()
        logger.info(f'WFV subsample: {len(df_wf):,} (ostatnie {WF_MAX_SAMPLES:,} z {len(df):,})')
    else:
        df_wf = df
    _wf = {}
    for _name in ([] if _skip_wf else _train_names):
        if _name not in ('lgb', 'xgb', 'rf', 'cat', 'histgb'):
            continue  # np. 'long_spec' - dedykowany booster, poza porownaniem WFV algorytmow
        logger.info(f'WFV {_name.upper()} ({len(MODEL_FEATURES[_name])} cech)...')
        _wf[_name] = walk_forward_eval(df_wf, _name, MODEL_FEATURES[_name])
        r = _wf[_name]
        logger.info(f'  {_name.upper()} WF: acc={r["wf_accuracy"]:.4f} prec={r["wf_precision"]:.4f} '
                    f'rec={r["wf_recall"]:.4f} f1={r["wf_f1"]:.4f} (folds={r["wf_n_folds"]})')
    _best_name, _best_reason = (None, None)
    if len(_wf) > 1:
        _best_name, _best_reason = select_best_model(_wf, logger)

    # Split chronologiczny - NO SHUFFLE (wspolny podzial wierszy dla wszystkich modeli)
    split_idx = int(len(df) * (1 - VAL_RATIO))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    # Domyslny label (glowny, 48h/4x/1x) - uzywany przez wszystkie modele
    # OPROCZ specjalistow z MODEL_LABEL_COLUMN (audyt 2026-07-04: lgb_fast24h
    # uzywa label_fast24h, cat_sharp6x uzywa label_sharp6x).
    y_train = train['label_long'].values  # 0=neutral 1=long 2=short (3-klasowy, audyt 2026-07-03)
    y_val = val['label_long'].values

    logger.info(f'Train: {len(train)} | Val: {len(val)}')
    logger.info(f'Train class balance: NEUTRAL={ (y_train==0).mean()*100:.1f}% LONG={(y_train==1).mean()*100:.1f}% SHORT={(y_train==2).mean()*100:.1f}%')
    logger.info(f'Val class balance:   NEUTRAL={ (y_val==0).mean()*100:.1f}% LONG={(y_val==1).mean()*100:.1f}% SHORT={(y_val==2).mean()*100:.1f}%')

    def _model_xy(name):
        """Kazdy model dostaje SWOJ podzbior kolumn (MODEL_FEATURES) i wlasny scaler."""
        feats = MODEL_FEATURES[name]
        sc = StandardScaler()
        Xtr = sc.fit_transform(train[feats].values)
        Xva = sc.transform(val[feats].values)
        return Xtr, Xva, sc, feats

    def _model_y(name):
        """Kazdy model moze miec SWOJ label (MODEL_LABEL_COLUMN, audyt 2026-07-04)
        - domyslnie label_long, specjalisci uzywaja innego okna/progu TP."""
        col = MODEL_LABEL_COLUMN.get(name, 'label_long')
        yt = train[col].values
        yv = val[col].values
        if col != 'label_long':
            logger.info(f'  {name}: label={col} | balance NEUTRAL={(yt==0).mean()*100:.1f}% '
                        f'LONG={(yt==1).mean()*100:.1f}% SHORT={(yt==2).mean()*100:.1f}%')
        return yt, yv

    def _multiclass_metrics(y_true, y_pred):
        """Metryki ogolne (macro) + per-klasa (long=1, short=2) - odpowiada
        na pytanie 'ktore dane najtrafniejsze na short, ktore na long'."""
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        prec_pc = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        rec_pc = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        f1_pc = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        return {
            'accuracy': acc, 'precision': prec_macro, 'recall': rec_macro, 'f1': f1_macro,
            'precision_long': float(prec_pc[1]), 'recall_long': float(rec_pc[1]), 'f1_long': float(f1_pc[1]),
            'precision_short': float(prec_pc[2]), 'recall_short': float(rec_pc[2]), 'f1_short': float(f1_pc[2]),
        }

    # jak w TST - neutral bazowo, long/short podbite (mniejszosciowe klasy).
    # `class_weights` (audyt 2026-07-05, panel 'Trening niestandardowy') -
    # opcjonalne nadpisanie z zewnatrz (np. {0:1.0,1:3.0,2:2.0} - asymetryczne
    # wzmocnienie LONG vs SHORT), domyslnie te same wartosci co zawsze.
    _CW3 = class_weights or {0: 1.0, 1: 2.5, 2: 2.5}

    results = {}

    # === RANDOM FOREST ===
    if 'rf' in _train_names:
        logger.info(f"Training RandomForest ({len(MODEL_FEATURES['rf'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('rf')
        yt, yv = _model_y('rf')
        rf = RandomForestClassifier(
            # n_estimators 300->200: przy 3M+ probek i class_weight='balanced' 300 drzew
            # thrashowalo pamiec (683MB wolne, swap 5.7GB, utkniete >1h bez postepu)
            # max_features=0.7 (2026-08-05, rptr): jawnie, zamiast domyslnego 'sqrt' -
            # RF ma widziec wiecej cech na splicie niz ET (odwrotnie niz ET).
            n_estimators=200, max_depth=12, min_samples_leaf=20, max_features=0.7,
            random_state=42, n_jobs=-1, class_weight=_CW3,
        )
        rf.fit(Xtr, yt)
        m = _multiclass_metrics(yv, rf.predict(Xva))
        logger.info(f"  RF: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['rf'] = {
            'model': rf, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === LIGHTGBM ===
    if 'lgb' in _train_names:
        logger.info(f"Training LightGBM ({len(MODEL_FEATURES['lgb'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('lgb')
        yt, yv = _model_y('lgb')
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            min_child_samples=20, num_leaves=31,
            objective='multiclass', num_class=3,
            random_state=42, n_jobs=-1, verbose=-1, class_weight=_CW3,
        )
        lgb_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, lgb_model.predict(Xva))
        logger.info(f"  LGB: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['lgb'] = {
            'model': lgb_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === XGBOOST ===
    if 'xgb' in _train_names:
        logger.info(f"Training XGBoost ({len(MODEL_FEATURES['xgb'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('xgb')
        yt, yv = _model_y('xgb')
        # scale_pos_weight nie dziala dla multiclass - uzywamy sample_weight z _CW3
        sample_weight = np.array([_CW3[int(v)] for v in yt])
        xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            objective='multi:softprob', num_class=3,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
        )
        xgb_model.fit(Xtr, yt, sample_weight=sample_weight)
        m = _multiclass_metrics(yv, xgb_model.predict(Xva))
        logger.info(f"  XGB: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['xgb'] = {
            'model': xgb_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === CATBOOST ===
    if 'cat' in _train_names:
        logger.info(f"Training CatBoost ({len(MODEL_FEATURES['cat'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('cat')
        yt, yv = _model_y('cat')
        cat_model = CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.05, random_state=42,
            verbose=False, loss_function='MultiClass', classes_count=3,
            class_weights=[1.0, 2.5, 2.5],
            thread_count=int(os.environ.get("HAI_TRAIN_THREADS", "-1")),
        )
        cat_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, cat_model.predict(Xva).astype(int).ravel())
        logger.info(f"  CAT: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['cat'] = {
            'model': cat_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === HISTGRADIENTBOOSTING ===
    if 'histgb' in _train_names:
        logger.info(f"Training HistGradientBoosting ({len(MODEL_FEATURES['histgb'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('histgb')
        yt, yv = _model_y('histgb')
        histgb_model = HistGradientBoostingClassifier(
            max_iter=400, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight=_CW3,
        )
        histgb_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, histgb_model.predict(Xva))
        logger.info(f"  HISTGB: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['histgb'] = {
            'model': histgb_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === EXTRA TREES (audyt 2026-07-05) - druga rodzina bagging obok RF,
    # losuje TEZ prog podzialu (nie tylko cechy) - test hipotezy dekorelacji
    # glosow po korelacji z ostatniego backtestu (LGB/XGB/HistGB niemal
    # identyczne, r=0.99+; RF jedyny realnie zroznicowany glos). ===
    if 'et' in _train_names:
        from sklearn.ensemble import ExtraTreesClassifier
        logger.info(f"Training ExtraTrees ({len(MODEL_FEATURES['et'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('et')
        yt, yv = _model_y('et')
        # 2026-08-05 (rptr, diagnoza "ET/RF milcza"): ET mial IDENTYCZNE hiperparametry
        # co RF (zero dywersyfikacji - sam problem co z cechami przed dzisiejszym
        # portem). min_samples_leaf 20->10 (ET z definicji ma byc "szumny", nie
        # konserwatywny jak RF), max_features=0.6 (mniej cech na split -> wiecej
        # wariancji miedzy drzewami), bootstrap=False jawnie (klasyczny ET - kazde
        # drzewo widzi WSZYSTKIE probki, tylko losowe progi/cechy dekoruluja),
        # n_estimators 200->300 (wiecej drzew, zeby usredniac ten dodatkowy szum).
        et_model = ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10, max_features=0.6,
            bootstrap=False, random_state=42, n_jobs=-1, class_weight=_CW3,
        )
        et_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, et_model.predict(Xva))
        logger.info(f"  ET: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['et'] = {
            'model': et_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === GRADIENT BOOSTING sklearn (audyt 2026-07-05) - kolejna odmiana GB,
    # class_weight niewspierany bezposrednio -> sample_weight z _CW3. ===
    if 'gb' in _train_names:
        from sklearn.ensemble import GradientBoostingClassifier
        logger.info(f"Training GradientBoosting ({len(MODEL_FEATURES['gb'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('gb')
        yt, yv = _model_y('gb')
        gb_sw = np.array([_CW3[int(v)] for v in yt])
        gb_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42,
        )
        gb_model.fit(Xtr, yt, sample_weight=gb_sw)
        m = _multiclass_metrics(yv, gb_model.predict(Xva))
        logger.info(f"  GB: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['gb'] = {
            'model': gb_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === ADABOOST (audyt 2026-07-05) - genuinie inny mechanizm (wazenie
    # probek po bledzie, nie gradienty). sample_weight z _CW3 w .fit(). ===
    if 'ada' in _train_names:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        logger.info(f"Training AdaBoost ({len(MODEL_FEATURES['ada'])} cech)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('ada')
        yt, yv = _model_y('ada')
        ada_sw = np.array([_CW3[int(v)] for v in yt])
        ada_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200,
            learning_rate=0.5, random_state=42,
        )
        ada_model.fit(Xtr, yt, sample_weight=ada_sw)
        m = _multiclass_metrics(yv, ada_model.predict(Xva))
        logger.info(f"  ADA: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['ada'] = {
            'model': ada_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === LONG SPECIALIST (audyt 2026-07-04) - dedykowany booster, LGB-based,
    # podbita waga klasy LONG (_CW3_LONG_SPEC), zestaw cech oczyszczony z
    # tych ktore w analizie IS_LONG-vs-IS_SHORT wyszly SHORT-owe ===
    if 'long_spec' in _train_names:
        logger.info(f"Training LONG SPECIALIST ({len(MODEL_FEATURES['long_spec'])} cech, waga LONG={_CW3_LONG_SPEC[1]})...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('long_spec')
        yt, yv = _model_y('long_spec')
        long_spec_model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            min_child_samples=20, num_leaves=31,
            objective='multiclass', num_class=3,
            random_state=42, n_jobs=-1, verbose=-1, class_weight=_CW3_LONG_SPEC,
        )
        long_spec_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, long_spec_model.predict(Xva))
        logger.info(f"  LONG_SPEC: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['long_spec'] = {
            'model': long_spec_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()  # audyt 2026-07-04 - ograniczenie narastania RAM miedzy modelami

    # === LGB FAST24H (audyt 2026-07-04) - drugi LGB, TEN SAM algorytm i
    # cechy co glowny LGB, ale label z 24h lookahead zamiast 48h - lapie
    # szybsze/bardziej natychmiastowe ruchy. Test "diversity of error
    # structure" przez rozny CEL treningu, nie tylko rozne cechy. ===
    if 'lgb_fast24h' in _train_names:
        logger.info(f"Training LGB FAST24H ({len(MODEL_FEATURES['lgb_fast24h'])} cech, label=label_fast24h)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('lgb_fast24h')
        yt, yv = _model_y('lgb_fast24h')
        lgb_fast_model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            min_child_samples=20, num_leaves=31,
            objective='multiclass', num_class=3,
            random_state=42, n_jobs=-1, verbose=-1, class_weight=_CW3,
        )
        lgb_fast_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, lgb_fast_model.predict(Xva))
        logger.info(f"  LGB_FAST24H: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['lgb_fast24h'] = {
            'model': lgb_fast_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === CAT SHARP6X (audyt 2026-07-04) - drugi CAT, TEN SAM algorytm i
    # cechy co glowny CAT (tylko Core-12), ale label z TP=6xATR zamiast 4x -
    # ignoruje slabe/srednie ruchy, uczy sie tylko na bardzo zdecydowanych. ===
    if 'cat_sharp6x' in _train_names:
        logger.info(f"Training CAT SHARP6X ({len(MODEL_FEATURES['cat_sharp6x'])} cech, label=label_sharp6x)...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy('cat_sharp6x')
        yt, yv = _model_y('cat_sharp6x')
        cat_sharp_model = CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.05, random_state=42,
            verbose=False, loss_function='MultiClass', classes_count=3,
            class_weights=[1.0, 2.5, 2.5],
        )
        cat_sharp_model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, cat_sharp_model.predict(Xva).astype(int).ravel())
        logger.info(f"  CAT_SHARP6X: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results['cat_sharp6x'] = {
            'model': cat_sharp_model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === TEST SYSTEMATYCZNY: RF/XGB/HISTGB na obu wariantach horyzontu
    # (audyt 2026-07-04) - czy zmiana labelu pomaga rowniez pozostalym
    # 3 algorytmom, tak jak pomogla LGB (fast24h) i CAT (sharp6x). ===
    for _rname in ('rf_fast24h', 'rf_sharp6x'):
        if _rname not in _train_names:
            continue
        _lbl = MODEL_LABEL_COLUMN[_rname]
        logger.info(f"Training RF {_rname.upper()} ({len(MODEL_FEATURES[_rname])} cech, label={_lbl})...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy(_rname)
        yt, yv = _model_y(_rname)
        _model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=-1, class_weight=_CW3,
        )
        _model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, _model.predict(Xva))
        logger.info(f"  {_rname.upper()}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results[_rname] = {
            'model': _model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    for _xname in ('xgb_fast24h', 'xgb_sharp6x'):
        if _xname not in _train_names:
            continue
        _lbl = MODEL_LABEL_COLUMN[_xname]
        logger.info(f"Training XGB {_xname.upper()} ({len(MODEL_FEATURES[_xname])} cech, label={_lbl})...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy(_xname)
        yt, yv = _model_y(_xname)
        sample_weight = np.array([_CW3[int(v)] for v in yt])
        _model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            objective='multi:softprob', num_class=3,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
        )
        _model.fit(Xtr, yt, sample_weight=sample_weight)
        m = _multiclass_metrics(yv, _model.predict(Xva))
        logger.info(f"  {_xname.upper()}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results[_xname] = {
            'model': _model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    for _hname in ('histgb_fast24h', 'histgb_sharp6x'):
        if _hname not in _train_names:
            continue
        _lbl = MODEL_LABEL_COLUMN[_hname]
        logger.info(f"Training HISTGB {_hname.upper()} ({len(MODEL_FEATURES[_hname])} cech, label={_lbl})...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy(_hname)
        yt, yv = _model_y(_hname)
        _model = HistGradientBoostingClassifier(
            max_iter=400, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight=_CW3,
        )
        _model.fit(Xtr, yt)
        m = _multiclass_metrics(yv, _model.predict(Xva))
        logger.info(f"  {_hname.upper()}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results[_hname] = {
            'model': _model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # === GENERYCZNA PETLA: skrajnosci horyzontu (fast6h/wide96h) na
    # wszystkich 5 algorytmach (audyt 2026-07-04). Zamiast kopiowac kolejne
    # prawie-identyczne bloki (jak powyzej dla fast24h/sharp6x), jeden
    # builder per algorytm bazowy + petla po HORIZON_SWEEP_NAMES. ===
    def _build_lgb():
        return lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            min_child_samples=20, num_leaves=31,
            objective='multiclass', num_class=3,
            random_state=42, n_jobs=-1, verbose=-1, class_weight=_CW3,
        )
    def _build_rf():
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=-1, class_weight=_CW3,
        )
    def _build_xgb():
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            objective='multi:softprob', num_class=3,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
        )
    def _build_cat():
        return CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.05, random_state=42,
            verbose=False, loss_function='MultiClass', classes_count=3,
            class_weights=[1.0, 2.5, 2.5],
        )
    def _build_histgb():
        return HistGradientBoostingClassifier(
            max_iter=400, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight=_CW3,
        )
    _ALGO_BUILDERS = {'lgb': _build_lgb, 'rf': _build_rf, 'xgb': _build_xgb,
                       'cat': _build_cat, 'histgb': _build_histgb}
    _NEEDS_SAMPLE_WEIGHT = {'xgb'}  # .fit() nie ma class_weight, trzeba sample_weight z _CW3

    for _hname in HORIZON_SWEEP_NAMES:
        if _hname not in _train_names:
            continue
        _algo = _hname.rsplit('_', 1)[0]
        _lbl = MODEL_LABEL_COLUMN[_hname]
        logger.info(f"Training {_algo.upper()} {_hname.upper()} ({len(MODEL_FEATURES[_hname])} cech, label={_lbl})...")
        t0 = time.time()
        Xtr, Xva, sc, feats = _model_xy(_hname)
        yt, yv = _model_y(_hname)
        _model = _ALGO_BUILDERS[_algo]()
        if _algo in _NEEDS_SAMPLE_WEIGHT:
            sample_weight = np.array([_CW3[int(v)] for v in yt])
            _model.fit(Xtr, yt, sample_weight=sample_weight)
        else:
            _model.fit(Xtr, yt)
        _pred = _model.predict(Xva)
        if _algo == 'cat':
            _pred = _pred.astype(int).ravel()
        m = _multiclass_metrics(yv, _pred)
        logger.info(f"  {_hname.upper()}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                    f"| L:prec={m['precision_long']:.3f} S:prec={m['precision_short']:.3f} ({time.time()-t0:.0f}s)")
        results[_hname] = {
            'model': _model, 'scaler': sc, **m,
            'samples': len(Xtr), 'features': feats,
            'trained_at': datetime.now(timezone.utc).isoformat(),
        }
        del Xtr, Xva; gc.collect()

    # Dopisz metryki walk-forward + oznacz zwyciezce
    for _n in results:
        if _n in _wf:
            results[_n].update({k: v for k, v in _wf[_n].items()})
            results[_n]['is_wf_best'] = (_n == _best_name)
    results['_wf_best'] = _best_name
    results['_wf_reason'] = _best_reason
    logger.info(f'Walk-forward zwyciezca: {_best_name} ({_best_reason})')

    return results


def save_models(results: Dict, suffix: str = '_NEW'):
    """Zapisuje modele z suffixem (default _NEW dla bezpiecznego promote)."""
    for name, data in results.items():
        if name.startswith('_') or not isinstance(data, dict) or 'model' not in data:
            continue  # pomijaj metadane WF (_wf_best, _wf_reason)
        out_path = MODELS_DIR / f'{name}{suffix}.pkl'
        joblib.dump(data, out_path)
        logger.info(f'Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)')


def train_and_save(suffix: str = '_NEW', only: list = None,
                    class_weights: "dict | None" = None) -> Dict:
    """Glowna funkcja - trenuj (domyslnie wszystkie 5 standardowe, albo tylko
    `only` - lista nazw modeli np. ['lgb_fast24h','rf_h72'] - audyt 2026-07-05,
    panel 'Trening niestandardowy' w AI Learning) i zapisz. `class_weights`
    (opcjonalnie) - nadpisuje domyslne wagi klas {0:1.0,1:2.5,2:2.5}."""
    t0 = time.time()
    _instance = Path(__file__).resolve().parent.parent.name.replace("HAI_", "")
    _ver = "ver.10 Final" if _instance == "EPV" else "ver.10f"
    logger.info(f'=== HAI_{_instance} ML Trainer {_ver} (OPTIMIZED for 6 cores) ===')
    logger.info(f'Symboli: {len(SYMBOLS)} | Lookahead: {LOOKAHEAD_BARS}h | TP/SL: {TP_ATR_MULT}/{SL_ATR_MULT} ATR')
    logger.info(f'Parallel workers: {N_PARALLEL_SYMBOLS}')

    df = build_dataset()
    logger.info(f'Dataset total: {len(df):,} samples')

    results = train_models(df, only=only, class_weights=class_weights)
    save_models(results, suffix=suffix)

    elapsed = time.time() - t0
    summary = {
        'elapsed_sec': round(elapsed, 1),
        'samples': len(df),
        'models': {name: {k: v for k, v in r.items() if k not in ('model', 'scaler')}
                   for name, r in results.items() if isinstance(r, dict)},
    }
    logger.info(f'=== KONIEC ({elapsed:.0f}s = {elapsed / 60:.1f} min) ===')
    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    summary = train_and_save(suffix='_NEW')
    print(json.dumps(summary, indent=2))

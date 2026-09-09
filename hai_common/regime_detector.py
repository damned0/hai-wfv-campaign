# ===========================================
# HAI_EPV Engine ver.10 Final — core/regime_detector.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: detect_from_closes() (HMM Viterbi, 0=trend_following/
# 1=mean_reversion/2=high_volatility), uzywane przez ai_strategy.py (live)
# do regime-blended wag ensemble - core/backtester.py ma OSOBNA,
# zwektoryzowana logike ktora regime=0 blokuje calkowicie.
# ===========================================
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

try:
    from .config import BASE_DIR
    MODELS_DIR = BASE_DIR / "data" / "models"
except Exception:
    MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# KONFIGURACJA
# ============================================
N_REGIMES = 3  # 0=trend, 1=mean-reversion, 2=high-vol
MIN_TRAIN_SAMPLES = 500
HMM_RANDOM_STATE = 42
HMM_N_ITER = 100

# Features do HMM (normalizowane)
REGIME_FEATURES = [
    "volatility_20",     # 20-bar std dev of returns
    "trend_strength",    # ADX(14) / 100
    "volume_intensity",  # avg volume / 50-bar avg
    "rsi_extreme",       # |RSI - 50| / 50
    "ret_mean_20",       # 20-bar mean return
]

# Nazwy re?im?w dla czytelno?ci
REGIME_NAMES = {
    0: "trend_following",
    1: "mean_reversion",
    2: "high_volatility",
}

# ============================================
# REGIME DETECTOR CLASS
# ============================================

class RegimeDetector:
    """Detekcja re?imu rynkowego u?ywaj?c Hidden Markov Model."""

    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler_params = None  # (mean, std) dla normalizacji
        self.is_trained = False
        self.current_regime = 0
        self.regime_probabilities = np.ones(n_regimes) / n_regimes
        self.time_in_regime = 0  # liczba ?wiec w obecnym re?imie
        self._load_model()

    # ?????????????????????????????????????????????????????
    # Feature Engineering dla HMM
    # ?????????????????????????????????????????????????????

    def _build_features(self, ohlcv_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Buduje cechy z DataFrame OHLCV (kolumny: timestamp, open, high, low, close, volume).
        Zwraca numpy array [n_samples, n_features].
        """
        if len(ohlcv_df) < 50:
            return None

        closes = ohlcv_df["close"].values.astype(np.float64)
        highs = ohlcv_df["high"].values
        lows = ohlcv_df["low"].values
        volumes = ohlcv_df["volume"].values

        # Returns i volatility
        returns = np.diff(closes) / closes[:-1]
        returns = np.insert(returns, 0, 0.0)

        # 1. Volatility 20-bar
        vol_20 = pd.Series(returns).rolling(20).std().fillna(0).values

        # 2. Trend strength (ADX proxy przez zakres high-low vs close)
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        tr[0] = highs[0] - lows[0]
        atr_14 = pd.Series(tr).rolling(14).mean().fillna(tr[0]).values
        # Directional movement proxy
        up_move = np.maximum(highs - np.roll(highs, 1), 0)
        down_move = np.maximum(np.roll(lows, 1) - lows, 0)
        smooth_up = pd.Series(up_move).rolling(14).mean().fillna(0).values
        smooth_down = pd.Series(down_move).rolling(14).mean().fillna(0).values
        pdi = np.where(atr_14 > 0, smooth_up / atr_14 * 100, 0)
        mdi = np.where(atr_14 > 0, smooth_down / atr_14 * 100, 0)
        dx = np.where((pdi + mdi) > 0, np.abs(pdi - mdi) / (pdi + mdi) * 100, 0)
        adx_14 = pd.Series(dx).rolling(14).mean().fillna(0).values
        trend_strength = adx_14 / 100.0

        # 3. Volume intensity
        avg_vol_50 = pd.Series(volumes).rolling(50).mean().fillna(volumes[0]).values
        volume_intensity = np.where(avg_vol_50 > 0, volumes / avg_vol_50, 1.0)

        # 4. RSI extreme
        rsi_14 = self._calc_rsi(closes, 14)
        rsi_extreme = np.abs(rsi_14 - 50) / 50.0

        # 5. Return mean 20-bar
        ret_mean_20 = pd.Series(returns).rolling(20).mean().fillna(0).values

        features = np.column_stack([
            vol_20,
            trend_strength,
            volume_intensity,
            rsi_extreme,
            ret_mean_20,
        ])
        return features.astype(np.float64)

    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Wilder's RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.zeros_like(prices)
        if len(gains) < period:
            return rsi + 50.0
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        return rsi

    # ?????????????????????????????????????????????????????
    # Trening HMM
    # ?????????????????????????????????????????????????????

    def train(self, ohlcv_df: pd.DataFrame) -> bool:
        """Trenuje HMM na danych historycznych."""
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.error("hmmlearn nie zainstalowane. pip install hmmlearn")
            return False

        features = self._build_features(ohlcv_df)
        if features is None or len(features) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Za ma?o danych: {len(features) if features is not None else 0}/{MIN_TRAIN_SAMPLES}")
            return False

        # Normalizacja
        self.scaler_params = (features.mean(axis=0), features.std(axis=0) + 1e-8)
        X = (features - self.scaler_params[0]) / self.scaler_params[1]

        # Trenuj HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=HMM_N_ITER,
            random_state=HMM_RANDOM_STATE,
            verbose=False,
        )
        self.model.fit(X)

        # Przypisz nazwy stan?w na podstawie ?redniej volatility
        state_stats = []
        for i in range(self.n_regimes):
            mask = self.model.predict(X) == i
            if mask.sum() > 0:
                avg_vol = features[mask, 0].mean()
                state_stats.append((i, avg_vol))
        state_stats.sort(key=lambda x: x[1])
        self._state_mapping = {s[0]: idx for idx, s in enumerate(state_stats)}
        logger.info(f"HMM state mapping: {self._state_mapping}")
        logger.info(f"  Regime 0 (trend):     mean vol={state_stats[0][1]:.4f}")
        logger.info(f"  Regime 1 (mean-rev):  mean vol={state_stats[1][1]:.4f}")
        logger.info(f"  Regime 2 (high-vol):  mean vol={state_stats[2][1]:.4f}")

        self.is_trained = True
        self._save_model()
        return True

    # ?????????????????????????????????????????????????????
    # Predykcja online
    # ?????????????????????????????????????????????????????

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalizuje features u?ywaj?c parametr?w z treningu."""
        if self.scaler_params is None:
            return features
        mean, std = self.scaler_params
        return (features - mean) / std

    def predict_online(self, ohlcv_df: pd.DataFrame) -> int:
        """
        Przewiduje obecny re?im na podstawie ostatnich 50 ?wiec.
        Zwraca 0, 1, lub 2.
        """
        if not self.is_trained or self.model is None:
            logger.debug("RegimeDetector: model nie wytrenowany, zwracam regime=0 (trend)")
            self.current_regime = 0
            self.regime_probabilities = np.ones(self.n_regimes) / self.n_regimes
            return 0

        features = self._build_features(ohlcv_df)
        if features is None or len(features) < 20:
            logger.debug(f"RegimeDetector: za ma?o danych ({len(features) if features is not None else 0}), zwracam poprzedni re?im ({self.current_regime})")
            return self.current_regime

        # U?yj ostatniego wiersza (bie??ca ?wieca)
        X = self._normalize_features(features)
        state = self.model.predict(X)[-1]

        # Mapuj stan HMM na nasze etykiety
        self.current_regime = self._state_mapping.get(state, state)

        # Oblicz prawdopodobie?stwa
        try:
            self.regime_probabilities = self.model.predict_proba(X)[-1]
        except Exception:
            self.regime_probabilities = np.ones(self.n_regimes) / self.n_regimes

        # Aktualizuj czas w re?imie
        self.time_in_regime += 1

        return self.current_regime

    def predict_proba(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """
        Zwraca prawdopodobie?stwa [P_regime_0, P_regime_1, P_regime_2].
        """
        if not self.is_trained or self.model is None:
            return self.regime_probabilities

        features = self._build_features(ohlcv_df)
        if features is None:
            return self.regime_probabilities

        X = self._normalize_features(features)
        try:
            return self.model.predict_proba(X)[-1]
        except Exception:
            return self.regime_probabilities

    def predict_next_regime(self, ohlcv_df: pd.DataFrame) -> int:
        """
        Przewiduje NAJBARDZIEJ PRAWDOPODOBNY nast?pny re?im
        na podstawie macierzy przej?cia HMM.
        """
        if not self.is_trained or self.model is None:
            return self.current_regime

        features = self._build_features(ohlcv_df)
        if features is None:
            return self.current_regime

        X = self._normalize_features(features)
        try:
            proba = self.model.predict_proba(X)[-1]  # P(state | features)
        except Exception:
            return self.current_regime

        transmat = self.model.transmat_  # macierz przej?cia
        if transmat is None:
            return self.current_regime

        # P(next_state) = sum_i P(current=i) * transmat[i, :]
        next_proba = proba @ transmat

        # Mapuj na nasze etykiety
        next_state = np.argmax(next_proba)
        return self._state_mapping.get(next_state, next_state)

    # ?????????????????????????????????????????????????????
    # Status dla API
    # ?????????????????????????????????????????????????????

    def get_status(self) -> Dict:
        """Zwraca status detektora dla endpointu /regime/status."""
        regime_name = REGIME_NAMES.get(self.current_regime, f"unknown_{self.current_regime}")
        return {
            "trained": self.is_trained,
            "current_regime": self.current_regime,
            "regime_name": regime_name,
            "regime_probabilities": {
                REGIME_NAMES.get(i, f"state_{i}"): round(float(p), 4)
                for i, p in enumerate(self.regime_probabilities)
            },
            "time_in_regime": self.time_in_regime,
            "n_regimes": self.n_regimes,
            "features": REGIME_FEATURES,
        }

    # ?????????????????????????????????????????????????????
    # Persistence
    # ?????????????????????????????????????????????????????

    def _model_path(self) -> Path:
        return MODELS_DIR / "regime_hmm.pkl"

    def _save_model(self):
        """Zapisuje model HMM i parametry."""
        if self.model is None:
            return
        try:
            joblib.dump(
                {
                    "model": self.model,
                    "scaler_params": self.scaler_params,
                    "state_mapping": self._state_mapping,
                    "n_regimes": self.n_regimes,
                },
                self._model_path(),
            )
            logger.info(f"Regime HMM zapisany: {self._model_path()}")
        except Exception as e:
            logger.error(f"Blad zapisu HMM: {e}")

    def _load_model(self):
        """Wczytuje model HMM je?li istnieje."""
        path = self._model_path()
        if not path.exists():
            logger.info("Brak zapisanego modelu HMM ? wymagany trening")
            return
        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler_params = data["scaler_params"]
            self._state_mapping = data.get("state_mapping", {})
            self.n_regimes = data.get("n_regimes", N_REGIMES)
            self.is_trained = True
            logger.info(f"Regime HMM wczytany: {path}")
        except Exception as e:
            logger.warning(f"Nie udalo sie wczytac HMM: {e}")

    # ?????????????????????????????????????????????????????
    # Trenuj na wielu symbolach
    # ?????????????????????????????????????????????????????

    def train_on_multiple(self, warehouse_path: str = "/root/ProjektHAI/data_warehouse/ohlcv/binance",
                           symbols: Optional[List[str]] = None,
                           tf: str = "1h",
                           days: int = 365) -> bool:
        """
        Pobiera dane z warehouse dla wielu symboli i trenuje HMM.
        U?ywa BTC jako g??wnego ?r?d?a + opcjonalnie ETH/SOL.
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]

        all_features = []
        wh = Path(warehouse_path)

        for sym in symbols:
            path = wh / tf / f"{sym}.parquet"
            if not path.exists():
                logger.warning(f"Brak pliku: {path}")
                continue

            try:
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")

                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
                if "timestamp" in df.columns:
                    df = df[df["timestamp"] >= cutoff]

                if len(df) < MIN_TRAIN_SAMPLES:
                    logger.warning(f"{sym}: za ma?o danych ({len(df)})")
                    continue

                features = self._build_features(df)
                if features is not None:
                    all_features.append(features)
                    logger.info(f"{sym}: {len(features)} pr?bek")
            except Exception as e:
                logger.error(f"Blad ladowania {sym}: {e}")

        if not all_features:
            logger.error("Brak danych do treningu HMM")
            return False

        # Po??cz wszystkie pr?bki (BTC ma priorytet = wi?cej danych)
        X_all = np.vstack(all_features)
        logger.info(f"??cznie pr?bek do HMM: {len(X_all)}")

        # Normalizacja i trening
        self.scaler_params = (X_all.mean(axis=0), X_all.std(axis=0) + 1e-8)
        X = (X_all - self.scaler_params[0]) / self.scaler_params[1]

        try:
            from hmmlearn import hmm
        except ImportError:
            logger.error("hmmlearn nie zainstalowane")
            return False

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=HMM_N_ITER,
            random_state=HMM_RANDOM_STATE,
            verbose=False,
        )
        self.model.fit(X)

        # Mapowanie stan?w
        state_stats = []
        for i in range(self.n_regimes):
            mask = self.model.predict(X) == i
            if mask.sum() > 0:
                avg_vol = X_all[mask, 0].mean()
                state_stats.append((i, avg_vol))
        state_stats.sort(key=lambda x: x[1])
        self._state_mapping = {s[0]: idx for idx, s in enumerate(state_stats)}

        self.is_trained = True
        self._save_model()
        logger.info(f"HMM wytrenowany na {len(symbols)} symbolach")
        logger.info(f"  Regime 0 (trend):     mean vol={state_stats[0][1]:.4f}")
        logger.info(f"  Regime 1 (mean-rev):  mean vol={state_stats[1][1]:.4f}")
        logger.info(f"  Regime 2 (high-vol):  mean vol={state_stats[2][1]:.4f}")
        return True


    def detect_from_closes(self, prices: List[float], volumes: List[float]) -> int:
        """
        Szybka detekcja reżimu z listy cen zamknięcia + wolumenów.
        Nie aktualizuje stanu — bezpieczna do wywołania per-symbol w live.
        high=low=close (proxy gdy brak pełnych danych OHLCV).
        """
        if not self.is_trained or self.model is None:
            return self.current_regime
        if len(prices) < 50:
            return self.current_regime
        closes = np.array(prices, dtype=np.float64)
        if volumes and len(volumes) >= len(closes):
            vols = np.array(volumes[-len(closes):], dtype=np.float64)
        elif volumes:
            vols = np.full(len(closes), float(np.mean(volumes)))
        else:
            vols = np.ones(len(closes))
        df = pd.DataFrame({
            "close": closes,
            "high": closes,
            "low": closes,
            "volume": vols,
        })
        # high=low=close powoduje zerowy ATR → ADX=NaN, ignorujemy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features = self._build_features(df)
        if features is None or len(features) < 20:
            return self.current_regime
        X = self._normalize_features(features)
        try:
            state = self.model.predict(X)[-1]
            return self._state_mapping.get(int(state), self.current_regime)
        except Exception:
            return self.current_regime


# ???????????????????????????????????????????
# SINGLETON
# ???????????????????????????????????????????

regime_detector = RegimeDetector()

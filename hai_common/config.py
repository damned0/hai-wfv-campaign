import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from hai_common._instance import detect_instance, instance_port

INSTANCE = detect_instance()

def _detect_base_dir() -> Path:
    env_dir = os.environ.get("HAI_INSTANCE_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    cwd = Path.cwd()
    if cwd.name.startswith("HAI_") and (cwd / ".env").exists():
        return cwd
    for p in Path(__file__).resolve().parents:
        if p.name.startswith("HAI_") and (p / ".env").exists():
            return p
    return cwd

BASE_DIR = _detect_base_dir()
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env.secrets")
load_dotenv(BASE_DIR / ".env", override=True)


class AIConfig:
    def __init__(self):
        self.enabled: bool = os.getenv("AI_ENABLED", "false").lower() == "true"
        self.learn_enabled: bool = os.getenv("AI_LEARN_ENABLED", "true").lower() == "true"
        self.trade_enabled: bool = os.getenv("AI_TRADE_ENABLED", "false").lower() == "true"
        self.mode: str = os.getenv("AI_MODE", "off")
        self.default_model: str = os.getenv("AI_DEFAULT_MODEL", "random_forest")
        self.confidence_min: float = float(os.getenv("AI_CONFIDENCE_MIN", "0.6"))
        self.auto_retrain: bool = os.getenv("AI_AUTO_RETRAIN", "true").lower() == "true"
        self.retrain_interval_h: int = int(os.getenv("AI_RETRAIN_INTERVAL_H", "24"))
        self.min_samples: int = int(os.getenv("AI_MIN_SAMPLES", "200"))
        self.lookback_days: int = int(os.getenv("AI_LOOKBACK_DAYS", "30"))
        self.timeframes: List[str] = os.getenv("AI_TIMEFRAMES", "1H,4H,1D").split(",")
        self.features: List[str] = os.getenv("AI_FEATURES", "rsi,ema,macd,volume").split(",")
        self.allow_long: bool = os.getenv("AI_ALLOW_LONG", "true").lower() == "true"
        self.allow_short: bool = os.getenv("AI_ALLOW_SHORT", "false").lower() == "true"
        self.use_as_filter: bool = os.getenv("AI_USE_AS_FILTER", "true").lower() == "true"
        self.override_strategy: bool = os.getenv("AI_OVERRIDE_STRATEGY", "false").lower() == "true"
        self.max_signals_daily: int = int(os.getenv("AI_MAX_SIGNALS_DAILY", "10"))
        self.signal_cooldown_min: int = int(os.getenv("AI_SIGNAL_COOLDOWN_MIN", "15"))
        self.save_model: bool = os.getenv("AI_SAVE_MODEL", "true").lower() == "true"
        self.meta_label_enabled: bool = os.getenv("META_LABEL_ENABLED", "false").lower() == "true"
        self.meta_label_threshold: float = float(os.getenv("META_LABEL_THRESHOLD", "0.5"))
        self.confidence_calib_enabled: bool = os.getenv("CONFIDENCE_CALIB_ENABLED", "false").lower() == "true"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def update(self, data: dict):
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)


class TradingConfig:
    def __init__(self):
        self.max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
        self.leverage: int = int(os.getenv("LEVERAGE", "5"))
        self.order_size_usdt: float = float(os.getenv("ORDER_SIZE_USDT", "10"))
        self.take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "5.0"))
        self.stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "3.0"))
        self.max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "50"))
        self.max_total_exposure: float = float(os.getenv("MAX_TOTAL_EXPOSURE", "500"))
        self.cooldown_min: int = int(os.getenv("TRADE_COOLDOWN_MIN", "5"))
        self.loop_interval_sec: int = int(os.getenv("LOOP_INTERVAL_SEC", "300"))
        # Koszty trades (wstawiane do pnl pozycji 2026-08-02). Zgodne z backtesterem.
        self.fee_taker: float = float(os.getenv("FEE_TAKER", "0.0006"))        # 0.06% opłata (open+close)
        self.slippage_entry: float = float(os.getenv("SLIPPAGE_ENTRY", "0.0005"))  # 0.05% slippage wejście
        self.slippage_exit: float = float(os.getenv("SLIPPAGE_EXIT", "0.0005"))    # 0.05% slippage wyjście

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def update(self, data: dict):
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)


class Config:
    def __init__(self):
        self.INSTANCE = INSTANCE
        self.BASE_DIR = BASE_DIR

        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_SECRET = os.getenv("BITGET_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
        self.BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

        self.MODE = os.getenv("MODE", "paper")
        self.TRADE_ENABLED = os.getenv("TRADE_ENABLED", "true").lower() == "true"

        self.DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "bitget")
        self.DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "ai_strategy")
        self.DEFAULT_SYMBOLS: List[str] = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "DOGE/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
            "MATIC/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "SUI/USDT:USDT",
        ]

        self.ai = AIConfig()
        self.trading = TradingConfig()
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", str(instance_port())))

    @property
    def AI_ENABLED(self) -> bool:
        return self.ai.enabled

    @AI_ENABLED.setter
    def AI_ENABLED(self, v: bool):
        self.ai.enabled = v

    @property
    def AI_LEARN_ENABLED(self) -> bool:
        return self.ai.learn_enabled

    @AI_LEARN_ENABLED.setter
    def AI_LEARN_ENABLED(self, v: bool):
        self.ai.learn_enabled = v

    @property
    def AI_TRADE_ENABLED(self) -> bool:
        return self.ai.trade_enabled

    @AI_TRADE_ENABLED.setter
    def AI_TRADE_ENABLED(self, v: bool):
        self.ai.trade_enabled = v

    @property
    def META_LABEL_ENABLED(self) -> bool:
        return self.ai.meta_label_enabled

    @META_LABEL_ENABLED.setter
    def META_LABEL_ENABLED(self, v: bool):
        self.ai.meta_label_enabled = v

    @property
    def META_LABEL_THRESHOLD(self) -> float:
        return self.ai.meta_label_threshold

    @META_LABEL_THRESHOLD.setter
    def META_LABEL_THRESHOLD(self, v: float):
        self.ai.meta_label_threshold = v

    @property
    def CONFIDENCE_CALIB_ENABLED(self) -> bool:
        return self.ai.confidence_calib_enabled

    @CONFIDENCE_CALIB_ENABLED.setter
    def CONFIDENCE_CALIB_ENABLED(self, v: bool):
        self.ai.confidence_calib_enabled = v

    @property
    def is_live(self) -> bool:
        return self.MODE == "live" and bool(self.BITGET_API_KEY)

    @property
    def effective_mode(self) -> str:
        return "live" if self.is_live else "paper"

    def get_exchange_keys(self, exchange: str) -> dict:
        return {
            "bitget": {"apiKey": self.BITGET_API_KEY,
                        "secret": self.BITGET_SECRET,
                        "password": self.BITGET_PASSPHRASE},
            "binance": {"apiKey": self.BINANCE_API_KEY,
                         "secret": self.BINANCE_SECRET},
        }.get(exchange, {})

    def get_available_exchanges(self) -> List[str]:
        result = ["bitget"]
        if self.BINANCE_API_KEY:
            result.append("binance")
        return result

    def to_dict(self) -> dict:
        return {
            "mode": self.effective_mode,
            "exchange": self.DEFAULT_EXCHANGE,
            "strategy": self.DEFAULT_STRATEGY,
            "ai": self.ai.to_dict(),
            "trading": self.trading.to_dict(),
        }


config = Config()

# Circuit breaker
config.MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT', '20.0'))
config.MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))
config.TRAILING_STOP_ENABLED = os.getenv('TRAILING_STOP_ENABLED', 'false').lower() == 'true'
config.TRAILING_DISTANCE_PCT = float(os.getenv('TRAILING_DISTANCE_PCT', '1.5'))
config.OI_FUNDING_ENABLED = os.getenv('OI_FUNDING_ENABLED', 'true').lower() == 'true'

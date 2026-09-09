# ===========================================
# HAI_EPV Engine ver.10 Final — core/logger_filter.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: konfiguracja osobnych logow (system.log/trading.log), rotacja
# (max 10MB x 5 plikow = 50MB kazdy).
# ===========================================
import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_LOG = LOG_DIR / "system.log"
TRADING_LOG = LOG_DIR / "trading.log"
APP_LOG = LOG_DIR / "app.log"   # backward compat (wszystko razem)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Komponenty traktowane jako "trading" — sygnały, pozycje, PnL
TRADING_COMPONENTS = {
    "core.engine",
    "core.risk",
    "strategies.base",
    "strategies.registry",
    "exchanges.bitget",
    "exchanges.registry",
    "core.backtester",
    "core.state",
}


class TradingFilter(logging.Filter):
    """Przepuszcza logi z komponentów handlowych."""

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(c) for c in TRADING_COMPONENTS)


class SystemFilter(logging.Filter):
    """Przepuszcza wszystko POZA logami handlowymi."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name.startswith(c) for c in TRADING_COMPONENTS)


def _make_rotating_handler(path: Path, filter_cls=None) -> logging.Handler:
    """Handler z rotacją: max 10MB × 5 plików."""
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,                # rotacja: x.log, x.log.1, ..., x.log.5
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    if filter_cls is not None:
        handler.addFilter(filter_cls())
    return handler


def setup_logging(level: str = "INFO"):
    """
    Konfiguruje logowanie:
    - logs/system.log    — start/stop, AI, watchdog, telegram, błędy
    - logs/trading.log   — sygnały, open/close, PnL, strategie
    - logs/app.log       — wszystko razem (backward compat)
    - stdout             — wszystko (dla systemd journalctl / docker)
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Wyczyść stare handlery (np. po reloadzie)
    for h in list(root.handlers):
        root.removeHandler(h)

    # 1) System log (wszystko poza trading)
    root.addHandler(_make_rotating_handler(SYSTEM_LOG, SystemFilter))

    # 2) Trading log (tylko engine/strategies/exchanges)
    root.addHandler(_make_rotating_handler(TRADING_LOG, TradingFilter))

    # 3) Backward compat — app.log ze wszystkim
    root.addHandler(_make_rotating_handler(APP_LOG))

    # UWAGA: nie dodajemy tu osobnego Console/StreamHandler(stderr) - supervisor.py
    # (_spawn) przekierowuje stdout=stderr=logs/app.log tego procesu, wiec
    # dodatkowy StreamHandler pisalby KAZDY komunikat DRUGI RAZ do tego samego
    # pliku (audyt 2026-07-05, znaleziono kazda linie zdublowana w app.log).

    # Tłumienie hałaśliwych bibliotek
    for noisy in ["urllib3", "ccxt", "websockets", "httpx", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging gotowy | system={SYSTEM_LOG} trading={TRADING_LOG} app={APP_LOG}"
    )
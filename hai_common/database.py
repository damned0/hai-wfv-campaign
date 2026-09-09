# ===========================================
# HAI_EPV Engine ver.10 Final — core/database.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: SQLAlchemy models — Position (features_json snapshot przy openie),
# SystemLog/TradingLog/AILog, engine z WAL mode (szybszy zapis concurrent).
# ===========================================
from sqlalchemy import (
    create_engine, event, Column, Integer,
    Float, String, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from pathlib import Path
import os

# ── IZOLACJA INSTANCJI (fix 2026-07-14) ──────────────────────────────────────
# BYLO: BASE_DIR = Path(__file__).resolve().parent.parent
#
# Ten plik lezy w hai_common/hai_common/, wiec parent.parent dawal ZAWSZE
# /root/ProjektHAI/hai_common — niezaleznie od instancji. Skutek: EPV, DEV, LAB,
# LIV i TST pisaly do JEDNEJ WSPOLNEJ bazy (hai_common/data/tai.db). Pozycje sie
# mieszaly, salda byly identyczne co do grosza, statystyki per instancja nie
# mialy sensu, a czyszczenie HAIs/HAI_*/data/tai.db nie robilo NIC (te pliki byly
# puste - prawdziwe dane szly gdzie indziej).
#
# Przed migracja do hai_common plik lezal w HAI_EPV/core/, gdzie parent.parent
# dawalo poprawnie katalog instancji. Migracja po cichu zlamala izolacje.
#
# TERAZ: ta sama detekcja co w config.py (HAI_INSTANCE_DIR -> cwd -> parents).
from .config import BASE_DIR   # noqa: E402  (config NIE importuje database - brak cyklu)

DB_PATH  = BASE_DIR / "data" / "tai.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# WAL mode — engine loop + HTTP requesty nie blokuja sie nawzajem
@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")
    dbapi_conn.execute("PRAGMA cache_size=-2000000")  # 2GB cache w RAM
    dbapi_conn.execute("PRAGMA temp_store=MEMORY")
    dbapi_conn.execute("PRAGMA mmap_size=30000000000")  # 30GB mmap
    dbapi_conn.execute("PRAGMA page_size=8192")
    dbapi_conn.execute("PRAGMA optimize")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class Position(Base):
    __tablename__ = "positions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(20), nullable=False, index=True)
    side          = Column(String(10), nullable=False)
    mode          = Column(String(10), default="paper")
    exchange      = Column(String(20), default="bitget")
    entry_price   = Column(Float, nullable=False)
    exit_price    = Column(Float, default=0.0)
    size_coins    = Column(Float, nullable=False)
    size_usdt     = Column(Float, nullable=False)
    leverage      = Column(Integer, default=5)
    pnl           = Column(Float, default=0.0)
    pnl_pct       = Column(Float, default=0.0)
    status        = Column(String(10), default="open")
    entry_time    = Column(DateTime, server_default=func.now())
    exit_time     = Column(DateTime, nullable=True)
    strategy      = Column(String(30), default="manual")
    reason        = Column(String(200), nullable=True)
    tp_price      = Column(Float, nullable=True)
    sl_price      = Column(Float, nullable=True)
    # NEW v6.0: snapshot 16 features w momencie openu (JSON string).
    # Pozwala na deterministyczny retrening bez parsowania reason.
    features_json = Column(Text, nullable=True)


class SystemLog(Base):
    __tablename__ = "system_logs"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    level     = Column(String(20), default="INFO")
    component = Column(String(30), default="core")
    message   = Column(Text, nullable=False)


class TradingLog(Base):
    __tablename__ = "trading_logs"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    level     = Column(String(20), default="INFO")
    symbol    = Column(String(20), nullable=True)
    action    = Column(String(20), nullable=True)
    message   = Column(Text, nullable=False)
    pnl       = Column(Float, default=0.0)
    mode      = Column(String(10), default="paper")


class AILog(Base):
    __tablename__ = "ai_logs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, server_default=func.now(), index=True)
    model_name    = Column(String(50), nullable=False)
    event         = Column(String(30), default="training")
    accuracy      = Column(Float, default=0.0)
    samples_count = Column(Integer, default=0)
    message       = Column(Text, nullable=True)


def init_db():
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate()

def _migrate():
    """Dodaje brakujace kolumny tp_price/sl_price do istniejacych baz."""
    from sqlalchemy import text
    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(positions)"))}
        if "tp_price" not in cols:
            conn.execute(text("ALTER TABLE positions ADD COLUMN tp_price Float"))
        if "sl_price" not in cols:
            conn.execute(text("ALTER TABLE positions ADD COLUMN sl_price Float"))


def get_session():
    return SessionLocal()

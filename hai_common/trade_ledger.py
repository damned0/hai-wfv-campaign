# ===========================================
# trade_ledger.py — centralna, append-only ksiegowosc trade'ow floty
# ===========================================
# Powstala po audycie ekonomicznym 2026-08-01: historia pozycji byla
# poszatkowana (bug "wspolnej bazy" 14.07, reset baz instancji 30.07) i nie
# bylo jednego trwalego zrodla prawdy o zamknietych trade'ach.
#
# Zasady:
#  - APPEND-ONLY: tabela `trades` przyjmuje wylacznie INSERT OR IGNORE,
#    nigdy UPDATE/DELETE. Zamkniety trade jest niezmienny.
#  - IDEMPOTENTNOSC: deduplikacja po (instance, symbol, entry_time) —
#    zapis tego samego trade'a N razy daje 1 wiersz. (Klucz naturalny;
#    position_id nie jest stabilny miedzy bazami po resetach).
#  - ZERO RYZYKA DLA TRADINGU: record_* nigdy nie rzuca wyjatku, krotkie
#    polaczenie per zapis, busy_timeout. Awaria ledgera = warning w logu.
#  - `opens` to biezacy SNAPSHOT otwartych pozycji (upsert/delete) —
#    nie jest czescia historii append-only, slzy podgladowi "co trwa".
#
# Sciezka bazy: HAI_LEDGER_DIR lub data_warehouse/ledger/trades_ledger.db
# ===========================================
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("/root/ProjektHAI/data_warehouse/ledger")

_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    instance     TEXT NOT NULL,
    position_id  INTEGER,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    mode         TEXT,
    exchange     TEXT,
    entry_price  REAL,
    exit_price   REAL,
    size_coins   REAL,
    size_usdt    REAL,
    leverage     INTEGER,
    pnl          REAL,
    pnl_pct      REAL,
    entry_time   TEXT NOT NULL,
    exit_time    TEXT,
    strategy     TEXT,
    reason       TEXT,
    features_json TEXT,
    source       TEXT NOT NULL DEFAULT 'hook',
    ledger_ts    TEXT NOT NULL,
    UNIQUE(instance, symbol, entry_time)
);
CREATE INDEX IF NOT EXISTS ix_trades_instance ON trades (instance);
CREATE INDEX IF NOT EXISTS ix_trades_exit ON trades (exit_time);
"""

_OPENS_DDL = """
CREATE TABLE IF NOT EXISTS opens (
    instance     TEXT NOT NULL,
    position_id  INTEGER,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    mode         TEXT,
    exchange     TEXT,
    entry_price  REAL,
    size_coins   REAL,
    size_usdt    REAL,
    leverage     INTEGER,
    entry_time   TEXT NOT NULL,
    strategy     TEXT,
    reason       TEXT,
    features_json TEXT,
    last_seen    TEXT NOT NULL,
    PRIMARY KEY (instance, symbol)
);
"""

_COLS = ("instance", "position_id", "symbol", "side", "mode", "exchange",
         "entry_price", "exit_price", "size_coins", "size_usdt", "leverage",
         "pnl", "pnl_pct", "entry_time", "exit_time", "strategy", "reason",
         "features_json")


def _ledger_path() -> Path:
    d = os.environ.get("HAI_LEDGER_DIR")
    return (Path(d) if d else _DEFAULT_DIR) / "trades_ledger.db"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ts(v) -> str:
    """Normalizacja datetime/str -> 'YYYY-MM-DD HH:MM:SS' (ucina mikrosekundy,
    zeby klucz deduplikacji byl stabilny miedzy zrodlami)."""
    if v is None or v == "":
        return ""
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    s = str(v).replace("T", " ").strip()
    return s[:19]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=3)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=3000")
    return con


def _init(con: sqlite3.Connection):
    for ddl in (_TRADES_DDL, _OPENS_DDL):
        for stmt in ddl.strip().split(";"):
            if stmt.strip():
                con.execute(stmt)


def _norm_instance(name: str) -> str:
    """'HAI_EPV' -> 'EPV'. Klucz deduplikacji musi byc spojny miedzy hookiem
    (config.INSTANCE='HAI_EPV') a collectorem/backfillem ('EPV')."""
    n = (name or "UNKNOWN").upper()
    return n[4:] if n.startswith("HAI_") else n


def _row_dict(pos, instance: str) -> dict:
    """Pozycja (obiekt ORM albo dict) -> plaski wiersz ledgera."""
    g = (lambda k: getattr(pos, k, None)) if not isinstance(pos, dict) else pos.get
    r = {c: g(c) for c in _COLS if c != "instance"}
    r["instance"] = _norm_instance(instance)
    r["entry_time"] = _ts(r.get("entry_time"))
    r["exit_time"] = _ts(r.get("exit_time"))
    return r


def record_closed(instance: str, pos, source: str = "hook") -> bool:
    """Zapisz ZAMKNIETA pozycje (append-only). True = wstawiono nowy wiersz.
    Nigdy nie rzuca wyjatku (hot path tradingu)."""
    try:
        r = _row_dict(pos, instance)
        if not r["symbol"] or not r["entry_time"]:
            logger.warning(f"ledger record_closed: brak symbol/entry_time: {r}")
            return False
        con = _connect(_ledger_path())
        try:
            _init(con)
            cur = con.execute(
                f"INSERT OR IGNORE INTO trades ({','.join(_COLS)}, source, ledger_ts)"
                f" VALUES ({','.join('?' * len(_COLS))}, ?, ?)",
                [r[c] for c in _COLS] + [source, _now()])
            con.execute("DELETE FROM opens WHERE instance=? AND symbol=?",
                        (r["instance"], r["symbol"]))
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"ledger record_closed FAILED (trading niedotkniety): {e}")
        return False


def record_open(instance: str, pos) -> bool:
    """Upsert OTWARTEJ pozycji do snapshotu `opens`. Nigdy nie rzuca."""
    try:
        r = _row_dict(pos, instance)
        if not r["symbol"] or not r["entry_time"]:
            return False
        con = _connect(_ledger_path())
        try:
            _init(con)
            con.execute(
                """INSERT INTO opens (instance, position_id, symbol, side, mode,
                      exchange, entry_price, size_coins, size_usdt, leverage,
                      entry_time, strategy, reason, features_json, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(instance, symbol) DO UPDATE SET
                      position_id=excluded.position_id, side=excluded.side,
                      entry_price=excluded.entry_price,
                      size_coins=excluded.size_coins, size_usdt=excluded.size_usdt,
                      leverage=excluded.leverage, reason=excluded.reason,
                      last_seen=excluded.last_seen""",
                (r["instance"], r["position_id"], r["symbol"], r["side"], r["mode"],
                 r["exchange"], r["entry_price"], r["size_coins"], r["size_usdt"],
                 r["leverage"], r["entry_time"], r["strategy"], r["reason"],
                 r["features_json"], _now()))
            con.commit()
            return True
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"ledger record_open FAILED (trading niedotkniety): {e}")
        return False


def collect_from_instance_db(db_path, instance: str, source: str,
                             ledger_path: Path = None, archive: bool = False) -> dict:
    """Collector/backfill: przeczytaj positions z bazy instancji READ-ONLY i
    dopisz brakujace (closed -> trades IGNORE, open -> opens upsert).
    archive=True pomija opens (zrodla archiwalne nie opisuja biezacego stanu).
    Zwraca statystyki. Bledy zwracane w dict, nie rzucane."""
    stats = {"instance": instance, "source": source, "db": str(db_path),
             "closed_seen": 0, "closed_new": 0, "open_seen": 0, "open_new": 0,
             "errors": []}
    db_path = Path(db_path)
    if not db_path.exists():
        stats["errors"].append(f"brak pliku {db_path}")
        return stats
    try:
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        src.row_factory = sqlite3.Row
        rows = [dict(r) for r in src.execute("SELECT * FROM positions")]
        src.close()
    except Exception as e:
        stats["errors"].append(f"odczyt {db_path}: {e}")
        return stats

    lp = ledger_path or _ledger_path()
    try:
        con = _connect(lp)
        try:
            _init(con)
            for row in rows:
                r = _row_dict(row, instance)
                if not r["symbol"] or not r["entry_time"]:
                    stats["errors"].append(f"pominieto wiersz bez symbol/entry_time: id={row.get('id')}")
                    continue
                if row.get("status") == "closed":
                    stats["closed_seen"] += 1
                    cur = con.execute(
                        f"INSERT OR IGNORE INTO trades ({','.join(_COLS)}, source, ledger_ts)"
                        f" VALUES ({','.join('?' * len(_COLS))}, ?, ?)",
                        [r[c] for c in _COLS] + [source, _now()])
                    if cur.rowcount > 0:
                        stats["closed_new"] += 1
                    con.execute("DELETE FROM opens WHERE instance=? AND symbol=?",
                                (r["instance"], r["symbol"]))
                elif row.get("status") == "open" and not archive:
                    stats["open_seen"] += 1
                    cur = con.execute(
                        """INSERT INTO opens (instance, position_id, symbol, side, mode,
                              exchange, entry_price, size_coins, size_usdt, leverage,
                              entry_time, strategy, reason, features_json, last_seen)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(instance, symbol) DO UPDATE SET
                              position_id=excluded.position_id, side=excluded.side,
                              entry_price=excluded.entry_price,
                              size_coins=excluded.size_coins,
                              size_usdt=excluded.size_usdt, leverage=excluded.leverage,
                              reason=excluded.reason, last_seen=excluded.last_seen""",
                        (r["instance"], r["position_id"], r["symbol"], r["side"],
                         r["mode"], r["exchange"], r["entry_price"], r["size_coins"],
                         r["size_usdt"], r["leverage"], r["entry_time"], r["strategy"],
                         r["reason"], r["features_json"], _now()))
                    stats["open_new"] += cur.rowcount > 0
            con.commit()
        finally:
            con.close()
    except Exception as e:
        stats["errors"].append(f"zapis ledgera: {e}")
    return stats


def summary(ledger_path: Path = None) -> dict:
    """Szybki podglad stanu ledgera (do weryfikacji/CLI)."""
    lp = ledger_path or _ledger_path()
    if not lp.exists():
        return {"exists": False, "path": str(lp)}
    con = sqlite3.connect(f"file:{lp}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        per_inst = [dict(r) for r in con.execute(
            "SELECT instance, COUNT(*) n, ROUND(SUM(pnl),2) pnl,"
            " MIN(entry_time) first, MAX(exit_time) last"
            " FROM trades GROUP BY instance ORDER BY instance")]
        per_src = [dict(r) for r in con.execute(
            "SELECT source, COUNT(*) n FROM trades GROUP BY source")]
        opens = [dict(r) for r in con.execute(
            "SELECT instance, COUNT(*) n FROM opens GROUP BY instance")]
        total = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    finally:
        con.close()
    return {"exists": True, "path": str(lp), "trades_total": total,
            "by_instance": per_inst, "by_source": per_src, "opens": opens}

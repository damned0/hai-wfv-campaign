# ===========================================
# HAI_EPV Engine ver.10 Final — core/app.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - create_app() - budowa FastAPI (routery, CORS, auth, static/templates)
# - lifespan() - start/stop silnika, watchdog, telegram
# - _print_startup_banner() - baner startowy (nazwa/wersja/liczba modeli i cech)
# ===========================================
import asyncio
import logging
import pty
import re
import secrets
import select
import signal
import subprocess
import termios
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Body, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware

import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BASE_DIR: najpierw z configu (instancja), potem fallback do __file__
from .config import config as _cfg
BASE_DIR = _cfg.BASE_DIR
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger    = logging.getLogger(__name__)
security  = HTTPBasic()

# === Auth ===
load_dotenv(Path(__file__).resolve().parents[2] / ".env.secrets")
load_dotenv(BASE_DIR / ".env")

def _raise(name):
    raise RuntimeError(f"Brak zmiennej środowiskowej {name} w .env — odmowa startu")

HAI_USER = os.getenv("HAI_USER") or _raise("HAI_USER")
HAI_PASS = os.getenv("HAI_PASS") or _raise("HAI_PASS")


def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, HAI_USER)
    correct_pass = secrets.compare_digest(credentials.password, HAI_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ─────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────

def _print_startup_banner():
    """Czytelny raport startowy (audyt 2026-07-05, na wyrazna prosbe -
    'pelny raport co sie wystartowalo'). Laduje ensemble wczesnie (zamiast
    leniwie przy pierwszym predict) zeby m.N/f.M byly znane od razu."""
    from .ensemble import ensemble
    from .ml_trainer import MODEL_FEATURES
    from .state import state
    if not ensemble.active:
        ensemble.load_models()
    n_models = len(ensemble.models)
    all_feats = set()
    for name in ensemble.models:
        all_feats |= set(MODEL_FEATURES.get(name, ensemble.feature_names.get(name, [])))
    n_feats = len(all_feats)
    instance = BASE_DIR.name.replace("HAI_", "")
    ver = "ver.10 Final" if instance == "EPV" else "ver.10f"
    banner = (
        "\n"
        "========================================\n"
        "  HAI Neural Trader\n"
        f"  HAI_{instance} m.{n_models} f.{n_feats} Engine {ver}\n"
        "========================================\n"
        "Engine starting"
    )
    logger.info(banner)
    state.add_log("system", "INFO", component="engine",
                  message=f"HAI Neural Trader | HAI_{instance} m.{n_models} f.{n_feats} Engine {ver} | Engine starting")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .state import state
    _print_startup_banner()
    try:
        from .config import config
        from .engine import engine

        # Engine ZAWSZE startuje
        asyncio.create_task(engine.start())

        # Watchdog ZAWSZE
        try:
            from .watchdog import watchdog
            asyncio.create_task(watchdog.start())
        except Exception as e:
            logger.warning(f"Watchdog start error: {e}")

        # Telegram ZAWSZE
        try:
            from .telegram import telegram
            asyncio.create_task(telegram.start())
            asyncio.create_task(telegram.daily_summary_loop())
        except Exception as e:
            logger.warning(f"Telegram start error: {e}")

        logger.info(f"Bootstrap done | AI_TRADE_ENABLED={config.AI_TRADE_ENABLED} "
                    f"| MODE={config.effective_mode}")
        logger.info("Engaged")
        state.add_log("system", "INFO", component="engine", message="Engaged")
    except Exception as e:
        logger.error(f"Lifespan start error: {e}", exc_info=True)

    yield

    logger.info("ProjektHAI zatrzymuje sie...")
    try:
        from .engine import engine
        await engine.stop()
    except Exception as e:
        logger.error(f"Engine stop error: {e}")


# ─────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="HAI_EPV",
        description="Trading AI by Hauzer & HyperAI",
        version="6.0.0",
        lifespan=lifespan,
    )

    # === Auth: strona logowania + sesje + role (wspólny moduł hai_auth) ===
    # Zastępuje globalny HTTP Basic. Basic zachowany jako fallback dla API.
    import sys
    sys.path.insert(0, str(BASE_DIR.parent.parent))
    import hai_auth
    hai_auth.install(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class UTF8Middleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if "text/html" in response.headers.get("content-type", ""):
                response.headers["content-type"] = "text/html; charset=utf-8"
            return response

    app.add_middleware(UTF8Middleware)

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    if _cfg.INSTANCE == "HAI_NL":
        _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-2]|\x1b[=>]|\r")
        # Komendy TUI/interaktywne — w web-terminalu (bez TTY) zwisaja do
        # timeoutu i zwracaja sciane ANSI-krakow. Odrzucamy od razu z podpowiedzia.
        _TUI_ALWAYS = ("vim", "vi", "nvim", "nano", "emacs", "less", "more",
                       "man", "watch", "top", "htop", "btop", "ranger", "mc",
                       "ssh", "screen", "su")
        _TUI_BARE = ("python", "python3", "node", "opencode")  # TUI tylko bez argumentow

        def _is_tui(command: str) -> bool:
            parts = command.split()
            if not parts:
                return False
            if "&" in command:                    # tlo — OK
                return False
            first = parts[0].split("/")[-1]
            if first in _TUI_ALWAYS:
                return True
            if first in _TUI_BARE and len(parts) == 1:
                return True
            if first == "opencode" and (len(parts) < 2 or parts[1] != "run"):
                return True
            if first == "tmux" and not (len(parts) >= 3 and parts[1] in ("new", "new-session") and "-d" in parts):
                return len(parts) == 1            # goły `tmux` = TUI; `tmux ls` itd. OK
            return False

        # ── Sesja terminala na PTY (2026-08-01, decyzja usera: "sesja jak w opencode")
        # Jeden dlugo zyjacy proces bash na pseudo-terminalu: cd, export, zmienne,
        # funkcje — wszystko trwale miedzy komendami. PTY (nie pipe), wiec ls/grep
        # moga uzywac kolorow (stripujemy ANSI przy odczycie). Timeout = SIGINT
        # (Ctrl+C) — sesja ZYJE dalej, nie zabijamy basha.
        class _TermSession:
            def __init__(self):
                self._lock = threading.Lock()
                self._start()

            def _start(self):
                self.master, slave = pty.openpty()
                self.proc = subprocess.Popen(
                    ["bash", "--norc", "-i"],
                    stdin=slave, stdout=slave, stderr=slave,
                    cwd=str(PROJECT_ROOT),
                    env={**os.environ, "TERM": "dumb", "PS1": "", "PS2": "",
                         "HAI_TERM": "1"},
                    preexec_fn=os.setsid, close_fds=True)
                os.close(slave)
                # wylacz echo komend na PTY (inaczej kazda komenda powtarzalaby sie w out)
                attrs = termios.tcgetattr(self.master)
                attrs[3] = attrs[3] & ~termios.ECHO
                termios.tcsetattr(self.master, termios.TCSANOW, attrs)
                self._drain(2.0)  # banner startowy basha

            def _drain(self, timeout: float) -> bytes:
                end = time.time() + timeout
                buf = b""
                while time.time() < end:
                    r, _, _ = select.select([self.master], [], [], 0.15)
                    if not r:
                        continue
                    try:
                        chunk = os.read(self.master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                return buf

            def run(self, command: str, timeout: int = 60) -> dict:
                with self._lock:
                    if self.proc.poll() is not None:   # bash umarl (np. `exit`) — nowa sesja
                        self._start()
                    mark = f"__HAI_RC_{int(time.time() * 1000) % 10 ** 9}__"
                    os.write(self.master, f"{command}; echo \"{mark}$?@$PWD\"\n".encode())
                    deadline = time.time() + timeout
                    buf = b""
                    timed_out = False
                    while mark.encode() not in buf:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            timed_out = True
                            os.killpg(self.proc.pid, signal.SIGINT)   # jak Ctrl+C
                            buf += self._drain(2.0)
                            break
                        r, _, _ = select.select([self.master], [], [], min(0.3, remaining))
                        if not r:
                            continue
                        try:
                            chunk = os.read(self.master, 65536)
                        except OSError:
                            break
                        if not chunk:
                            break
                        buf += chunk
                    buf += self._drain(0.4)  # resztka linii za markerem
                    text = buf.decode("utf-8", "replace")
                    rc, cwd = None, ""
                    out_lines = []
                    for line in text.split("\n"):
                        if mark in line:
                            tail = line.split(mark, 1)[1].strip().rstrip("\r")
                            if "@" in tail:
                                rcp, _, cwdp = tail.partition("@")
                                if rcp.isdigit():
                                    rc = int(rcp)
                                cwd = cwdp
                            continue
                        out_lines.append(line)
                    output = _ANSI_RE.sub("", "\n".join(out_lines)).strip("\n")[-20000:]
                    if timed_out:
                        output += "\n[timeout 60s — wyslano Ctrl+C, sesja zyje dalej]"
                        rc = 124
                    return {"ok": rc == 0, "code": rc if rc is not None else -1,
                            "output": output, "cwd": cwd}

        _term = None

        def _get_term() -> _TermSession:
            nonlocal _term
            if _term is None:
                _term = _TermSession()
            return _term

        def _run_term(command: str, user: str = "?") -> dict:
            r = _get_term().run(command, timeout=60)
            logger.info(f"terminal[{user}]: {command!r} -> rc={r['code']}")
            return r

        @app.post("/api/terminal")
        def api_terminal(request: Request, payload: dict = Body(...)):
            """Admin-only command terminal for the HAI-NL dashboard.
            UWAGA: sync `def` (nie async) — FastAPI wykonuje go w threadpoolu,
            wiec 60s komenda NIE blokuje event loopa panelu."""
            who = getattr(request.state, "hai_user", None)
            if not who or who.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Terminal wymaga roli admin")
            command = str(payload.get("command", "")).strip()
            if not command:
                raise HTTPException(status_code=400, detail="Brak komendy")
            if len(command) > 2000:
                raise HTTPException(status_code=413, detail="Komenda jest za dluga")
            if _is_tui(command):
                first = command.split()[0]
                return {"ok": False, "code": 126, "cwd": "",
                        "output": (f"⛔ '{first}' to program interaktywny (TUI) — w web-terminalu "
                                   f"nie zadziala (brak prawdziwego terminala).\n"
                                   f"Uzyj SSH. Do pracy w tle: `nohup {command} > log.txt 2>&1 &` "
                                   f"albo `tmux new -d '{command}'`.")}
            return _run_term(command, who.get("user", "?"))

    import importlib
    _route_modules = ["dashboard", "trading", "logs", "settings", "debug", "ai", "screening", "signals_routes", "ctrl"]
    for mod in _route_modules:
        try:
            m = importlib.import_module(f"routes.{mod}")
            app.include_router(m.router)
        except ModuleNotFoundError:
            logger.warning(f"routes.{mod} not found, skipping")

    return app

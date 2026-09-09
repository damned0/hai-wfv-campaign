# ===========================================
# HAI_EPV Engine ver.10 Final — core/watchdog.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: monitoring zdrowia procesu (engine/AI/exchange), auto-restart
# przy zawieszeniu, raportowanie statusu.
# ===========================================
import asyncio
import logging
import os
import time
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

class Watchdog:
    def __init__(self):
        self.enabled = True
        self.check_interval = 25
        self.restart_delay = 8
        self.max_restarts = 6
        self.restart_window = 3600
        self.restart_count = 0
        self.restart_times = []
        self.last_check = None
        self._running = False
        # Naprawa (audyt 2026-07-07) - wczesniej port/nazwa/cwd byly na sztywno
        # zakodowane na EPV (port=5010, cwd=".../HAI_EPV"), a plik byl
        # identyczna kopia na DEV/LAB - kazda instancja monitorowala i mogla
        # zabic/restartowac EPV zamiast SIEBIE. Teraz czyta wlasny port z
        # core/config.py (ktory juz poprawnie czyta .env kazdej instancji) i
        # wlasny katalog z __file__, wiec dziala poprawnie niezaleznie od tego
        # ktora instancja go uruchamia.
        try:
            from .config import config as _cfg
            self.port = _cfg.PORT
        except Exception:
            self.port = 5010
        self.instance_dir = str(Path(__file__).resolve().parent.parent)
        self.process_name = Path(self.instance_dir).name  # np. "HAI_EPV", "HAI_DEV"

    async def start(self):
        self._running = True
        logger.info(f"Watchdog v1.2 start | monitoruje {self.process_name} na porcie {self.port}")
        await asyncio.sleep(45)  # czas na start engine
        while self._running:
            try:
                await self._check()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            await asyncio.sleep(self.check_interval)

    async def stop(self):
        self._running = False

    async def _check(self):
        self.last_check = datetime.now(timezone.utc)
        issues = []

        # 1. Sprawdz czy proces na WLASNYM porcie dziala
        if not await self._is_process_running():
            issues.append("ENGINE_DOWN")

        # 2. Sprawdz czy sa aktualne dane cenowe
        from .engine import engine
        if len(engine._prices) < 5:
            issues.append("NO_PRICE_DATA")

        # 3. Sprawdz historie
        if len(engine._price_history_1h) < 3:
            issues.append("NO_HISTORY")

        if issues:
            logger.warning(f"Watchdog wykryl problemy: {issues}")
            if "ENGINE_DOWN" in issues:
                await self._restart_self()
            else:
                await self._send_alert(issues)

    async def _is_process_running(self) -> bool:
        try:
            result = subprocess.run(
                ["fuser", f"{self.port}/tcp"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _restart_self(self):
        now = time.time()
        self.restart_times = [t for t in self.restart_times if now - t < self.restart_window]

        if len(self.restart_times) >= self.max_restarts:
            logger.critical(f"Watchdog: Osiagnieto limit restartow ({self.process_name})! Wymagana interwencja reczna.")
            return

        logger.info(f"Watchdog: Restartuje {self.process_name} (port {self.port})...")

        # Zabij proces na WLASNYM porcie
        subprocess.run(["fuser", "-k", f"{self.port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        await asyncio.sleep(self.restart_delay)

        # Uruchom ponownie z WLASNEGO katalogu
        try:
            log_path = f"/tmp/{self.process_name.lower()}_watchdog_restart.log"
            subprocess.Popen(
                ["python3", "main.py"],
                cwd=self.instance_dir,
                stdout=open(log_path, "a"),
                stderr=open(log_path, "a")
            )
            self.restart_times.append(now)
            logger.info(f"{self.process_name} zrestartowany przez Watchdog")
        except Exception as e:
            logger.error(f"Nie udalo sie uruchomic {self.process_name}: {e}")

    async def _send_alert(self, issues: List[str]):
        msg = f"WATCHDOG ALERT | {self.process_name} | {', '.join(issues)}"
        logger.warning(msg)
        try:
            from .telegram import telegram
            await telegram.alert_watchdog(msg)
        except:
            pass

    def get_status(self) -> Dict:
        from .engine import engine
        return {
            "running": self._running,
            "port": self.port,
            "engine_running": engine._running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "restarts": len(self.restart_times),
            "uptime": str(timedelta(seconds=int((datetime.now(timezone.utc) - datetime.fromisoformat(self.stats["uptime_start"])).total_seconds()))) if hasattr(self, 'stats') else "N/A"
        }


watchdog = Watchdog()

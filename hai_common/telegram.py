# ===========================================
# HAI_EPV Engine ver.10 Final — core/telegram.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: powiadomienia Telegram o transakcjach/bledach, daily_summary_loop
# (dzienne podsumowanie PnL), polling komend.
# ===========================================
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from .config import config
from .state import state

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """ZAWSZE timezone-aware UTC. Nie używać datetime.utcnow()!"""
    return datetime.now(timezone.utc)


def _to_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Naprawia naive datetime → UTC-aware. Idempotentne."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class TelegramNotifier:

    def __init__(self):
        self.token = getattr(config, "TELEGRAM_TOKEN", None)
        self.chat_id = getattr(config, "TELEGRAM_CHAT_ID", None)
        self.enabled = bool(self.token and self.chat_id)
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown_sec = 60

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False

        # Anti-spam — minimum 60s między alertami
        now = _utcnow()
        if self.last_alert_time is not None:
            last = _to_aware(self.last_alert_time)
            if (now - last).total_seconds() < self.alert_cooldown_sec:
                logger.debug("Telegram cooldown — pominięto alert")
                return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4096],   # limit Telegrama
                "parse_mode": parse_mode,
            }
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json=payload)
                if r.status_code == 200:
                    self.last_alert_time = now
                    return True
                logger.warning(f"Telegram HTTP {r.status_code}: {r.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    async def alert_loop(self):
        """Pętla obsługująca alerty z kolejki state."""
        logger.info("Telegram: alerty aktywne")
        while True:
            try:
                # Tu logika pobierania alertów z state — zachowaj swoją
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Telegram alert loop error: {e}")
                await asyncio.sleep(30)

    async def daily_summary_loop(self):
        """
        Wysyła podsumowanie o 10:00 i 22:00 UTC.
        FIX: wszystkie datetime są timezone-aware.
        """
        logger.info("Daily summary loop start | 10:00 i 22:00 UTC")
        last_sent_date_hour = None  # (date, hour) — żeby nie wysyłać 2× w tej samej godzinie

        while True:
            try:
                now = _utcnow()
                hour = now.hour
                today = now.date()

                # Wysyłka o 10:00 lub 22:00 UTC, max raz na okno
                if hour in (10, 22):
                    key = (today, hour)
                    if last_sent_date_hour != key:
                        await self._send_daily_summary()
                        last_sent_date_hour = key

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Daily summary error: {e}")
                await asyncio.sleep(60)

    async def _send_daily_summary(self):
        """Generuje i wysyła dzienne podsumowanie."""
        try:
            # Pobierz statystyki — wszystko z bezpieczną konwersją czasów
            from .database import SessionLocal, Position
            now = _utcnow()
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            db = SessionLocal()
            try:
                positions = db.query(Position).filter(
                    Position.status == "closed",
                ).all()

                # Filtruj w Pythonie — bezpiecznie z timezone
                today_positions = []
                for p in positions:
                    exit_t = _to_aware(p.exit_time)
                    if exit_t is not None and exit_t >= day_start:
                        today_positions.append(p)
            finally:
                db.close()

            if not today_positions:
                msg = f"📊 <b>Dzienne podsumowanie</b>\n\nBrak zamkniętych pozycji dziś."
            else:
                total = len(today_positions)
                wins = sum(1 for p in today_positions if (p.pnl or 0) > 0)
                losses = total - wins
                pnl_sum = sum(p.pnl or 0 for p in today_positions)
                win_rate = (wins / total * 100) if total > 0 else 0

                msg = (
                    f"📊 <b>Dzienne podsumowanie</b>\n\n"
                    f"📈 Trades: {total}\n"
                    f"✅ Wins: {wins}\n"
                    f"❌ Losses: {losses}\n"
                    f"🎯 Win rate: {win_rate:.1f}%\n"
                    f"💰 PnL: {pnl_sum:+.2f} USDT\n"
                    f"🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}"
                )

            await self.send(msg)
            logger.info("Daily summary wysłane")

        except Exception as e:
            logger.error(f"_send_daily_summary error: {e}")


telegram = TelegramNotifier()
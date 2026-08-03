# ===========================================
# HAI_EPV Engine ver.10 Final — core/ohlcv_cache.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: cache OHLCV — RAM (ostatnie 200 świec) + dysk Parquet (historia
# do 1 roku), odczyt/zapis warehouse.
# ===========================================
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ohlcv"


class OHLCVCache:

    def __init__(self):
        self.ram_limit = 200   # max swiec w RAM
        self.max_days  = 365   # max dni na dysku
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ?? Zapis na dysk ??????????????????????????????????

    def save(self, symbol: str, timeframe: str, candles: List[Dict]):
        """Zapisuje swiecze na dysk jako Parquet."""
        if not candles:
            return
        try:
            import pandas as pd
            path = self._path(symbol, timeframe)
            df_new = pd.DataFrame(candles)
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

            # Dolacz do istniejacych danych
            if path.exists():
                df_old = pd.read_parquet(path)
                df = pd.concat([df_old, df_new]).drop_duplicates(
                    subset=["timestamp"]
                ).sort_values("timestamp")
            else:
                df = df_new.sort_values("timestamp")

            # Usun dane starsze niz max_days
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.max_days)
            df = df[df["timestamp"] > cutoff]

            df.to_parquet(path, index=False)
            logger.debug(f"Saved {len(df)} candles: {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"OHLCV save error {symbol}: {e}")

    def load_ram(self, symbol: str, timeframe: str) -> List[Dict]:
        """Laduje ostatnie N swiec do RAM."""
        try:
            import pandas as pd
            path = self._path(symbol, timeframe)
            if not path.exists():
                return []
            df = pd.read_parquet(path).tail(self.ram_limit)
            return df.to_dict("records")
        except Exception as e:
            logger.error(f"OHLCV load error {symbol}: {e}")
            return []

    def load_full(self, symbol: str, timeframe: str,
                  days: int = 90) -> List[Dict]:
        """Laduje pelna historie ? dla backtestera/AI."""
        try:
            import pandas as pd
            path = self._path(symbol, timeframe)
            if not path.exists():
                return []
            df = pd.read_parquet(path)
            if days:
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
                df = df[df["timestamp"] > cutoff]
            return df.to_dict("records")
        except Exception as e:
            logger.error(f"OHLCV load_full error {symbol}: {e}")
            return []

    def get_stats(self) -> Dict:
        """Statystyki cache."""
        try:
            files  = list(CACHE_DIR.glob("*.parquet"))
            total  = sum(f.stat().st_size for f in files)
            return {
                "files":    len(files),
                "size_mb":  round(total / 1024 / 1024, 2),
                "dir":      str(CACHE_DIR),
            }
        except Exception:
            return {"files": 0, "size_mb": 0}

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return CACHE_DIR / f"{safe}_{timeframe}.parquet"


ohlcv_cache = OHLCVCache()

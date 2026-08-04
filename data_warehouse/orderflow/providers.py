#!/usr/bin/env python3
# ===========================================
# gen.Flow — dostawcy danych order-flow (pluggable)
# ===========================================
# Abstrakcja nad zrodlami danych trade-level / order-flow. Kazdy dostawca
# zwraca ZNORMALIZOWANE transakcje jako liste dictow:
#   {"ts": int_ms, "price": float, "qty": float, "side": +1|-1}
#   side = +1 (agresywne KUPNO, taker buy) / -1 (agresywne SPRZEDANIE, taker sell)
# Na tym feature builder liczy CVD/delta/absorpcje — niezaleznie od zrodla.
#
# AKTYWNY: Binance aggTrades (darmowy, trade-level, REST+WS).
# GOTOWE DO PODPIECIA: mmt.gg (API $199/mc, X-API-Key), + inne (Bybit/OKX).
# Wybor przez ENV ORDERFLOW_PROVIDER (domyslnie "binance") lub get_provider(name).
# ===========================================
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger("orderflow.providers")

# Mapowanie symbolu kanonicznego (BTC) -> symbol per dostawca. Rozbudowywac
# gdy dojdzie multi-symbol; na dzis reżim BTC-only.
_SYMBOL_MAP = {
    "binance": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "mmt":     {"BTC": "btc/usd", "ETH": "eth/usd"},
    "bybit":   {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
}


class OrderFlowProvider(ABC):
    """Kontrakt dostawcy. Kazda implementacja zwraca znormalizowane trady."""
    name: str = "base"
    active: bool = False

    @abstractmethod
    def fetch_trades(self, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
        """Historyczne trady w oknie [start_ms, end_ms). Znormalizowane."""
        ...

    def health(self) -> Dict:
        """Szybki test dostepnosci (do dashboardu/audytu)."""
        return {"name": self.name, "active": self.active}

    def map_symbol(self, canonical: str) -> str:
        return _SYMBOL_MAP.get(self.name, {}).get(canonical, canonical)


# ── AKTYWNY: Binance aggTrades (darmowy) ─────────────────────────────────────
class BinanceAggTrades(OrderFlowProvider):
    """Binance USD-M Futures aggTrades. Darmowy, trade-level.
    Pole 'm' (isBuyerMaker): True => kupujacy byl makerem => AGRESOR SPRZEDAL
    (side=-1). False => agresor KUPIL (side=+1). To jest sygnal agresji."""
    name = "binance"
    active = True
    BASE = "https://fapi.binance.com"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch_trades(self, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
        import requests
        sym = self.map_symbol(symbol)
        out: List[Dict] = []
        cur = start_ms
        # aggTrades: max 1000/req, paginacja po startTime (okna <=1h zeby nie
        # przekroczyc limitu; Binance wymaga startTime/endTime < 1h przy braku fromId)
        while cur < end_ms:
            win_end = min(cur + 55 * 60 * 1000, end_ms)  # <1h okno
            try:
                r = requests.get(f"{self.BASE}/fapi/v1/aggTrades",
                                 params={"symbol": sym, "startTime": cur,
                                         "endTime": win_end, "limit": 1000},
                                 timeout=self.timeout)
                rows = r.json()
                if not isinstance(rows, list) or not rows:
                    cur = win_end
                    continue
                for t in rows:
                    out.append({
                        "ts": int(t["T"]),
                        "price": float(t["p"]),
                        "qty": float(t["q"]),
                        "side": -1 if t["m"] else 1,
                    })
                # nastepne okno od ostatniego ts+1 (albo win_end jesli <1000)
                last_ts = int(rows[-1]["T"])
                cur = (last_ts + 1) if len(rows) >= 1000 else win_end
            except Exception as e:
                logger.warning(f"binance aggTrades {sym} @{cur}: {e}")
                cur = win_end
            time.sleep(0.12)  # ~budzet wagi, bezpiecznie
        return out


# ── GOTOWE DO PODPIECIA: mmt.gg (API $199/mc) ────────────────────────────────
class MMTProvider(OrderFlowProvider):
    """mmt.gg developer API. NIEAKTYWNY do czasu podania MMT_API_KEY.
    Ma gotowe: volume delta z bucketami rozmiaru, footprint, liquidation/
    orderbook heatmapy, funding/depth. REST (X-API-Key) + WS. Model 'weight'.
    Endpointy z /api — do uzupelnienia po wykupieniu i lekturze docs; szkielet
    trzyma kontrakt (znormalizowane trady), zeby feature builder dzialal bez
    zmian gdy zrodlo sie przelaczy."""
    name = "mmt"
    active = False
    BASE = "https://api.mmt.gg"  # do potwierdzenia w docs

    def __init__(self):
        self.key = os.environ.get("MMT_API_KEY", "")
        self.active = bool(self.key)

    def fetch_trades(self, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
        if not self.active:
            raise RuntimeError("MMT nieaktywny: brak MMT_API_KEY")
        import requests
        sym = self.map_symbol(symbol)
        # SZKIELET — endpoint/parametry do wypelnienia wg docs mmt /api.
        # Kontrakt wyjscia MUSI zostac: {"ts","price","qty","side"}.
        r = requests.get(f"{self.BASE}/v1/trades",
                         headers={"X-API-Key": self.key},
                         params={"symbol": sym, "start": start_ms, "end": end_ms},
                         timeout=20)
        rows = r.json().get("trades", []) if r.ok else []
        out = []
        for t in rows:
            # normalizacja pol mmt -> kontrakt (nazwy pol do potwierdzenia)
            out.append({"ts": int(t.get("ts") or t.get("time")),
                        "price": float(t.get("price") or t.get("p")),
                        "qty": float(t.get("qty") or t.get("size") or t.get("q")),
                        "side": 1 if (t.get("side") in ("buy", "b", 1)) else -1})
        return out


# ── Rejestr + selektor ───────────────────────────────────────────────────────
_REGISTRY = {
    "binance": BinanceAggTrades,
    "mmt": MMTProvider,
    # "bybit": BybitProvider,  # miejsce na kolejne zrodla
}


def get_provider(name: Optional[str] = None) -> OrderFlowProvider:
    name = (name or os.environ.get("ORDERFLOW_PROVIDER", "binance")).lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"nieznany dostawca order-flow: {name} (dostepne: {list(_REGISTRY)})")
    return cls()


def list_providers() -> List[Dict]:
    """Do audytu/dashboardu: ktore zrodla sa aktywne."""
    res = []
    for n, cls in _REGISTRY.items():
        try:
            res.append(cls().health())
        except Exception as e:
            res.append({"name": n, "active": False, "err": str(e)})
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Dostawcy order-flow:", list_providers())
    p = get_provider("binance")
    now = int(time.time() * 1000)
    tr = p.fetch_trades("BTC", now - 60_000, now)
    print(f"Binance BTC ostatnia minuta: {len(tr)} tradow")
    if tr:
        buys = sum(t["qty"] for t in tr if t["side"] == 1)
        sells = sum(t["qty"] for t in tr if t["side"] == -1)
        print(f"  agresja BUY={buys:.3f} SELL={sells:.3f} delta={buys-sells:+.3f} BTC")

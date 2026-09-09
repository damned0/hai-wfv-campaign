# ===========================================
# HAI_EPV Engine ver.10 Final — core/coingecko.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: CoinGecko API klient — domyslne zrodlo cen/danych dla trybu PAPER.
# ===========================================
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

# Mapowanie symbol -> CoinGecko ID
SYMBOL_TO_ID = {
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "SOL":   "solana",
    "BNB":   "binancecoin",
    "DOGE":  "dogecoin",
    "AVAX":  "avalanche-2",
    "LINK":  "chainlink",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "SUI":   "sui",
    "MATIC": "matic-network",
    "PEPE":  "pepe",
    "WIF":   "dogwifcoin",
    "INJ":   "injective-protocol",
    "TIA":   "celestia",
    "SEI":   "sei-network",
    "APT":   "aptos",
    "FTM":   "fantom",
    "NEAR":  "near",
    "ATOM":  "cosmos",
    "XRP":   "ripple",
    "ADA":   "cardano",
    "DOT":   "polkadot",
    "LTC":   "litecoin",
    "SHIB":  "shiba-inu",
    "TRX":   "tron",
    "AAVE":  "aave",
    "FIL":   "filecoin",
    "ORDI":  "ordinals",
    "RUNE":  "thorchain",
    "ENA":   "ethena",
    "FET":   "fetch-ai",
    "POL":   "matic-network",
}

BASE_URL = "https://api.coingecko.com/api/v3"
FGI_URL  = "https://api.alternative.me/fng/?limit=1"
GATE_FUT_URL = "https://api.gateio.ws/api/v4/futures/usdt/tickers"

# Stablecoiny: zostawiamy tylko USDT i USDC, resztę wycinamy z top100.
STABLE_KEEP = {"USDT", "USDC"}
STABLE_BAD = {
    "DAI", "TUSD", "FDUSD", "USDE", "USDG", "USD1", "USDD", "USDP", "GUSD",
    "PYUSD", "FRAX", "LUSD", "USDJ", "EURT", "EURC", "EURS", "USDS", "BUSD",
    "USTC", "USDL", "RLUSD", "USD0", "GHO", "SUSD", "USDX", "CUSD", "MUSD",
    "DOLA", "USDB", "USDY", "USDF", "USDK", "HUSD", "OUSD", "MIM", "DUSD",
    "USN", "VUSD", "USDR", "AEUR", "EUROC", "BFUSD", "USDGO", "XUSD", "USDO",
}


# Tokenizowane akcje / ETF-y / surowce / pre-IPO na Gate.io futures.
# Gate trzyma je w tych samych USDT-futures i API ich NIE oznacza — dlatego
# kuratorowany blocklist (crypto-only wszechswiat, decyzja usera 2026-07-24).
# NIE blokujemy krypto-tokenow zlota (PAXG, XAUT) — to realne aktywa krypto.
EQUITY_BAD = {
    # półprzewodniki / hardware
    "SNDK", "SKHYNIX", "SKHY", "MU", "INTC", "ARM", "MRVL", "SAMSUNG", "AAOI",
    "NOK", "AMD", "NBIS", "MSFT", "TSM", "GLW", "DELL", "IBM", "QCOM", "WDC",
    "AMAT", "ASML", "AXTI", "CRWV", "IREN", "RKLB", "DRAM", "CXMT", "XIAOMI",
    "SMCI", "AVGO", "ALAB", "MVLL", "ON", "SLX", "ORCL",
    # spolki / konsument
    "BABA", "HIMS", "ASTS", "NFLX", "BE", "ONDS", "CBRS", "BEAT", "AAPL",
    "TSLA", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "COIN", "HOOD", "MSTR",
    "PLTR", "CRCL", "NKE", "DIS", "PYPL", "SQ", "SHOP", "UBER", "ABNB",
    # tokenizowane akcje z sufiksem X
    "QQQX", "TSLAX", "METAX", "GOOGLX", "AAPLX", "SNXX", "CRCLX", "COINX",
    "SPYX", "AMZNX", "MSTRX", "PLTRX", "NVDAX", "HOODX", "TQQQX", "ORCLX",
    "MSFTX", "NFLXX", "AMDX", "INTCX",
    # ETF-y / indeksy
    "SOXL", "SOXS", "EWY", "NAS100", "US", "SPY", "QQQ", "SPX", "NDX", "DXY",
    # surowce (syntetyczne feedy cenowe, NIE krypto-tokeny zlota)
    "XAU", "XAG", "XPT", "XPD", "CL", "WTI", "BRENT", "NG", "HG", "GC", "SI",
    # pre-IPO / AI spolki tokenizowane
    "SPCX", "ANTHROPIC", "OPENAI", "ZHIPU", "BARD", "INTW",
}

# Tokenizowane RWA: obligacje/fundusze/T-bille/private credit (BlackRock BUIDL,
# Circle USYC, Ondo OUSG, Figure Heloc, Janus/Invesco/Spiko fundusze...).
# Wysokie kapitalizacje -> lezą w top100 CMC/CoinGecko, ale to NIE handlowalne
# perpy krypto. Plus kolejne stable spoza glownej listy.
RWA_BAD = {
    "FIGR_HELOC", "FIGR", "USYC", "BUIDL", "JTRSY", "JAAA", "USTB", "BCAP",
    "EUTBL", "OUSG", "YLDS", "A7A5", "BENJI", "WTGXX", "FOBXX", "TBILL",
    "USTBL", "USYC", "SPIKO", "HASH", "OUST", "TBY", "JIUSD",
    # dodatkowe stable / syntetyki
    "STABLE", "USX", "U", "USD0PP",
}


def _is_rwa(sym: str) -> bool:
    return (sym or "").upper() in RWA_BAD


def _is_bad_stable(sym: str) -> bool:
    s = (sym or "").upper()
    if s in STABLE_KEEP:
        return False
    if s in STABLE_BAD:
        return True
    return s.startswith("USD") or s.endswith("USD")


def _is_excluded(sym: str) -> bool:
    """Wyklucz: stablecoin (poza USDT/USDC), tokenizowana akcja/ETF/surowiec,
    albo tokenizowane RWA (obligacje/fundusze/T-bille)."""
    s = (sym or "").upper()
    if s in STABLE_KEEP:
        return False
    if not s.isascii():          # spam-tokeny z chinskimi/emoji symbolami
        return True
    return _is_bad_stable(s) or s in EQUITY_BAD or s in RWA_BAD

# Prosty cache w pamieci ? nie fetchujemy co sekunde
_cache: Dict[str, dict] = {}
_cache_ts: Dict[str, float] = {}
_CACHE_TTL = 60  # sekund


def _is_fresh(key: str) -> bool:
    import time
    return key in _cache_ts and (time.time() - _cache_ts[key]) < _CACHE_TTL


def _set_cache(key: str, data):
    import time
    _cache[key]    = data
    _cache_ts[key] = time.time()


class CoinGeckoClient:

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    logger.warning("CoinGecko rate limit ? czekam 10s")
                    await asyncio.sleep(10)
                else:
                    logger.error(f"CoinGecko HTTP {r.status_code}: {url}")
        except Exception as e:
            logger.error(f"CoinGecko request error: {e}")
        return None

    def symbol_to_id(self, symbol: str) -> str:
        sym = symbol.replace("/USDT:USDT", "").replace("/USDT", "").upper()
        return SYMBOL_TO_ID.get(sym, sym.lower())

    async def get_price(self, symbols: List[str]) -> Dict[str, dict]:
        """Ceny, zmiana 24h, wolumen dla listy symboli."""
        cache_key = "prices_" + "_".join(sorted(symbols))
        if _is_fresh(cache_key):
            return _cache[cache_key]

        ids = [self.symbol_to_id(s) for s in symbols]
        ids_str = ",".join(ids)
        data = await self._get(f"{BASE_URL}/simple/price", params={
            "ids":                  ids_str,
            "vs_currencies":        "usd",
            "include_24hr_change":  "true",
            "include_24hr_vol":     "true",
            "include_market_cap":   "true",
        })
        if not data:
            return {}

        result = {}
        for sym in symbols:
            cg_id = self.symbol_to_id(sym)
            if cg_id in data:
                d = data[cg_id]
                result[sym] = {
                    "price":      d.get("usd", 0),
                    "change_24h": d.get("usd_24h_change", 0),
                    "volume_24h": d.get("usd_24h_vol", 0),
                    "market_cap": d.get("usd_market_cap", 0),
                }
        _set_cache(cache_key, result)
        return result

    async def get_ohlcv(self, symbol: str,
                        days: int = 1) -> List[Dict]:
        """
        OHLCV dla wykresu.
        days=1  -> dane co ~30min
        days=7  -> dane co ~4h
        days=30 -> dane dzienne
        """
        cache_key = f"ohlcv_{symbol}_{days}"
        if _is_fresh(cache_key):
            return _cache[cache_key]

        cg_id = self.symbol_to_id(symbol)
        data  = await self._get(f"{BASE_URL}/coins/{cg_id}/ohlc", params={
            "vs_currency": "usd",
            "days":        str(days),
        })
        if not data:
            return []

        result = [{
            "time":  int(c[0] / 1000),
            "open":  c[1],
            "high":  c[2],
            "low":   c[3],
            "close": c[4],
        } for c in data]

        _set_cache(cache_key, result)
        return result

    async def get_coin_detail(self, symbol: str) -> Optional[Dict]:
        """Szczegolowe dane monety ? RSI, EMA liczone lokalnie."""
        cache_key = f"detail_{symbol}"
        if _is_fresh(cache_key):
            return _cache[cache_key]

        cg_id = self.symbol_to_id(symbol)
        data  = await self._get(
            f"{BASE_URL}/coins/{cg_id}",
            params={"localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false"}
        )
        if not data:
            return None

        md = data.get("market_data", {})
        result = {
            "symbol":      symbol,
            "name":        data.get("name", symbol),
            "price":       md.get("current_price", {}).get("usd", 0),
            "change_24h":  md.get("price_change_percentage_24h", 0),
            "change_7d":   md.get("price_change_percentage_7d", 0),
            "volume_24h":  md.get("total_volume", {}).get("usd", 0),
            "market_cap":  md.get("market_cap", {}).get("usd", 0),
            "high_24h":    md.get("high_24h", {}).get("usd", 0),
            "low_24h":     md.get("low_24h", {}).get("usd", 0),
            "ath":         md.get("ath", {}).get("usd", 0),
            "image":       data.get("image", {}).get("small", ""),
        }
        _set_cache(cache_key, result)
        return result

    async def get_fgi(self) -> Dict:
        """Fear & Greed Index z alternative.me"""
        if _is_fresh("fgi"):
            return _cache["fgi"]

        data = await self._get(FGI_URL)
        if data and "data" in data:
            d = data["data"][0]
            result = {
                "value":       int(d.get("value", 50)),
                "class":       d.get("value_classification", "Neutral").upper(),
                "timestamp":   d.get("timestamp", ""),
            }
            _set_cache("fgi", result)
            return result
        return {"value": 50, "class": "NEUTRAL", "timestamp": ""}

    async def _get_gate(self, url: str, params: dict = None) -> Optional[list]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params,
                                     headers={"Accept": "application/json"})
                if r.status_code == 200:
                    return r.json()
                logger.error(f"Gate.io HTTP {r.status_code}: {url}")
        except Exception as e:
            logger.error(f"Gate.io request error: {e}")
        return None

    async def get_top100(self) -> List[Dict]:
        """TOP 100 monet po market cap (ranking jak CoinMarketCap; CMC wymaga
        klucza, CoinGecko daje te sama kolejnosc bez klucza).
        Naturalna kolejnosc — stable NIE sa wypychane na gore (USDT ~#3, USDC ~#5).
        Wyciete: stable poza USDT/USDC, tokenizowane akcje/ETF/RWA (BUIDL, FIGR_HELOC...).
        Fallback: Gate.io USDT-futures po wolumenie."""
        if _is_fresh("top100"):
            return _cache["top100"]

        # --- glowne zrodlo: CoinGecko markets po market cap (ekwiwalent CMC) ---
        cg = await self._get(f"{BASE_URL}/coins/markets", params={
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": 1, "sparkline": "false",
            "price_change_percentage": "24h",
        })
        if cg:
            result = []
            for d in cg:
                sym = (d.get("symbol") or "").upper()
                if _is_excluded(sym):
                    continue
                result.append({
                    "symbol": sym, "id": d.get("id"), "name": d.get("name"),
                    "price": d.get("current_price", 0),
                    "change_24h": d.get("price_change_percentage_24h", 0),
                    "volume_24h": d.get("total_volume", 0),
                    "market_cap": d.get("market_cap", 0),
                    "image": d.get("image", ""), "source": "coingecko_mcap",
                })
                if len(result) >= 100:
                    break
            if result:
                _set_cache("top100", result)
                return result

        # --- fallback: Gate.io USDT-futures po wolumenie ---
        data = await self._get_gate(GATE_FUT_URL)
        if not data:
            return []
        rows = []
        for t in data:
            contract = t.get("contract", "")
            if not contract.endswith("_USDT"):
                continue
            base = contract[:-5]
            if _is_excluded(base):
                continue
            try:
                vol = float(t.get("volume_24h_quote") or 0)
            except Exception:
                vol = 0.0
            rows.append({
                "symbol": base, "id": base.lower(), "name": base,
                "price": float(t.get("last") or 0),
                "change_24h": float(t.get("change_percentage") or 0),
                "volume_24h": vol, "image": "", "source": "gate_futures",
            })
        rows.sort(key=lambda r: r["volume_24h"], reverse=True)
        result = rows[:100]
        _set_cache("top100", result)
        return result


coingecko = CoinGeckoClient()

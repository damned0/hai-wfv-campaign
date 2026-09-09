# ===========================================
# HAI_EPV Engine ver.10 Final — core/symbols.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: SYMBOLS (Single Source of Truth listy symboli, dane w warehouse
# 1h+4h+1d), is_supported() — zastepuje dawne rozproszone whitelisty.
# ===========================================
"""
Lista 30 symboli wspierana przez ProjektHAI.

ZASADA: symbol jest tu TYLKO jeśli ma dane w warehouse 1h+4h+1d.
Dodanie nowego symbolu = (1) pobranie historii do warehouse, (2) dodanie tu.

Format użycia:
  - z ":USDT" (Bitget swap): TRADING_SYMBOLS
  - bez ":USDT" (Binance/warehouse): WAREHOUSE_SYMBOLS
"""

# Bazowe nazwy (jak w warehouse: BTC, ETH...)
WAREHOUSE_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "LTC", "NEAR", "OP", "SUI", "ARB", "ATOM", "WIF",
    "TRX", "AAVE", "FIL", "INJ", "TIA", "FET", "APT", "ORDI", "RUNE", "ENA",
    "SHIB", "PEPE", "MATIC",
]

# Format Bitget swap (USDT perpetuals)
# v6.3: TRADING_SYMBOLS to whitelist PAPER/LIVE TRADING (6 symboli z 3/3 backtest)
# WAREHOUSE_SYMBOLS (30) zostaje dla ml_trainer i backtester
# 2026-08-02: WL50 — top 50 coinow wg sygnałów 6h (honest WFV). Zgodny z PAPER_WHITELIST w .env.
PAPER_WHITELIST = [
    'BTC','ETH','DOGE','XRP','1000SHIB','ETC','BNB','BCH','TRX','SAND',
    'AXS','FIL','LTC','STORJ','NEO','AVAX','ADA','ZRX','UNI','SOL',
    'RUNE','ANKR','AAVE','MANA','CHZ','RSR','ENJ','BAND','SKL','MASK',
    'SUSHI','ONT','BAT','1INCH','ZEC','GRT','LINK','CELR','DOT','COMP',
    'ONE','KAVA','KSM','SNX','ZIL','XLM','EGLD','VET','HBAR','GALA',
]
TRADING_SYMBOLS = [f"{s}/USDT:USDT" for s in PAPER_WHITELIST]


def to_warehouse(symbol: str) -> str:
    """BTC/USDT:USDT -> BTC | BTC/USDT -> BTC | BTC -> BTC"""
    return symbol.split("/")[0].upper()


def to_trading(symbol: str) -> str:
    """BTC -> BTC/USDT:USDT | BTC/USDT:USDT -> BTC/USDT:USDT"""
    if "/" in symbol:
        return symbol if ":" in symbol else f"{symbol}:USDT"
    return f"{symbol}/USDT:USDT"


def is_supported(symbol: str) -> bool:
    """Czy ten symbol jest w naszym TRADING whitelist (6 z 3/3)?"""
    return to_warehouse(symbol) in PAPER_WHITELIST

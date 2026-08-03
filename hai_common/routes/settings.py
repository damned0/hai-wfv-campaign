# ===========================================
# HAI_EPV Engine ver.10 Final — routes/settings.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: /module/section_1a..6 (panele ustawien dashboardu HTMX-style),
# legacy AI_LEARN_ENABLED toggle endpoint.
# ===========================================
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from ..config import config
from ..state import state
from ..app import templates

router = APIRouter()

# Moduły widoku

@router.get("/module/section_1a", response_class=HTMLResponse)
async def section_1a(request: Request):
    mode        = config.effective_mode
    balance     = state.get_paper_balance() if mode != "live" else state.get_live_balance()
    system_logs = state.get_logs("system", 20) or []
    return templates.TemplateResponse(request, "modules/section_1a_footer.html", {
        "mode":         mode,
        "balance":      round(balance, 2),
        "live_balance": round(state.get_live_balance(), 2),
        "pnl":          0.0,
        "ai_enabled":   config.AI_ENABLED,
        "ai_learn":     config.AI_LEARN_ENABLED,
        "ai_trade":     config.AI_TRADE_ENABLED,
        "system_logs":  system_logs,
        "strategy":     config.DEFAULT_STRATEGY,
        "exchange":     config.DEFAULT_EXCHANGE,
    })

@router.get("/module/section_2", response_class=HTMLResponse)
async def section_2(request: Request):
    from ..coingecko import coingecko
    from ..engine import engine
    from ..strategies.registry import get_strategy

    fgi  = await coingecko.get_fgi()
    top5 = []
    try:
        s = get_strategy()
        if s and hasattr(s, "select_top5") and engine._price_history_1h:
            d1h, vol = engine._prepare_data("1H")
            d4h, _   = engine._prepare_data("4H")
            d1d, _   = engine._prepare_data("1D")
            top5 = s.select_top5(engine._top_symbols[:50], d1h, d4h, d1d, vol)
    except Exception:
        pass

    top10 = [
        {"symbol": "BTC",  "change_24h": 0.0},
        {"symbol": "ETH",  "change_24h": 0.0},
        {"symbol": "SOL",  "change_24h": 0.0},
        {"symbol": "BNB",  "change_24h": 0.0},
        {"symbol": "DOGE", "change_24h": 0.0},
        {"symbol": "AVAX", "change_24h": 0.0},
        {"symbol": "LINK", "change_24h": 0.0},
        {"symbol": "ARB",  "change_24h": 0.0},
        {"symbol": "OP",   "change_24h": 0.0},
        {"symbol": "SUI",  "change_24h": 0.0},
    ]

    try:
        syms   = [c["symbol"] for c in top10]
        prices = await coingecko.get_price(syms)
        for c in top10:
            if c["symbol"] in prices:
                c["change_24h"] = round(prices[c["symbol"]].get("change_24h", 0.0), 2)
                c["price"]      = prices[c["symbol"]].get("price", 0.0)
    except Exception:
        pass

    return templates.TemplateResponse(request, "modules/section_2_coins.html", {
        "top5":         top5,
        "fgi_value":    fgi.get("value", 50),
        "fgi_class":    fgi.get("class", "NEUTRAL"),
        "top10_coins":  top10,
        "custom_coins": [],
    })

@router.get("/module/section_3", response_class=HTMLResponse)
async def section_3(request: Request):
    return templates.TemplateResponse(request, "modules/section_3_coins.html", {
        "top10_coins": [
            {"symbol": "BTC"}, {"symbol": "ETH"},  {"symbol": "SOL"},
            {"symbol": "BNB"}, {"symbol": "DOGE"}, {"symbol": "AVAX"},
            {"symbol": "LINK"},{"symbol": "ARB"},  {"symbol": "OP"},
            {"symbol": "SUI"},
        ],
        "custom_coins": [],
    })

@router.get("/module/section_4", response_class=HTMLResponse)
async def section_4(request: Request):
    return templates.TemplateResponse(
        request, "modules/section_4_chart_sentiment.html", {
            "fgi_value": 45,
            "fgi_class": "FEAR",
        }
    )

@router.get("/module/section_5", response_class=HTMLResponse)
async def section_5(request: Request):
    paper = state.get_open_positions("paper")
    live  = state.get_open_positions("live")
    return templates.TemplateResponse(
        request, "modules/section_5_positions.html", {
            "paper_positions":  paper,
            "live_positions":   live,
        }
    )

@router.get("/module/section_6", response_class=HTMLResponse)
async def section_6(request: Request):
    from ..engine import engine
    return templates.TemplateResponse(request, "modules/section_6_status.html", {
        "engine_running": engine._running,
        "mode":           config.effective_mode,
        "strategy":       config.DEFAULT_STRATEGY,
        "symbols_loaded": len(engine._top_symbols),
        "loop_interval":  config.trading.loop_interval_sec,
    })

@router.get("/module/section_5a", response_class=HTMLResponse)
async def section_5a(request: Request):
    all_trading = state.get_logs("trading", 30) or []
    paper_logs  = [l for l in all_trading if l.get("mode") != "live"][:15]
    return templates.TemplateResponse(request, "modules/section_5a_log_paper.html", {
        "paper_logs": paper_logs,
    })

@router.get("/module/section_5b", response_class=HTMLResponse)
async def section_5b(request: Request):
    all_trading = state.get_logs("trading", 30) or []
    live_logs   = [l for l in all_trading if l.get("mode") == "live"][:15]
    return templates.TemplateResponse(request, "modules/section_5b_log_live.html", {
        "live_logs": live_logs,
    })

@router.get("/module/section_4a", response_class=HTMLResponse)
async def section_4a(request: Request):
    return templates.TemplateResponse(request, "modules/section_4a_positions.html", {
        "paper_positions":  state.get_open_positions("paper"),
    })

@router.get("/module/section_4b", response_class=HTMLResponse)
async def section_4b(request: Request):
    return templates.TemplateResponse(request, "modules/section_4b_positions.html", {
        "live_positions":  state.get_open_positions("live"),
    })

@router.get("/module/section_1b", response_class=HTMLResponse)
async def section_1b(request: Request):
    return templates.TemplateResponse(request, "modules/section_1b_trade.html", {
        "balance":      round(state.get_paper_balance(), 2),
        "live_balance": round(state.get_live_balance(), 2),
        "pnl":          0.0,
        "ai_enabled":   config.AI_ENABLED,
        "ai_trade":     config.AI_TRADE_ENABLED,
        "system_logs":  state.get_logs("system", 20) or [],
    })

# Togglei AI

@router.post("/settings/ai/sys", response_class=HTMLResponse)
async def toggle_ai_sys():
    config.AI_ENABLED = not config.AI_ENABLED
    if not config.AI_ENABLED:
        config.AI_LEARN_ENABLED = False
        config.AI_TRADE_ENABLED = False
        config.MODE = "paper"
        state.add_log("system", "WARNING", component="ai",
                      message="HAI SYS OFF — Learn/Trade off, tryb Paper")
    else:
        state.add_log("system", "INFO", component="ai",
                      message="HAI SYS ON")
    return HTMLResponse(str(config.AI_ENABLED))

@router.post("/settings/ai/learn", response_class=HTMLResponse)
async def toggle_ai_learn():
    if not config.AI_ENABLED:
        state.add_log("system", "WARNING", component="ai",
                      message="Turn On Power — Learn zablokowane")
        return HTMLResponse("blocked")
    config.AI_LEARN_ENABLED = not config.AI_LEARN_ENABLED
    state.add_log("system", "INFO", component="ai",
                  message=f"HAI LEARN: {'ON' if config.AI_LEARN_ENABLED else 'OFF'}")
    return HTMLResponse(str(config.AI_LEARN_ENABLED))

@router.post("/settings/ai/trade", response_class=HTMLResponse)
async def toggle_ai_trade():
    if not config.AI_ENABLED:
        state.add_log("system", "WARNING", component="ai",
                      message="Turn On Power — Trade zablokowane")
        return HTMLResponse("blocked")
    config.AI_TRADE_ENABLED = not config.AI_TRADE_ENABLED
    state.add_log("system", "INFO", component="ai",
                  message=f"HAI TRADE: {'ON' if config.AI_TRADE_ENABLED else 'OFF'}")
    return HTMLResponse(str(config.AI_TRADE_ENABLED))

@router.post("/settings/ai/trade_stop", response_class=HTMLResponse)
async def stop_ai_trade():
    config.AI_TRADE_ENABLED = False
    state.add_log("system", "INFO", component="ai", message="HAI TRADE STOP")
    return HTMLResponse("stopped")

@router.post("/settings/ai/model")
async def set_ai_model(data: dict):
    model = data.get("ai_model", "random_forest")
    config.ai.default_model = model
    state.add_log("system", "INFO", component="ai",
                  message=f"Silnik AI -> {model}")
    return {"status": "ok"}

@router.post("/settings/ai/mode")
async def set_ai_mode(data: dict):
    mode = data.get("ai_mode", "filter")
    config.ai.mode = mode
    state.add_log("system", "INFO", component="ai",
                  message=f"Tryb AI -> {mode}")
    return {"status": "ok"}

@router.post("/settings/mode")
async def set_mode(data: dict):
    config.MODE = data.get("mode", "paper")
    state.add_log("system", "INFO", component="core",
                  message=f"Tryb -> {config.MODE}")
    return {"status": "ok"}

# Logi

@router.post("/settings/logs/clear", response_class=HTMLResponse)
async def clear_logs():
    from ..database import SystemLog, SessionLocal
    db = SessionLocal()
    try:
        db.query(SystemLog).delete()
        db.commit()
    finally:
        db.close()
    return HTMLResponse('<div style="color:#666;">--- Logi wyczyszczone ---</div>')

@router.post("/settings/logs/clear/trading", response_class=HTMLResponse)
async def clear_trading_logs():
    from ..database import TradingLog, SessionLocal
    db = SessionLocal()
    try:
        db.query(TradingLog).delete()
        db.commit()
    finally:
        db.close()
    return HTMLResponse('<div style="color:#666;">--- Logi tradingowe wyczyszczone ---</div>')

@router.get("/settings/logs/download")
async def download_logs():
    logs = state.get_logs("system", 200) or []
    text = "\n".join([f"[{l['timestamp']}] {l['message']}" for l in logs])
    return PlainTextResponse(
        content=text, media_type="text/plain",
        headers={"Content-Disposition": "attachment;filename=system_logs.txt"},
    )

@router.get("/settings/logs/download/trading")
async def download_trading_logs():
    logs = state.get_logs("trading", 200) or []
    text = "\n".join([f"[{l['timestamp']}] {l['message']}" for l in logs])
    return PlainTextResponse(
        content=text, media_type="text/plain",
        headers={"Content-Disposition": "attachment;filename=trading_logs.txt"},
    )

@router.post("/settings/reset_paper")
async def reset_paper():
    """Usuwa wszystkie paper positions (open+closed) → saldo wraca do 500$."""
    from ..database import SessionLocal, Position
    db = SessionLocal()
    try:
        deleted = db.query(Position).filter(Position.mode == "paper").delete()
        db.commit()
        state.add_log("system", "INFO", component="state",
                      message=f"PAPER RESET: usunięto {deleted} pozycji → saldo 500$")
        return {"status": "ok", "deleted_positions": deleted,
                "new_balance": float(os.environ.get("PAPER_START_BALANCE", "200"))}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


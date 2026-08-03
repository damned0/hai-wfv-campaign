# ===========================================
# HAI_EPV Engine ver.10 Final — routes/dashboard.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: GET / — serwuje glowny dashboard (hai_v2.html) z auth.
# ===========================================
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd

router = APIRouter()


@router.get("/api/fear_greed")
async def fear_greed():
    """Fear & Greed index (ostatnia wartość z warehouse)."""
    try:
        f = Path("/root/ProjektHAI/warehouse/market-data/macro/fear_greed.parquet")
        if not f.exists():
            return {"value": None, "error": "brak danych"}
        df = pd.read_parquet(f)
        last = df.iloc[-1]
        v = float(last.get("value", last.get("fng_value", 0)))
        label = ("Ekstremalny strach" if v <= 25 else
                 "Strach" if v <= 45 else
                 "Neutralnie" if v <= 55 else
                 "Chciwość" if v <= 75 else "Ekstremalna chciwość")
        ts = str(last.get("timestamp", ""))[:10]
        return {"value": round(v), "label": label, "ts": ts}
    except Exception as e:
        return {"value": None, "error": str(e)}

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_path = Path(__file__).resolve().parent.parent / "templates" / "hai_v2.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return """
    <html>
    <head><title>HAI_EPV DEV</title></head>
    <body style="background:#111;color:#0f0;font-family:monospace;padding:2rem;">
        <h1>HAI_EPV DEV</h1>
        <p>Port 5010 ? dzia?a poprawnie.</p>
        <p><a href="/docs">/docs</a> | <a href="/ai/status">AI status</a></p>
    </body>
    </html>
    """

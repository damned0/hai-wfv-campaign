# ===========================================
# HAI_EPV Engine ver.10 Final — routes/trading.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: engine start/stop/status, risk/stats, ceny/OHLCV/symbole,
# backtest (quick/full/wfv) + status, watchdog, pozycje/saldo/zamkniecie
# (recznie/close-all/close-stale), diagnostics, ai/cache/refresh.
# ===========================================
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..engine import engine
from ..state import state
from ..config import config
from pathlib import Path
from datetime import datetime
import asyncio
import json

router = APIRouter()

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest_results"

# == Engine ==

@router.post("/engine/start")
async def engine_start():
    if engine._running:
        return {"status": "already_running"}
    asyncio.create_task(engine.start())
    state.add_log("system", "INFO", component="engine", message="Engine start przez API")
    return {"status": "ok", "message": "Engine startuje..."}

@router.post("/engine/stop")
async def engine_stop():
    if not engine._running:
        return {"status": "already_stopped"}
    await engine.stop()
    return {"status": "ok", "message": "Engine zatrzymany"}

@router.get("/engine/status")
async def engine_status():
    return engine.get_status()

@router.get("/risk/stats")
async def risk_stats():
    from ..risk import risk
    return risk.get_daily_stats()


# == CoinGecko ==

@router.get("/api/price/{symbol}")
async def coin_price(symbol: str):
    from ..coingecko import coingecko
    detail = await coingecko.get_coin_detail(symbol)
    return detail or {"error": "Brak danych"}

@router.get("/api/fgi")
async def fgi():
    from ..coingecko import coingecko
    return await coingecko.get_fgi()

@router.get("/api/top100")
async def top100():
    from ..coingecko import coingecko
    return await coingecko.get_top100()


# == OHLCV z engine cache (Bitget) ==

@router.get("/api/ohlcv/{symbol:path}")
async def ohlcv_data(symbol: str, tf: str = "1H", limit: int = 100):
    """
    OHLCV z engine cache. Symbol w formacie BTC/USDT:USDT lub BTCUSDT.
    tf: 1H, 4H, 1D
    """
    # Normalizuj symbol
    sym = symbol.replace("%2F", "/").replace("%3A", ":")
    if "/" not in sym:
        sym = sym.replace("USDT", "/USDT:USDT")

    store = {
        "1H": engine._price_history_1h,
        "4H": engine._price_history_4h,
        "1D": engine._price_history_1d,
    }.get(tf.upper(), engine._price_history_1h)

    candles = store.get(sym, [])

    # Jesli brak w cache — pobierz na zywo
    if not candles:
        try:
            raw = await engine._exchange.fetch_ohlcv(sym, tf, limit)
            if raw:
                candles = engine._parse_candles(raw)
        except Exception:
            pass

    return {
        "symbol":  sym,
        "tf":      tf,
        "candles": candles[-limit:],
    }

@router.get("/api/symbols")
async def symbols_list():
    """Lista dostepnych symboli z engine."""
    return {
        "symbols": engine._top_symbols[:100],
        "prices":  {k: v for k, v in list(engine._prices.items())[:100]},
    }


# == AI ==




# == Backtest (bez duplikatow!) ==

@router.post("/backtest/quick")
async def backtest_quick(symbol: str = "BTC/USDT:USDT", days: int = 30):
    from ..backtester import backtester
    asyncio.create_task(backtester.quick_backtest(symbol, days))
    return {"status": "ok", "message": f"Backtest start: {symbol} {days}d"}

@router.post("/backtest/full")
async def backtest_full(days: int = 90, mode: str = "neutral", regime_adaptive: bool = False):
    """`regime_adaptive` (audyt 2026-07-05, "napisz nowa strategie trzymajaca
    sie doktryn") - gdy True, w regime=0 (trend_following) zamiast ekstremum
    BB (mean-reversion) uzywana jest strefa KONTYNUACJI (kupuj dolek w
    trendzie zamiast grac powrot do sredniej). Domyslnie False = zero zmiany."""
    from ..backtester import backtester
    from ..backtester import backtester as _bt_mod
    from ..state import state

    # Guard (audyt 2026-07-06) - _REGIME_ADAPTIVE_MODE jest globalna zmienna
    # modulowa czytana w toku (przez minuty) przez run_simulation_ai(). Drugi
    # request na TEJ SAMEJ instancji w trakcie pierwszego mogl ja cicho
    # nadpisac w polowie liczenia - czesc symboli policzy sie ze stara
    # wartoscia, czesc z nowa, bez zadnego bledu. Blokujemy zamiast pozwalac.
    if isinstance(backtester.stats, dict) and backtester.stats.get("status") == "running":
        return {"status": "error", "message": "Backtest juz trwa na tej instancji - odczekaj albo sprawdz /backtest/status"}

    _bt_mod._REGIME_ADAPTIVE_MODE = regime_adaptive

    async def _run():
        backtester.stats = {"status": "running", "days": days, "mode": mode,
                            "regime_adaptive": regime_adaptive,
                            "started_at": datetime.now().isoformat()}
        try:
            result = await backtester.run_full_ai(days=days, mode=mode)
            pnl    = result.get("total_pnl_usdt", 0)
            pf     = result.get("profit_factor", 0)
            wr     = result.get("win_rate", 0)
            trades = result.get("total_trades", 0)

            # Auto-save to file
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts    = datetime.now().strftime("%Y%m%d_%H%M")
            fname = f"backtest_{days}d_{mode}_{ts}.json"
            save  = {"saved_at": datetime.now().isoformat(), "days": days, "mode": mode, **result}
            # default=str - audyt 2026-07-06, ten sam wzorzec co numpy.bool_ w
            # _wfv_verdict() (naprawione dzis w nocy) - wynik moze zawierac
            # numpy.float64/int64 z obliczen w run_simulation_ai(), json.dumps
            # bez tego bezpiecznika wywala sie "Object of type X is not JSON
            # serializable" przy pierwszym niefortunnym typie.
            (_RESULTS_DIR / fname).write_text(json.dumps(save, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            backtester.stats = {**result, "saved_file": fname}
            state.add_log("ai", "INFO", event="BACKTEST",
                          message=f"Backtest {days}d/{mode} | PnL={pnl:+.2f}$ PF={pf:.2f} WR={wr:.1f}% trades={trades} → {fname}")
        except Exception as e:
            backtester.stats = {"status": "error", "error": str(e)}
            state.add_log("ai", "ERROR", event="BACKTEST", message=f"Backtest error: {e}")

    asyncio.create_task(_run())
    return {"status": "ok", "message": f"Full backtest AI start: {days}d {mode}"}

@router.get("/backtest/status")
async def backtest_status():
    from ..backtester import backtester
    return backtester.stats or {"status": "idle"}

_wfv_state = {"status": "idle"}

@router.post("/backtest/wfv")
async def backtest_wfv(n_windows: int = 6, window_days: int = 90, embargo_days: int = 7, mode: str = "neutral",
                        regime_adaptive: bool = False, voting_mode: str = "weighted",
                        decision_threshold: float = 0.35, conf_sizing: bool = False,
                        meta_label: bool = False,
                        threshold_long: float = None, threshold_short: float = None,
                        consensus_min: int = 0, consensus_sizing: bool = False,
                        vote_gate: float = 0.52, doctrine_free: bool = False,
                        model_config: str = None, include_trade_log: bool = False,
                        core_v3_profile: str = None):
    """Trigger Walk-Forward Validation z dashboardu (audyt 2026-07-04 - wczesniej
    WFV dalo sie odpalic tylko przez standalone skrypt, bez sladu w dashboardzie).
    `regime_adaptive` - patrz backtest_full(), ta sama logika.
    `voting_mode` (audyt 2026-07-06, "testy mechanizmow glosowania") - "weighted"
    (domyslny) lub "majority" (kazdy model = 1 glos, patrz backtester.py).
    `decision_threshold` (audyt 2026-07-07, "5. mechanizm: threshold sweep") -
    domyslnie 0.40, nadpisuje prog decyzyjny long_score/short_score.
    `conf_sizing` (audyt 2026-07-07, "confidence-based position sizing", z
    planu Opusa) - domyslnie False, wlacza skalowanie wielkosci pozycji
    marginesem pewnosci ponad prog (0.5x-1.5x).
    `meta_label` (audyt 2026-07-07) - domyslnie False, ustawia config.META_LABEL_ENABLED
    na czas tego przebiegu (przywracane po zakonczeniu). Uwaga: wczesniej testowany
    2x z konkluzja "zero efektu" - ten test laczy go pierwszy raz z conf_sizing."""
    from ..backtester import backtester
    from ..backtester import backtester as _bt_mod
    from ..state import state
    from ..config import config as _cfg

    # Guard (audyt 2026-07-06) - patrz komentarz w backtest_full() powyzej,
    # ten sam race condition ryzyko dla WFV (trwa jeszcze dluzej, wieksze
    # okno podatnosci).
    if _wfv_state.get("status") == "running":
        return {"status": "error", "message": "WFV juz trwa na tej instancji - odczekaj albo sprawdz /backtest/wfv/status"}

    # model_config (2026-07-12): BRAK tego parametru byl przyczyna falszywych
    # wynikow — wfv_runner.py wysylal &model_config=..., FastAPI po cichu
    # ignorowal, WFV liczyl na AKTUALNIE zaladowanym ensemble instancji,
    # a wynik dostawal etykiete configu, ktorego nikt nie testowal.
    # Mechanizm jak w HAI_DEV/routes/trading.py:153 (env + reload).
    if model_config:
        import os as _os_switch
        _os_switch.environ["HAI_MODEL_CONFIG"] = model_config
        from ..ensemble import ensemble as _ens_switch
        _ens_switch.load_models()

    # core_v3_profile (przywrocone przy F0 gen.Dir-v1, 2026-07-18 - zgubione
    # przy migracji do hai_common; semantyka 1:1 z AD*/routes/trading.py:
    # per-request wstrzykniecie mnoznika reżimowego dla et_h72_COREv2_iter2
    # do REGIME_WEIGHTS, reset przed KAZDYM przebiegiem, rowniez gdy None -
    # zero wyciekow stanu miedzy runami. UWAGA: statyczny wpis "low" dodany
    # na stale do REGIME_WEIGHTS (LAB live, 2026-07-11) jest przez ten reset
    # nadpisywany TYLKO w procesie, ktory dostal requesta WFV z tym parametrem
    # - instancje live (LAB/EPV) nie wolaja tego endpointu, wiec ich stan
    # zostaje nietkniety.
    from ..ensemble import REGIME_WEIGHTS as _RW3
    for _reg in _RW3:
        _RW3[_reg].pop("et_h72_COREv2_iter2", None)
    if core_v3_profile:
        _V3_PROFILES = {
            "low":  {0: 1.0, 1: 0.3, 2: 0.5},
            "med":  {0: 1.5, 1: 0.6, 2: 0.8},
            "high": {0: 2.5, 1: 0.3, 2: 0.5},
        }
        _prof = _V3_PROFILES.get(core_v3_profile, {})
        for _reg, _mult in _prof.items():
            _RW3.setdefault(_reg, {})["et_h72_COREv2_iter2"] = _mult

    _bt_mod._REGIME_ADAPTIVE_MODE = regime_adaptive
    _bt_mod._VOTING_MODE = voting_mode
    _bt_mod._DECISION_THRESHOLD = decision_threshold
    _bt_mod._CONF_SIZING_ENABLED = conf_sizing
    _bt_mod._THRESHOLD_LONG = threshold_long
    _bt_mod._THRESHOLD_SHORT = threshold_short
    _bt_mod._CONSENSUS_MIN = consensus_min
    _bt_mod._CONSENSUS_SIZING = consensus_sizing
    _bt_mod._VOTE_GATE = vote_gate
    _bt_mod._DOCTRINE_FREE = doctrine_free
    _bt_mod._INCLUDE_TRADE_LOG = include_trade_log
    _cfg.META_LABEL_ENABLED = meta_label

    async def _run():
        global _wfv_state
        _wfv_state = {"status": "running", "n_windows": n_windows, "window_days": window_days, "mode": mode,
                      "regime_adaptive": regime_adaptive, "voting_mode": voting_mode, "model_config": model_config,
                      "decision_threshold": decision_threshold, "conf_sizing": conf_sizing,
                      "meta_label": meta_label, "consensus_min": consensus_min,
                      "consensus_sizing": consensus_sizing, "vote_gate": vote_gate,
                      "doctrine_free": doctrine_free,
                      "started_at": datetime.now().isoformat()}
        state.add_log("ai", "INFO", event="WFV",
                      message=f"WFV start: {n_windows} okien x {window_days}d, embargo={embargo_days}d, mode={mode}")
        try:
            result = await backtester.run_wfv(n_windows=n_windows, window_days=window_days,
                                              embargo_days=embargo_days, mode=mode)
            verdict = result.get("verdict", {})

            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            fname = f"wfv_{n_windows}w{window_days}d_{mode}_{ts}.json"
            # Pelna prowieniencja przebiegu (audyt 2026-07-07, na prosbe usera
            # "odnotuj na jakich danych byl robiony wfv threshold itd") -
            # wczesniej pliki wynikowe NIE zapisywaly parametrow uruchomienia,
            # tylko wynik. Teraz kazdy .json ma run_params: co, na czym, z jakimi
            # progami/mechanizmami - zeby wynik byl odtwarzalny i porownywalny.
            from ..ensemble import ensemble as _ens_p
            from ..config import config as _cfg_p
            import os as _os_p
            run_params = {
                "n_windows": n_windows, "window_days": window_days, "embargo_days": embargo_days,
                "mode": mode, "regime_adaptive": regime_adaptive, "voting_mode": voting_mode,
                "decision_threshold": decision_threshold,
                "threshold_long": threshold_long, "threshold_short": threshold_short,
                "conf_sizing": conf_sizing, "meta_label": meta_label,
                "consensus_min": consensus_min, "consensus_sizing": consensus_sizing,
                "vote_gate": vote_gate, "doctrine_free": doctrine_free,
                "include_trade_log": include_trade_log, "core_v3_profile": core_v3_profile,
                "model_config": _os_p.getenv("HAI_MODEL_CONFIG"),
                "models": sorted(_ens_p.models.keys()) if _ens_p.models else [],
                "model_weights": dict(_ens_p.weights) if _ens_p.weights else {},
                "n_symbols": result.get("symbols"),
                "instance": _os_p.path.basename(str(Path(__file__).resolve().parent.parent)),
                "tp_atr": getattr(_cfg_p, "TAKE_PROFIT_ATR", None),
                "sl_atr": getattr(_cfg_p, "STOP_LOSS_ATR", None),
            }
            save = {"saved_at": datetime.now().isoformat(), "run_params": run_params, **result}
            # default=str - patrz komentarz w backtest_full() powyzej
            (_RESULTS_DIR / fname).write_text(json.dumps(save, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            _wfv_state = {**result, "saved_file": fname}
            from ..ensemble import ensemble as _ens
            from routes.ctrl import _log_to_file
            wagi = ", ".join(f"{m}={w:.3f}" for m, w in sorted((_ens.weights or {}).items(), key=lambda x: -x[1]))
            windows_str = " | ".join(f"{w.get('window')}:PF{w.get('profit_factor')}" for w in result.get("windows", []))
            msg = (f"WFV koniec: decision={verdict.get('decision')} avg_pf={verdict.get('avg_pf')} "
                   f"max_dd={verdict.get('max_dd')} weak_windows={verdict.get('weak_windows')} → {fname} "
                   f"| wagi: {wagi} | okna: {windows_str}")
            state.add_log("ai", "INFO", event="WFV", message=msg)
            _log_to_file(msg)
        except Exception as e:
            _wfv_state = {"status": "error", "error": str(e)}
            state.add_log("ai", "ERROR", event="WFV", message=f"WFV error: {e}")
            try:
                from routes.ctrl import _log_to_file as _ltf
                _ltf(f"WFV error: {e}")
            except Exception:
                pass

    asyncio.create_task(_run())
    return {"status": "ok", "message": f"WFV start: {n_windows} okien x {window_days}d, mode={mode}"}

@router.get("/backtest/wfv/status")
async def backtest_wfv_status():
    from ..backtester import backtester
    out = dict(_wfv_state)
    if out.get("status") == "running":
        out["current_window"] = getattr(backtester, "wfv_progress", None)
    return out

@router.get("/backtest/results")
async def backtest_results_list():
    if not _RESULTS_DIR.exists():
        return {"files": []}
    files = sorted(_RESULTS_DIR.glob("backtest_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files[:30]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "file":     f.name,
                "days":     d.get("days"),
                "mode":     d.get("mode"),
                "pnl":      d.get("total_pnl_usdt"),
                "wr":       d.get("win_rate"),
                "pf":       d.get("profit_factor"),
                "trades":   d.get("total_trades"),
                "saved_at": d.get("saved_at"),
            })
        except Exception:
            out.append({"file": f.name})
    return {"files": out}

@router.get("/backtest/results/{filename}")
async def backtest_result_download(filename: str):
    if not filename.startswith("backtest_") or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    fpath = _RESULTS_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(fpath), media_type="application/json", filename=filename)


# == Watchdog / cache ==

async def _read_master_status_json(timeout: float = 3.0):
    """Reads master_status.json from SSHFS with timeout — never blocks event loop."""
    from pathlib import Path
    import json
    _path = Path("/mnt/asustor/warehouse_v2/meta/master_status.json")
    def _read():
        if not _path.exists():
            return None
        with open(_path) as f:
            return json.load(f)
    return await asyncio.wait_for(asyncio.to_thread(_read), timeout=timeout)


@router.get("/watchdog/status")
async def watchdog_status():
    """v5.0 LIV: Czyta status z Asustor watchdog (master_status.json)."""
    from datetime import datetime, timezone

    try:
        data = await _read_master_status_json()
    except asyncio.TimeoutError:
        return {
            "enabled": False,
            "status": "SSHFS_TIMEOUT",
            "reason": "SSHFS mount nie odpowiada (timeout 3s)",
            "last_check": None,
            "restart_by": "N/A",
        }

    if data is None:
        return {
            "enabled": False,
            "status": "DISABLED",
            "reason": "Asustor master_status.json not found",
            "last_check": None,
            "restart_by": "N/A",
        }

    try:
        updated_at = data.get("updated_at", "")
        version = data.get("watchdog_version", "?")
        summary = data.get("summary", {})

        is_fresh = False
        if updated_at:
            try:
                ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
                is_fresh = age_min < 15
            except Exception:
                pass

        instances_up = summary.get("instances_up", 0)
        instances_down = summary.get("instances_down", 0)
        total = instances_up + instances_down

        return {
            "enabled": is_fresh,
            "status": "ACTIVE" if is_fresh else "STALE",
            "reason": f"Asustor watchdog v{version} - monitoring {total} instances",
            "last_check": updated_at,
            "restart_by": "Asustor (87.205.4.25)",
            "instances_up": instances_up,
            "instances_down": instances_down,
            "version": version,
        }
    except Exception as e:
        return {
            "enabled": False,
            "status": "ERROR",
            "reason": str(e)[:200],
            "last_check": None,
        }

@router.get("/ohlcv/stats")
async def ohlcv_stats():
    from ..ohlcv_cache import ohlcv_cache
    return ohlcv_cache.get_stats()

@router.get("/redis/info")
async def redis_info():
    from ..redis_cache import redis_cache
    return redis_cache.get_info()

@router.post("/redis/flush")
async def redis_flush():
    from ..redis_cache import redis_cache
    redis_cache.flush()
    return {"status": "ok"}


# == Balance / Positions ==

@router.get("/api/balance")
async def api_balance():
    from ..risk import risk
    balance = state.get_paper_balance()
    stats   = risk.get_daily_stats()
    unrealized = 0.0
    try:
        for pos in state.get_open_positions("paper"):
            sym = pos["symbol"]
            cur = engine._prices.get(sym)
            if cur is None:
                h1 = engine._price_history_1h.get(sym, [])
                if not h1 or h1[-1].get("close") is None:
                    continue
                cur = h1[-1]["close"]
            entry = pos["entry_price"]
            # FIX: SHORT PnL
            if pos["side"] == "SHORT":
                unrealized += (entry - cur) * pos["size_coins"]
            else:
                unrealized += (cur - entry) * pos["size_coins"]
    except Exception:
        pass
    # Koszty wejścia na otwartych (fee taker + slippage entry), żeby saldo wg pozycji
    # odzwierciedlało realny equity — bez kosztów wyjścia (jeszcze nie zamknięta).
    try:
        for pos in state.get_open_positions("paper"):
            notch = float(pos["size_usdt"])
            unrealized -= notch * (config.trading.fee_taker + config.trading.slippage_entry)
    except Exception:
        pass
    return {
        "paper":          round(balance + unrealized, 2),
        "realized":       round(balance, 2),
        "unrealized":     round(unrealized, 2),
        "pnl":            stats["daily_pnl"],
        "open_positions": stats["open_positions"],
    }

@router.get("/api/positions")
async def api_positions():
    positions = state.get_open_positions("paper")
    for pos in positions:
        sym = pos["symbol"]
        cur = engine._prices.get(sym)

        # Fallback — pobierz cenę przez REST jeśli nie ma w WS
        if not cur:
            try:
                import asyncio
                if engine._exchange:
                    ticker = await engine._exchange._safe_call("fetch_ticker", sym)
                    if ticker:
                        cur = ticker.get("last") or ticker.get("close")
            except Exception:
                cur = None

        if cur:
            entry = pos["entry_price"]
            if pos["side"] == "SHORT":
                pnl_pct = ((entry - cur) / entry) * 100  # v5.0: spot only, spojne z SL
                pnl     = (entry - cur) * pos["size_coins"]
            else:
                pnl_pct = ((cur - entry) / entry) * 100  # v5.0: spot only, spojne z SL
                pnl     = (cur - entry) * pos["size_coins"]
            pos["pnl_pct"]       = round(pnl_pct, 2)
            pos["pnl"]           = round(pnl, 2)
            pos["current_price"] = round(cur, 6)
    return positions

@router.get("/api/closed")
async def api_closed(limit: int = 20):
    """Ostatnie zamkniete pozycje."""
    return state.get_closed_pnl(limit=limit, mode="paper")


# == Manual Trading ==

@router.post("/trade/open")
async def trade_open(symbol: str, side: str, leverage: int = 5):
    """Reczne otwarcie pozycji."""
    from ..risk import risk
    side = side.upper()
    if side not in ("LONG", "SHORT"):
        return {"status": "error", "message": "side musi byc LONG lub SHORT"}

    sym = symbol.replace("%2F", "/").replace("%3A", ":")

    # Pobierz cene
    price = engine._prices.get(sym, 0)
    if not price:
        return {"status": "error", "message": f"Brak ceny dla {sym}"}

    check = risk.check_entry(sym, side, price)
    if not check.allowed:
        return {"status": "error", "message": check.reason}

    size_coins, size_usdt = risk.calculate_position_size(price, leverage)
    pos_id = state.open_position(
        symbol=sym, side=side,
        entry_price=price,
        size_coins=size_coins,
        size_usdt=size_usdt,
        leverage=leverage,
        strategy="manual",
        reason=f"Manual {side} @ {price}",
    )
    state.add_log("trading", "SIGNAL", symbol=sym,
                  action="OPEN", mode="paper",
                  message=f"MANUAL OPEN {side} {sym} @ {price}")
    return {"status": "ok", "position_id": pos_id, "price": price, "side": side}

@router.post("/trade/close")
async def trade_close(symbol: str):
    """Reczne zamkniecie pozycji."""
    sym   = symbol.replace("%2F", "/").replace("%3A", ":")
    price = engine._prices.get(sym, 0)
    # FIX v5: fallback REST jesli brak w WS cache
    if not price and engine._exchange:
        try:
            ticker = await engine._exchange._safe_call("fetch_ticker", sym)
            if ticker:
                price = ticker.get("last") or ticker.get("close") or 0
        except Exception:
            pass
    if not price:
        return {"status": "error", "message": f"Brak ceny dla {sym}"}

    result = state.close_position(sym, price, reason="Manual close")
    if not result:
        return {"status": "error", "message": f"Brak otwartej pozycji dla {sym}"}

    from ..risk import risk
    risk.record_trade(result["pnl"])
    state.add_log("trading", "SIGNAL", symbol=sym,
                  action="CLOSE", mode="paper",
                  message=f"MANUAL CLOSE {sym} @ {price} | PnL: {result['pnl']}")
    return {"status": "ok", "pnl": result["pnl"], "pnl_pct": result["pnl_pct"]}

@router.post("/trading/close_all")
async def close_all_positions():
    """Awaryjne zamkniecie WSZYSTKICH otwartych pozycji (audyt 2026-07-05 -
    'Stop All' na dashboardzie wolal endpoint ktory nigdy nie istnial,
    naprawione zeby faktycznie dzialalo)."""
    from ..risk import risk
    open_positions = list(state.get_open_positions())
    closed, errors = [], []
    for pos in open_positions:
        sym = pos["symbol"]
        price = engine._prices.get(sym, 0)
        if not price and engine._exchange:
            try:
                ticker = await engine._exchange._safe_call("fetch_ticker", sym)
                if ticker:
                    price = ticker.get("last") or ticker.get("close") or 0
            except Exception:
                pass
        if not price:
            errors.append({"symbol": sym, "message": "brak ceny"})
            continue
        result = state.close_position(sym, price, reason="Emergency close-all")
        if not result:
            errors.append({"symbol": sym, "message": "brak otwartej pozycji"})
            continue
        risk.record_trade(result["pnl"])
        state.add_log("trading", "SIGNAL", symbol=sym, action="CLOSE", mode="paper",
                      message=f"EMERGENCY CLOSE {sym} @ {price} | PnL: {result['pnl']}")
        closed.append({"symbol": sym, "pnl": result["pnl"]})
    return {"status": "ok", "closed": closed, "errors": errors, "count": len(closed)}


@router.get("/trade/stale-positions")
async def stale_positions():
    """Lista pozycji otwartych spoza whitelist."""
    from ..symbols import is_supported
    from ..state import state
    from ..engine import engine

    open_positions = state.get_open_positions()
    stale = []

    for pos in open_positions:
        symbol = pos["symbol"]
        if not is_supported(symbol):
            last_price = engine._prices.get(symbol, pos["entry_price"])
            entry = pos["entry_price"]
            if pos["side"] == "SHORT":
                pnl_pct = ((entry - last_price) / entry) * 100
            else:
                pnl_pct = ((last_price - entry) / entry) * 100
            stale.append({
                **pos,
                "last_price": last_price,
                "estimated_pnl_pct": round(pnl_pct, 2),
                "tracked_by_engine": symbol in engine._prices,
            })

    return {
        "count": len(stale),
        "stale": stale,
        "whitelist_size": 30,
    }


@router.post("/trade/close-stale")
async def close_stale_positions(dry_run: bool = False):
    """Zamyka wszystkie pozycje spoza whitelist po ostatniej znanej cenie."""
    from ..symbols import is_supported
    from ..state import state
    from ..engine import engine
    from ..risk import risk

    open_positions = state.get_open_positions()
    actions = []

    for pos in open_positions:
        symbol = pos["symbol"]
        if is_supported(symbol):
            continue

        last_price = engine._prices.get(symbol, pos["entry_price"])

        if dry_run:
            actions.append({
                "symbol": symbol,
                "side": pos["side"],
                "entry": pos["entry_price"],
                "exit": last_price,
                "action": "WOULD_CLOSE",
            })
            continue

        result = state.close_position(
            symbol, last_price, pos["mode"],
            reason=f"STALE_CLEANUP @ {last_price}"
        )
        if result:
            risk.record_trade(result["pnl"])
            actions.append({
                "symbol": symbol,
                "side": pos["side"],
                "entry": pos["entry_price"],
                "exit": last_price,
                "pnl": result["pnl"],
                "action": "CLOSED",
            })

    return {
        "status": "dry_run" if dry_run else "ok",
        "count": len(actions),
        "actions": actions,
    }






# == Diagnostics + Collectors (v5.0 LIV) ==

@router.get("/diagnostics")
async def diagnostics():
    """Status pliku derivatives_cache.json."""
    from pathlib import Path
    import json
    from datetime import datetime, timezone

    cache_path = Path(__file__).resolve().parent.parent / "data" / "cache" / "derivatives_cache.json"
    result = {"derivatives_cache": {"exists": False}}

    if cache_path.exists():
        try:
            stat = cache_path.stat()
            with open(cache_path) as f:
                data = json.load(f)
            age_sec = datetime.now().timestamp() - stat.st_mtime
            # v5.0: cache ma strukture {updated_at, symbols: {...}, missing, count}
            sym_count = 0
            if isinstance(data, dict):
                if "count" in data:
                    sym_count = int(data["count"])
                elif "symbols" in data and isinstance(data["symbols"], dict):
                    sym_count = len(data["symbols"])
                else:
                    sym_count = len(data)
            result["derivatives_cache"] = {
                "exists": True,
                "age_hours": round(age_sec / 3600, 2),
                "size_bytes": stat.st_size,
                "symbols": sym_count,
            }
        except Exception as e:
            result["derivatives_cache"]["error"] = str(e)[:100]
    return result


@router.post("/ai/cache/refresh")
async def refresh_deriv_cache():
    """Buduje derivatives_cache.json z warehouse (funding_rates + open_interest)."""
    import asyncio
    async def _build():
        try:
            from pathlib import Path
            import json, numpy as np, pandas as pd
            from datetime import datetime, timezone

            WH = Path("/root/ProjektHAI/data_warehouse/derivatives")
            CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "derivatives_cache.json"
            CACHE.parent.mkdir(parents=True, exist_ok=True)

            fr_dir = WH / "funding_rates"
            oi_dir = WH / "open_interest"
            symbols = sorted(set(
                p.stem for p in fr_dir.glob("*.parquet")
            ) | set(p.stem for p in oi_dir.glob("*.parquet")))

            result, missing = {}, []
            for sym in symbols:
                try:
                    fr_path = fr_dir / f"{sym}.parquet"
                    oi_path = oi_dir / f"{sym}.parquet"
                    fr_row = oi_row = None
                    fund, fund_prev, fund_ext = 0.0, 0.0, 0.0
                    oi_cur, oi_prev, oi_zscore = 0.0, 0.0, 0.0

                    if fr_path.exists():
                        df_f = pd.read_parquet(fr_path).dropna()
                        if len(df_f) >= 2:
                            col = "funding_rate" if "funding_rate" in df_f.columns else "close"
                            fund      = float(df_f[col].iloc[-1])
                            fund_prev = float(df_f[col].iloc[-2])
                            fund_ext  = float(abs(fund) > 0.05)

                    if oi_path.exists():
                        df_o = pd.read_parquet(oi_path).dropna()
                        if len(df_o) >= 2:
                            col = "close"
                            oi_cur  = float(df_o[col].iloc[-1])
                            oi_prev = float(df_o[col].iloc[-2])
                            vals30  = df_o[col].iloc[-30:].values
                            mu, sd  = vals30.mean(), vals30.std()
                            oi_zscore = float((oi_cur - mu) / sd) if sd > 0 else 0.0

                    result[sym] = {
                        "oi_total_log":      float(np.log1p(oi_cur)),
                        "oi_change_24h":     float((oi_cur - oi_prev) / oi_prev) if oi_prev > 0 else 0.0,
                        "oi_zscore_30d":     round(oi_zscore, 4),
                        "funding_rate":      round(fund * 100, 6),
                        "funding_change_24h": round((fund - fund_prev) * 100, 6),
                        "funding_extreme":   fund_ext,
                    }
                except Exception:
                    missing.append(sym)

            cache_data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "count": len(result),
                "symbols": result,
                "missing": missing,
            }
            with open(CACHE, "w") as f:
                json.dump(cache_data, f)

            from ..state import state
            state.add_log("ai", "INFO", event="CACHE_REFRESH",
                          message=f"Deriv cache rebuilt: {len(result)} symbols, {len(missing)} missing")
        except Exception as e:
            from ..state import state
            state.add_log("ai", "ERROR", event="CACHE_REFRESH", message=f"Cache refresh error: {e}")

    asyncio.create_task(_build())
    return {"status": "ok", "message": "Deriv cache rebuild started (from Binance warehouse)"}


@router.get("/system/collectors")
async def system_collectors():
    """Czyta master_status.json z Asustora."""
    try:
        data = await _read_master_status_json()
    except asyncio.TimeoutError:
        return {"error": "SSHFS mount nie odpowiada (timeout 3s)"}
    if data is None:
        return {"error": "master_status.json not found"}
    try:
        collectors = data.get("collectors", {})
        return {name: {"status": info.get("status", "unknown"), "age_h": info.get("age_h"), "last_run": info.get("last_run")} for name, info in collectors.items()}
    except Exception as e:
        return {"error": str(e)[:200]}


@router.post("/watchdog/send_report")
async def watchdog_send_report():
    """Wysylka pelnego raportu Watchdoga na Telegram."""
    import os
    import requests
    from datetime import datetime, timezone

    TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
    TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

    try:
        data = await _read_master_status_json()
    except asyncio.TimeoutError:
        return {"status": "error", "message": "SSHFS mount nie odpowiada (timeout 3s)"}
    if data is None:
        return {"status": "error", "message": "master_status.json not found"}

    try:

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"\U0001F988 *HAI Watchdog Report*", f"_{now}_", ""]

        instances = data.get("instances", {})
        summary = data.get("summary", {})
        up = summary.get("instances_up", 0)
        total = up + summary.get("instances_down", 0)
        lines.append(f"*INSTANCES ({up}/{total} UP)*")
        for short, info in instances.items():
            name = info.get("name", short)
            port = info.get("port", "?")
            status = info.get("status", "?")
            if status == "UP":
                paper = info.get("paper", 0)
                opn = info.get("open_positions", 0)
                pnl = info.get("pnl", 0)
                unreal = info.get("unrealized", 0)
                lines.append(f"  \u2705 `{name}` :{port}")
                lines.append(f"      ${paper:.2f} | open={opn} | PnL={pnl:+.2f} | unreal={unreal:+.2f}")
            else:
                lines.append(f"  \u274C `{name}` DOWN")
        lines.append("")

        collectors = data.get("collectors", {})
        fresh = summary.get("collectors_fresh", 0)
        stale = summary.get("collectors_stale", 0)
        missing = summary.get("collectors_missing", 0)
        total_col = fresh + stale + missing
        lines.append(f"*COLLECTORS ({fresh}/{total_col} FRESH)*")
        for name, info in collectors.items():
            status = info.get("status", "?")
            age = info.get("age_h")
            age_str = f"{age:.1f}h" if isinstance(age, (int, float)) else "?"
            icon = "\u2705" if status == "fresh" else "\u26A0\uFE0F"
            lines.append(f"  {icon} `{name}` {age_str}")
        lines.append("")

        version = data.get("watchdog_version", "?")
        lines.append(f"_Watchdog v{version} | Asustor 87.205.4.25_")

        msg = "\n".join(lines)

        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code == 200:
            return {"status": "ok", "message": "Report wyslany na Telegram", "chars": len(msg)}
        else:
            return {"status": "error", "message": f"Telegram {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:300]}


@router.get("/system/master")
async def system_master():
    """Pelny master_status.json z Asustora (instances + collectors + summary).
    Zrodlo 3 raportow Watchdog w dashboardzie. Watchdog v1.0 pisze co 5 min."""
    from datetime import datetime, timezone

    try:
        data = await _read_master_status_json()
    except asyncio.TimeoutError:
        return {"error": "SSHFS mount nie odpowiada (timeout 3s)", "exists": None}
    if data is None:
        return {"error": "master_status.json not found", "exists": False}
    try:
        updated_at = data.get("updated_at", "")
        age_min = None
        is_fresh = False
        if updated_at:
            try:
                ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
                is_fresh = age_min < 15
            except Exception:
                pass
        return {
            "exists": True,
            "fresh": is_fresh,
            "age_min": round(age_min, 1) if age_min is not None else None,
            "updated_at": updated_at,
            "watchdog_version": data.get("watchdog_version", "?"),
            "instances": data.get("instances", {}),
            "collectors": data.get("collectors", {}),
            "summary": data.get("summary", {}),
        }
    except Exception as e:
        return {"error": str(e)[:200], "exists": True}

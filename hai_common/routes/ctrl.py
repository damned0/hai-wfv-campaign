"""
HAI_EPV Engine ver.10 Final — routes/ctrl.py
Created by Hauzer | Coded & produced by Claude Sonnet 5

AI Control panel endpoints (universal) — dziala na EPV/DEV/LAB/LIV/TST,
autodetekcja modeli i neurali.

GET  /ctrl/status             — pełny stan panelu
POST /ctrl/model-toggle       — włącz/wyłącz model {model, enabled}
POST /ctrl/model-reload       — przeładuj wszystkie modele z dysku
POST /ctrl/filter             — ustaw filtr {key, value}
POST /ctrl/ai-trade           — toggle AI_TRADE_ENABLED {enabled}
POST /ctrl/ai-learn           — toggle AI_LEARN_ENABLED {enabled}
POST /ctrl/meta-label         — toggle meta-labeling filtr {enabled, threshold}
POST /ctrl/confidence-calib   — toggle kalibracja pewności {enabled}
POST /ctrl/screening          — odpal screening
POST /ctrl/backtest           — odpal backtest {days}
GET  /ctrl/backtest/last      — ostatni wynik backtestu
GET  /ctrl/screening/last     — zawartość ostatniego pliku screeningu
GET  /ctrl/neural/status      — status neurali (wg. możliwości instancji)
POST /ctrl/neural/toggle      — włącz/wyłącz neural generator (tylko TST/LIV-gen)
GET  /ctrl/zoo                — lista modeli w model_zoo
POST /ctrl/zoo/load-tree      — załaduj model z zoo
GET  /ctrl/data-sources       — pelna lista zrodel danych (gielda/API, swiezosc)
"""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ctrl", tags=["Control"])
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

_filters = {
    "confidence_min":  0.60,
    "doctrine":        True,
    "regime_blend":    True,
    "bb_pre_long":     0.22,
    "bb_pre_short":    0.78,
    "adx_min":         20,
}
_disabled_models: set = set()
_last_bt: dict = {}


def _save_bt(data: dict):
    try:
        import json as _json
        _BT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BT_FILE.write_text(_json.dumps(data, default=str))
        # Kopia z ladna nazwa + data (audyt 2026-07-04) - last_backtest.json
        # zostaje jako wskaznik "najnowszy", ale kazdy bieg ma tez wlasny,
        # trwaly plik zeby nie gubic historii poprzednich wynikow.
        _BT_HIST_DIR.mkdir(parents=True, exist_ok=True)
        ts_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        (_BT_HIST_DIR / f"bt_epv_{ts_name}.json").write_text(_json.dumps(data, default=str))
    except Exception as e:
        logger.warning("_save_bt error: %s", e)


def _load_bt() -> dict:
    try:
        import json as _json
        if _BT_FILE.exists():
            return _json.loads(_BT_FILE.read_text())
    except Exception:
        pass
    return {}

INST_DIR = Path(__file__).resolve().parent.parent
_BT_FILE = INST_DIR / "data" / "last_backtest.json"
_BT_HIST_DIR = INST_DIR / "data" / "backtests"
_AI_DECISIONS_LOG = INST_DIR / "logs" / "ai_decisions.log"


def _log_to_file(line: str):
    """Zapis do logs/ai_decisions.log (audyt 2026-07-04, na wyrazna prosbe) -
    obok wpisu do bazy (AI Log w dashboardzie), TAKZE plik tekstowy w
    katalogu logs/ instancji - do bezposredniej inspekcji bez dashboardu."""
    try:
        _AI_DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(_AI_DECISIONS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} {line}\n")
    except Exception as e:
        logger.warning("_log_to_file error: %s", e)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensemble():
    from ..ensemble import ensemble
    return ensemble


def _config():
    from ..config import config
    return config


def _model_names():
    """Autodetekcja nazw modeli bazowych (drzewa)."""
    try:
        from ..ensemble import TREE_MODELS
        return TREE_MODELS
    except ImportError:
        pass
    try:
        from ..ensemble import MODEL_NAMES
        return MODEL_NAMES
    except ImportError:
        pass
    # fallback — introspect z ensemble
    ens = _ensemble()
    if hasattr(ens, 'models') and ens.models:
        return list(ens.models.keys())
    return ["lgb", "rf", "xgb"]


def _inst_name():
    return INST_DIR.name  # np. "HAI_LIV"


# ── Neural helpers ────────────────────────────────────────────────────────────

def _neural_status_auto() -> dict:
    """Zwraca status neurali — adaptuje się do architektury instancji."""
    ens = _ensemble()

    # TST-style: ensemble ma neural_status()
    if hasattr(ens, 'neural_status') and callable(ens.neural_status):
        return ens.neural_status()

    # LIV-style: nbeats wbudowany w features.py z modelu w model_zoo
    zoo = Path("/root/ProjektHAI/model_zoo/store/neural")
    inst = _inst_name()
    result = {}
    if zoo.exists():
        import joblib
        for pkl in sorted(zoo.glob(f"{inst}__*.pkl"), reverse=True):
            parts = pkl.stem.split("__")  # HAI_LIV__nbeats__20260629...
            if len(parts) >= 2:
                arch = parts[1]
                if arch not in result:
                    try:
                        data = joblib.load(pkl)
                        entry = {
                            "loaded":     True,
                            "enabled":    True,
                            "n_params":   data.get("n_params", data.get("windows", 0)),
                            "trained_at": str(data.get("trained_at", ""))[:10],
                            "note":       "built-in feature (not toggleable)",
                        }
                        if "accuracy" in data:
                            # Klasyfikator (mlp/lstm/tcn/transformer_cls, np. EPV/DEV) —
                            # metryka to accuracy, nie MAE (MAE nie ma sensu dla klasyfikacji,
                            # a te modele nigdy nie zapisują val_mae/val_loss).
                            entry["metric"]      = "accuracy"
                            entry["accuracy"]    = round(float(data["accuracy"]), 4)
                        else:
                            # Model regresyjny (nbeats/transformer/patchtst/tft_lite, LIV/TST) —
                            # TST-format: val_mae, LIV-format: val_loss.
                            mae = data.get("val_mae") or data.get("val_loss")
                            entry["metric"]  = "val_mae"
                            entry["val_mae"] = round(float(mae), 5) if mae else 0.0
                        result[arch] = entry
                    except Exception:
                        result[arch] = {"loaded": False, "enabled": False}
    return result


# ── endpoints ────────────────────────────────────────────────────────────────

@router.get("/status")
async def ctrl_status():
    ens = _ensemble()
    if not ens.active:
        ens.load_models()
    cfg = _config()

    names = _model_names()
    models_info = {}
    for m in names:
        loaded = m in ens.models
        models_info[m] = {
            "loaded":     loaded,
            "enabled":    m not in _disabled_models,
            "acc":        round(ens.accuracies.get(m, 0.0), 4) if loaded else None,
            "weight":     round(ens.weights.get(m, 0.0), 4)    if loaded else None,
            "n_features": len(ens.feature_names.get(m, []))   if loaded else None,
        }

    n_feat_max = max((v["n_features"] or 0) for v in models_info.values()) if models_info else 0

    return {
        "models":  models_info,
        "filters": dict(_filters),
        "system": {
            "ai_trade":        cfg.AI_TRADE_ENABLED,
            "ai_learn":        cfg.AI_LEARN_ENABLED,
            "ensemble_active": ens.active,
            "ensemble_count":  len(ens.models),
            "n_features":      n_feat_max,
            "meta_label_enabled": cfg.META_LABEL_ENABLED,
            "meta_label_threshold": cfg.META_LABEL_THRESHOLD,
            "meta_label_available": (MODELS_DIR / "meta_label.pkl").exists(),
            "confidence_calib_enabled": cfg.CONFIDENCE_CALIB_ENABLED,
            "confidence_calib_available": (MODELS_DIR / "confidence_calib.pkl").exists(),
        },
    }


@router.post("/model-toggle")
async def model_toggle(model: str, enabled: bool):
    ens = _ensemble()
    names = _model_names()
    if model not in names:
        return {"status": "error", "message": f"Nieznany model: {model}"}

    if enabled:
        _disabled_models.discard(model)
        if model not in ens.models:
            ens._load_one(model)
            ens._recalc_weights()
    else:
        _disabled_models.add(model)
        ens.models.pop(model, None)
        ens.scalers.pop(model, None)
        ens.f1_scores.pop(model, None)
        ens.accuracies.pop(model, None)
        getattr(ens, 'feature_names', {}).pop(model, None)
        ens._recalc_weights()
        ens.active = len(ens.models) >= 2
        ens._cache.clear()

    logger.info("CTRL model-toggle: %s → %s", model, "ON" if enabled else "OFF")
    return {"status": "ok", "model": model, "enabled": enabled,
            "active_models": list(ens.models.keys())}


@router.post("/model-reload")
async def model_reload():
    ens = _ensemble()
    _disabled_models.clear()
    ens.load_models()
    logger.info("CTRL model-reload: przeładowano")
    return {"status": "ok", "models": list(ens.models.keys()), "weights": ens.weights}


@router.post("/filter")
async def set_filter(key: str, value: str):
    if key not in _filters:
        return {"status": "error", "message": f"Nieznany filtr: {key}"}
    try:
        old = _filters[key]
        if isinstance(old, bool):
            _filters[key] = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(old, int):
            _filters[key] = int(float(value))
        else:
            _filters[key] = float(value)
        if key == "confidence_min":
            try:
                _config().ai.confidence_min = _filters[key]
            except Exception:
                pass
        return {"status": "ok", "key": key, "value": _filters[key]}
    except (ValueError, TypeError) as e:
        return {"status": "error", "message": str(e)}


@router.post("/ai-trade")
async def toggle_ai_trade(enabled: bool):
    _config().AI_TRADE_ENABLED = enabled
    return {"status": "ok", "ai_trade": enabled}


@router.post("/meta-label")
async def toggle_meta_label(enabled: bool, threshold: float = None):
    """Vote-labeling post-hoc (audyt 2026-07-05) - filtr NAD juz wytrenowanym
    ensemble, dziala bez retrenu modeli bazowych. Wymaga data/models/
    meta_label.pkl (lub meta_label_diversified.pkl) - jesli brak pliku,
    wlaczenie nie zrobi nic (ensemble.predict sprawdza dostepnosc)."""
    cfg = _config()
    cfg.META_LABEL_ENABLED = enabled
    if threshold is not None:
        cfg.META_LABEL_THRESHOLD = max(0.0, min(1.0, threshold))
    return {"status": "ok", "meta_label_enabled": enabled, "meta_label_threshold": cfg.META_LABEL_THRESHOLD}


@router.post("/confidence-calib")
async def toggle_confidence_calib(enabled: bool):
    """Kalibracja pewnosci per model (Platt scaling) - przeskalowuje surowe
    wyjscie kazdego modelu, bez retrenu. Wymaga data/models/confidence_calib.pkl."""
    _config().CONFIDENCE_CALIB_ENABLED = enabled
    return {"status": "ok", "confidence_calib_enabled": enabled}


@router.post("/ai-learn")
async def toggle_ai_learn(enabled: bool):
    _config().AI_LEARN_ENABLED = enabled
    return {"status": "ok", "ai_learn": enabled}


@router.post("/screening")
async def ctrl_screening():
    import os, sys
    script = INST_DIR / "screen_parallel.py"
    if not script.exists():
        return {"status": "error", "message": "Brak screen_parallel.py"}
    logf = open(INST_DIR / "screen_run.log", "a")

    async def _run_and_log():
        from ..state import state
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), cwd=str(INST_DIR),
            env={**os.environ, "HAI_INSTANCE_DIR": str(INST_DIR)},
            stdout=logf, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await proc.wait()
        # Wyciagnij podsumowanie z pliku wynikowego (jesli powstal) - liczba
        # linii jako proxy liczby przeskanowanych/zakwalifikowanych symboli.
        inst_name_short = _inst_name().replace("HAI_", "")
        files = sorted(INST_DIR.glob(f"screening_{inst_name_short}_*.txt"), reverse=True)
        n_lines = 0
        fname = None
        if files:
            fname = files[0].name
            try:
                n_lines = len(files[0].read_text(encoding="utf-8").splitlines())
            except Exception:
                pass
        if rc == 0:
            msg = f"SCREENING zakonczony (kod={rc}) | plik={fname or '-'} | linii={n_lines}"
            state.add_log("ai", "INFO", event="SCREENING", message=msg)
        else:
            msg = f"SCREENING zakonczony z bledem (kod={rc})"
            state.add_log("ai", "ERROR", event="SCREENING", message=msg)
        _log_to_file(msg)

    asyncio.create_task(_run_and_log())
    return {"status": "ok", "message": "Screening uruchomiony w tle"}


@router.post("/backtest")
async def ctrl_backtest(days: int = 90):
    import importlib, sys
    _bt_mod = importlib.import_module("core.backtester")
    if not hasattr(_bt_mod, "backtester"):
        _bt_mod.backtester = _bt_mod.Backtester()
    backtester = _bt_mod.backtester

    async def _run():
        global _last_bt
        from ..state import state
        from ..ensemble import ensemble as _ens
        try:
            result = await backtester.run_full_ai(days=days)
            data = {
                "days": days,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result if isinstance(result, dict) else {},
            }
            _last_bt = data
            _save_bt(data)
            # Log wzbogacony (audyt 2026-07-04, na wyrazna prosbe) - nie tylko
            # PnL/PF/WR, ale KTO glosowal z jaka waga i kto ile transakcji
            # zlapal - widoczne w AI Log bez pobierania plikow.
            if isinstance(result, dict):
                r = result
                wagi = ", ".join(f"{m}={w:.3f}" for m, w in sorted((_ens.weights or {}).items(), key=lambda x: -x[1]))
                attr = r.get("model_attribution", {})
                attr_str = " | ".join(
                    f"{m}:{s['count']}tr/PF{s['pf']:.2f}" for m, s in sorted(attr.items(), key=lambda x: -x[1]['count'])
                )
                msg = (f"BACKTEST {days}d | PF={r.get('profit_factor')} WR={r.get('win_rate')}% "
                       f"trades={r.get('total_trades')} PnL={r.get('total_pnl_usdt')} maxDD={r.get('max_drawdown_pct')}% "
                       f"| wagi: {wagi} | kto zlapal: {attr_str}")
                state.add_log("ai", "INFO", event="BACKTEST", message=msg)
                _log_to_file(msg)
        except Exception as e:
            logger.warning("backtest task error: %s", e)
            try:
                from ..state import state as _st
                _st.add_log("ai", "ERROR", event="BACKTEST", message=f"Backtest error: {e}")
                _log_to_file(f"BACKTEST error: {e}")
            except Exception:
                pass

    asyncio.create_task(_run())
    return {"status": "ok", "message": f"Backtest {days}d uruchomiony"}


@router.get("/backtest/last")
async def ctrl_backtest_last(full: bool = False):
    """Domyslnie BEZ trade_log (audyt 2026-07-04) - pelna lista transakcji
    (4000+ rekordow, kilka MB) spowalniala/zawieszala dashboard przy kazdym
    zwyklym odswiezeniu statusu. full=true zwraca kompletny raport z trade_log."""
    data = _last_bt or _load_bt()
    if not data:
        return {"status": "error", "message": "Brak danych — uruchom backtest"}
    if full:
        return {"status": "ok", **data}
    data_light = dict(data)
    result = data_light.get("result")
    if isinstance(result, dict) and "trade_log" in result:
        data_light["result"] = {k: v for k, v in result.items() if k != "trade_log"}
        data_light["result"]["trade_log_count"] = len(result["trade_log"])
    return {"status": "ok", **data_light}


@router.get("/backtest/history")
async def ctrl_backtest_history():
    """Lista zapisanych biegow backtestu (bt_epv_{data}.json), najnowsze pierwsze
    (audyt 2026-07-04) - kazdy /ctrl/backtest zapisuje TRWALY plik obok
    last_backtest.json, wiec historia sie nie gubi przy kolejnych biegach."""
    if not _BT_HIST_DIR.exists():
        return {"status": "ok", "files": []}
    files = sorted(_BT_HIST_DIR.glob("bt_epv_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:50]:
        try:
            import json as _json
            d = _json.loads(f.read_text())
            r = d.get("result", {})
            out.append({
                "file": f.name, "finished_at": d.get("finished_at"),
                "days": d.get("days"), "pf": r.get("profit_factor"),
                "trades": r.get("total_trades"), "wr": r.get("win_rate"),
                "pnl": r.get("total_pnl_usdt"),
            })
        except Exception:
            continue
    return {"status": "ok", "files": out}


@router.get("/backtest/download/{filename}")
async def ctrl_backtest_download(filename: str):
    """Pobierz konkretny plik biegu backtestu (audyt 2026-07-05, na wyrazna
    prosbe - lista sciagalnych plikow w AI Status obok logu)."""
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    if not filename.startswith("bt_epv_") or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    fpath = _BT_HIST_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(fpath), media_type="application/json", filename=filename)


_WFV_FILE = INST_DIR / "data" / "last_wfv.json"
_last_wfv: dict = {}


def _save_wfv(data: dict):
    try:
        import json as _json
        _WFV_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WFV_FILE.write_text(_json.dumps(data, default=str))
    except Exception as e:
        logger.warning("_save_wfv error: %s", e)


def _load_wfv() -> dict:
    try:
        import json as _json
        if _WFV_FILE.exists():
            return _json.loads(_WFV_FILE.read_text())
    except Exception:
        pass
    return {}


@router.post("/backtest/wfv")
async def ctrl_backtest_wfv(
    n_windows:    int = 6,
    window_days:  int = 90,
    embargo_days: int = 7,
):
    import importlib
    _bt_mod = importlib.import_module("core.backtester")
    if not hasattr(_bt_mod, "backtester"):
        _bt_mod.backtester = _bt_mod.Backtester()
    bt = _bt_mod.backtester

    async def _run():
        global _last_wfv
        try:
            result = await bt.run_wfv(
                n_windows=n_windows,
                window_days=window_days,
                embargo_days=embargo_days,
            )
            from datetime import datetime, timezone
            data = {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
            }
            _last_wfv = data
            _save_wfv(data)
        except Exception as e:
            logger.warning("wfv task error: %s", e)

    asyncio.create_task(_run())
    return {"status": "ok", "message": f"WFV {n_windows}×{window_days}d uruchomiony"}


@router.get("/backtest/wfv/last")
async def ctrl_backtest_wfv_last():
    data = _last_wfv or _load_wfv()
    if not data:
        return {"status": "error", "message": "Brak danych — uruchom WFV"}
    return {"status": "ok", **data}


@router.get("/screening/last")
async def ctrl_screening_last():
    inst_name_short = _inst_name().replace("HAI_", "")
    files = sorted(INST_DIR.glob(f"screening_{inst_name_short}_*.txt"), reverse=True)
    if not files:
        # fallback: dowolny plik screeningu
        files = sorted(INST_DIR.glob("screening_*.txt"), reverse=True)
    if not files:
        return {"status": "error", "message": "Brak pliku screeningu"}
    f = files[0]
    try:
        return {"status": "ok", "filename": f.name, "content": f.read_text(encoding="utf-8")}
    except Exception as e:
        return {"status": "error", "message": str(e)[:100]}


# ── Neural ───────────────────────────────────────────────────────────────────

@router.get("/neural/status")
async def neural_status():
    return _neural_status_auto()


@router.post("/neural/toggle")
async def neural_toggle(name: str, enabled: bool):
    ens = _ensemble()
    if hasattr(ens, 'enable_neural') and callable(ens.enable_neural):
        ok = ens.enable_neural(name, enabled)
        if ok:
            return {"status": "ok", "name": name, "enabled": enabled}
        return {"status": "error", "message": f"Nie można załadować: {name}"}
    return {"status": "error", "message": "Neural toggle niedostępny dla tej instancji"}


@router.post("/neural/unload")
async def neural_unload(name: str):
    ens = _ensemble()
    if hasattr(ens, 'unload_neural'):
        ens.unload_neural(name)
    return {"status": "ok", "name": name}


# ── Zoo ──────────────────────────────────────────────────────────────────────

_TREE_NAMES = {"lgb", "xgb", "rf", "cat", "histgb", "et", "vol_lgb"}
_SKIP_NAMES = {"regime_hmm", "_wf_reason", "_wf_best"}
# Warianty horyzontu (audyt 2026-07-04, fix #4) - CALA nazwa to WLASNY,
# osobny aktywny czlon ensemble, nie "backup" swojego rodzica (lgb_fast24h
# to NIE plik zapasowy lgb, to inny, rownolegle glosujacy model).
_KNOWN_VARIANTS = {
    "lgb_fast24h", "cat_sharp6x",
    "lgb_fast6h", "rf_fast6h", "xgb_fast6h", "cat_fast6h", "histgb_fast6h",
    "lgb_wide96h", "rf_wide96h", "xgb_wide96h", "cat_wide96h", "histgb_wide96h",
    "lgb_multi_horizon",
}
# Czytelna etykieta do wyswietlenia w dashboardzie (audyt 2026-07-04, na
# wyrazna prosbe "popodpisuj pliki 24h i x6 zeby widziec ktory model jest
# ktory") - zamiast surowej nazwy pliku.
_VARIANT_LABELS = {
    "lgb_fast24h": "LGB (24h lookahead)",
    "cat_sharp6x": "CAT (TP=6x ATR)",
    "lgb_fast6h": "LGB (6h lookahead)",
    "rf_fast6h": "RF (6h lookahead)",
    "xgb_fast6h": "XGB (6h lookahead)",
    "cat_fast6h": "CAT (6h lookahead)",
    "histgb_fast6h": "HistGB (6h lookahead)",
    "rf_wide96h": "RF (96h lookahead)",
    "xgb_wide96h": "XGB (96h lookahead)",
    "cat_wide96h": "CAT (96h lookahead)",
    "histgb_wide96h": "HistGB (96h lookahead)",
    "lgb_multi_horizon": "LGB (multi-horyzont 4-72h)",
}


def _guess_model(stem: str) -> str:
    """Wyciąga nazwę modelu bazowego ze stem pliku (np. 'lgb_OLD_v7' → 'lgb').
    Znane warianty horyzontu zwracaja CALA nazwe (osobny aktywny model),
    reszta - prefiks bazowego algorytmu (backup/staging)."""
    s = stem.lower()
    if s in _KNOWN_VARIANTS:
        return s
    # staging wariantu (np. "lgb_fast24h_NEW") - rozpoznaj po ucieciu
    # znanego sufiksu (_NEW/_backup) i sprawdzeniu czy reszta to wariant.
    for suffix in ("_new", "_backup"):
        if s.endswith(suffix) and s[:-len(suffix)] in _KNOWN_VARIANTS:
            return s[:-len(suffix)]
    for name in _TREE_NAMES:
        if s == name or s.startswith(name + "_") or s.startswith(name + "-"):
            return name
    # fallback: bierz pierwszą część przed _
    return s.split("_")[0]


def _file_meta(f: Path) -> dict:
    import datetime
    st = f.stat()
    return {
        "file":    f.name,
        "path":    str(f),
        "size_mb": round(st.st_size / 1_048_576, 1),
        "mtime":   datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


@router.get("/zoo")
async def zoo_list():
    inst = _inst_name()
    tree_zoo   = Path("/root/ProjektHAI/model_zoo/store/tree")
    neural_zoo = Path("/root/ProjektHAI/model_zoo/store/neural")

    # ── 1. Aktywne modele instancji (data/models/)
    local, local_backup = [], []
    models_dir = INST_DIR / "data" / "models"
    if models_dir.exists():
        for f in sorted(models_dir.glob("*.pkl")):
            if any(skip in f.stem for skip in _SKIP_NAMES):
                continue
            model = _guess_model(f.stem)
            is_clean = f.stem == model  # "lgb.pkl" — aktywny
            label = _VARIANT_LABELS.get(model, model.upper())
            entry = {**_file_meta(f), "model": model, "label": label, "active": is_clean}
            if is_clean:
                local.append(entry)
            else:
                local_backup.append(entry)

    # ── 2. Backupy w podfolderach instancji (oldies/, models.old_*/)
    for backup_dir in INST_DIR.glob("*"):
        if not backup_dir.is_dir():
            continue
        if backup_dir.name in ("data", "core", "routes", "templates", "__pycache__"):
            continue
        for f in backup_dir.rglob("*.pkl"):
            if any(skip in f.stem for skip in _SKIP_NAMES):
                continue
            model = _guess_model(f.stem)
            label = _VARIANT_LABELS.get(model, model.upper())
            entry = {**_file_meta(f), "model": model, "label": label, "active": False,
                     "subdir": str(f.relative_to(INST_DIR))}
            local_backup.append(entry)

    # ── 3. Model zoo (tree + neural) — tylko pliki tej instancji
    def _scan_zoo(zoo_dir):
        out = []
        if not zoo_dir.exists():
            return out
        for f in sorted(zoo_dir.glob(f"{inst}__*.pkl"), reverse=True):
            parts = f.stem.split("__")
            entry = _file_meta(f)
            if len(parts) == 3:
                entry.update({"instance": parts[0], "model": parts[1], "ts": parts[2]})
            out.append(entry)
        return out

    return {
        "local":        local,
        "local_backup": sorted(local_backup, key=lambda x: x["mtime"], reverse=True),
        "trees":        _scan_zoo(tree_zoo),
        "neurals":      _scan_zoo(neural_zoo),
    }


@router.post("/zoo/load-tree")
async def zoo_load_tree(model: str, path: str):
    p = Path(path)
    if not p.exists() or "/model_zoo/" not in str(p):
        return {"status": "error", "message": "Niedozwolona ścieżka"}
    ens = _ensemble()
    try:
        import joblib as jl
        data = jl.load(p)
        ens.models[model]       = data["model"]
        ens.scalers[model]      = data.get("scaler")
        ens.accuracies[model]   = data.get("accuracy", 0.0)
        if hasattr(ens, 'feature_names'):
            ens.feature_names[model] = data.get("feature_names") or data.get("features") or []
        ens.f1_scores[model]    = data.get("f1") or data.get("f1_long", 0.0)
        ens._recalc_weights()
        ens.active = len(ens.models) >= 2
        ens._cache.clear()
        _disabled_models.discard(model)
        logger.info("CTRL zoo-load-tree: %s ← %s", model, p.name)
        return {"status": "ok", "model": model, "file": p.name,
                "acc": ens.accuracies[model], "weights": ens.weights}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}


_AUDIT_INSTANCES = [
    {"name": "EPV", "port": 5010},
    {"name": "DEV", "port": 5015},
    {"name": "LAB", "port": 5020},
    {"name": "LIV", "port": 5025},
    {"name": "TST", "port": 5030},
]

@router.get("/instances-audit")
async def instances_audit():
    import httpx, base64
    import os
    user = os.getenv("HAI_USER", "")
    pw   = os.getenv("HAI_PASS", "")
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    results = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for inst in _AUDIT_INSTANCES:
            try:
                r = await client.get(f"http://localhost:{inst['port']}/ctrl/status", headers=headers)
                d = r.json()
                results.append({"name": inst["name"], "port": inst["port"],
                                 "ok": True, "models": d.get("models", {})})
            except Exception as e:
                results.append({"name": inst["name"], "port": inst["port"],
                                 "ok": False, "error": str(e)[:80]})
    return {"status": "ok", "instances": results}


# audyt 2026-07-05, na wyrazna prosbe "podlacz nas tez do innych gield i
# zrodel danych... a dane kolektorow do ai status i main" - pelna, prawdziwa
# lista zrodel danych (nie stary waski widget sprawdzajacy tylko
# derivatives_cache.json/ohlcv/stats) - macro (yfinance/CoinGecko/
# Alternative.me) + derivatives per gielda (Binance/Coinalyze).
_WAREHOUSE = Path("/root/ProjektHAI/data_warehouse")


def _file_age_hours(path: Path) -> float:
    import time
    if not path.exists():
        return 999999.0
    return (time.time() - path.stat().st_mtime) / 3600.0


def _parquet_last_ts_age_hours(path: Path) -> float:
    """Wiek NAJNOWSZEGO wiersza w parquet (nie mtime pliku) - lepsze dla
    zrodel gdzie plik moze byc 'swiezy' (skopiowany) ale dane w srodku stare."""
    try:
        import pandas as pd
        from datetime import datetime, timezone
        df = pd.read_parquet(path)
        tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
        last = pd.Timestamp(df[tcol].max())
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        return (datetime.now(timezone.utc) - last.to_pydatetime()).total_seconds() / 3600.0
    except Exception:
        return _file_age_hours(path)


@router.get("/data-sources")
async def data_sources():
    def _entry(name, exchange, data, path, age_fn, ok_hours, warn_hours, count=None):
        age = age_fn(path)
        status = "ok" if age < ok_hours else "warn" if age < warn_hours else "err"
        return {"name": name, "exchange": exchange, "data": data,
                "age_hours": round(age, 1), "status": status, "count": count,
                "exists": path.exists()}

    ohlcv_dir = _WAREHOUSE / "ohlcv" / "binance" / "1h"
    ohlcv_files = list(ohlcv_dir.glob("*.parquet")) if ohlcv_dir.exists() else []
    ohlcv_age = _parquet_last_ts_age_hours(ohlcv_files[0]) if ohlcv_files else 999999.0

    def _cnt(dir_path: Path) -> int:
        return len(list(dir_path.glob("*.parquet"))) if dir_path.exists() else 0

    sources = [
        {"name": "OHLCV (ceny 1h/4h/1d)", "exchange": "Binance", "data": "open/high/low/close/volume",
         "age_hours": round(ohlcv_age, 1), "status": "ok" if ohlcv_age < 26 else "warn" if ohlcv_age < 50 else "err",
         "count": len(ohlcv_files), "exists": len(ohlcv_files) > 0},
        _entry("Funding Rate", "Binance", "koszt utrzymania pozycji (8h)",
               _WAREHOUSE / "derivatives" / "funding_rates" / "BTC.parquet",
               _parquet_last_ts_age_hours, 26, 50, _cnt(_WAREHOUSE / "derivatives" / "funding_rates")),
        _entry("Open Interest", "Coinalyze", "suma otwartych pozycji (dzienne)",
               _WAREHOUSE / "derivatives" / "open_interest" / "BTC.parquet",
               _parquet_last_ts_age_hours, 30, 72, _cnt(_WAREHOUSE / "derivatives" / "open_interest")),
        _entry("Taker Buy Ratio", "Binance", "presja kupno/sprzedaz agresorow (1h)",
               _WAREHOUSE / "derivatives" / "taker_ratio" / "BTC.parquet",
               _parquet_last_ts_age_hours, 26, 50, _cnt(_WAREHOUSE / "derivatives" / "taker_ratio")),
        _entry("Fear & Greed Index", "Alternative.me", "sentyment rynkowy (dzienne)",
               _WAREHOUSE / "macro" / "fear_greed.parquet",
               _parquet_last_ts_age_hours, 30, 72),
        _entry("Gold (futures)", "Yahoo Finance", "cena zlota (dzienne OHLC)",
               _WAREHOUSE / "macro" / "gold.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("WTI Oil", "Yahoo Finance", "cena ropy (dzienne OHLC)",
               _WAREHOUSE / "macro" / "oil_wti.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("S&P 500", "Yahoo Finance", "indeks gieldowy USA (dzienne OHLC)",
               _WAREHOUSE / "macro" / "sp500.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("VIX", "Yahoo Finance", "indeks zmiennosci/strachu (dzienne OHLC)",
               _WAREHOUSE / "macro" / "vix.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("US 10Y Treasury", "Yahoo Finance", "rentownosc obligacji (dzienne OHLC)",
               _WAREHOUSE / "macro" / "us10y_yield.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("DXY (Dollar Index)", "Yahoo Finance", "sila dolara (dzienne)",
               _WAREHOUSE / "macro" / "dxy.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("BTC Dominance", "CoinGecko", "7d% zmiana BTC mcap, proxy dominacji (dzienne)",
               _WAREHOUSE / "macro" / "btc_dominance.parquet", _parquet_last_ts_age_hours, 48, 96),
        _entry("Likwidacje (WS)", "Bitget", "zdarzenia likwidacji pozycji (live)",
               _WAREHOUSE / "derivatives" / "liquidations_live",
               lambda p: 999999.0 if not any(p.glob("*.jsonl")) else _file_age_hours(sorted(p.glob("*.jsonl"))[-1]),
               6, 24),
    ]
    ok = sum(1 for s in sources if s["status"] == "ok")
    return {"status": "ok", "sources": sources, "summary": f"{ok}/{len(sources)}"}

# ===========================================
# HAI_EPV Engine ver.10 Final — routes/ai.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: /thoughts (mysli ensemble per symbol), /ensemble/status+reload,
# /models/available+load-slots (test kombinacji bez trwalej promocji),
# /features/correlation-check (live-check przed treningiem niestandardowym),
# /train (pelny, wybor modeli w chain) + /train/custom (algorytm x horyzont
# x cechy x class_weight x val_ratio) + /train/status + /train/horizons,
# /system-config (pelny raport BT: modele/cechy/horyzonty/ATR),
# /promote (_NEW.pkl -> aktywne) + /rollback (_OLD.pkl -> przywroc).
# ===========================================
import asyncio
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter

from ..ensemble import ensemble
from ..features import build_features_live, FEATURE_NAMES
from ..state import state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI"])

# Markery etapow treningu (parsowane z logow ml_trainera).
# Kolejnosc wazna: build dataset -> RF -> LGB -> XGB -> zapis.
# Dopasowanie po substringu w tresci logu.
_TRAIN_STAGE_MARKERS = [
    ("Building features for",       "dataset"),
    ("Dataset built in",            "dataset_done"),
    ("Walk-forward evaluation",     "wfv"),
    ("Training RandomForest",       "rf"),
    ("Training LightGBM",           "lgb"),
    ("Training XGBoost",            "xgb"),
    ("Training CatBoost",           "cat"),
    ("Training HistGradientBoosting", "histgb"),
    ("Training LONG SPECIALIST",    "long_spec"),
    ("KONIEC",                      "done"),
    ("Saved:",                      "save"),
]
_STAGE_LABELS = {
    "dataset": "Budowanie datasetu", "dataset_done": "Dataset gotowy",
    "wfv": "Walk-Forward (wybor modelu)", "rf": "Trening: RandomForest",
    "lgb": "Trening: LightGBM", "xgb": "Trening: XGBoost",
    "cat": "Trening: CatBoost", "histgb": "Trening: HistGradientBoost",
    "long_spec": "Trening: Long Specialist", "save": "Zapisywanie modeli",
    "done": "Gotowe",
}


_WFV_MARKERS = ("Walk-forward", "WFV ", " WF:", "WALK-FORWARD", "walk-forward", "zwyciezca")


def _categorize_trainer_msg(msg: str) -> str:
    """ml_trainer loguje zarowno WFV (ocena kandydatow) jak i faktyczny
    trening finalnych modeli w tym samym loggerze - rozroznij po tresci
    zeby AI Log mial osobna kategorie WFV vs TRAINING (audyt 2026-07-04)."""
    if any(marker in msg for marker in _WFV_MARKERS):
        return "WFV"
    return "TRAINING"


class _StateLogHandler(logging.Handler):
    """Most logger -> state: kazdy rekord ml_trainera trafia do state.add_log('ai', ...)
    oraz aktualizuje _train_state['stage'] po dopasowaniu markera etapu.
    event = kategoria tresci (TRAINING/WFV), nie poziom logu - poziom
    (INFO/WARNING/ERROR) trafia teraz do prefiksu message zamiast nadpisywac
    kategorie (audyt 2026-07-04, wczesniej event=lvl gubil kategorie)."""

    def emit(self, record):
        try:
            msg = record.getMessage()
            lvl = record.levelname  # INFO / WARNING / ERROR
            category = _categorize_trainer_msg(msg)
            try:
                prefixed = msg if lvl == "INFO" else f"[{lvl}] {msg}"
                state.add_log("ai", lvl, event=category, model_name="ml_trainer",
                              message=prefixed[:480])
            except Exception:
                pass
            for marker, stage in _TRAIN_STAGE_MARKERS:
                if marker in msg:
                    _train_state["stage"] = stage
                    break
        except Exception:
            pass

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# Stan ostatniego treningu (in-memory)
_train_state: Dict = {
    "status": "idle",          # idle | running | done | error
    "stage": None,             # dataset | dataset_done | rf | lgb | xgb | save | done
    "started_at": None,
    "finished_at": None,
    "summary": None,
    "error": None,
}


def _build_features_from_engine(engine, symbol: str) -> Optional[Dict]:
    """Buduje 16 features dla symbolu na podstawie danych z engine."""
    from ..strategies.registry import get_strategy
    from .strategies import ai_strategy  # rejestracja

    prices_1h = [c["close"] for c in engine._price_history_1h.get(symbol, [])]
    prices_4h = [c["close"] for c in engine._price_history_4h.get(symbol, [])]
    prices_1d = [c["close"] for c in engine._price_history_1d.get(symbol, [])]
    volumes_1h = [c["volume"] for c in engine._price_history_1h.get(symbol, [])]

    if len(prices_1h) < 50:
        return None

    # Singleton strategy z engine jesli dostepny, fallback do nowej instancji
    strategy = engine._strategy
    if strategy is None:
        strategy = get_strategy('ai_strategy')

    return build_features_live(
        strategy=strategy,
        prices_1h=prices_1h,
        prices_4h=prices_4h,
        prices_1d=prices_1d,
        volumes_1h=volumes_1h,
        funding_rate=0.0,
        funding_change_24h=0.0,
    )


# ─────────────────────────────────────────────────────────────
# THOUGHTS
# ─────────────────────────────────────────────────────────────

@router.get("/thoughts/{symbol:path}")
async def get_model_thoughts(symbol: str):
    """Podglad mysli wszystkich modeli AI dla symbolu.
    Przyklad: GET /ai/thoughts/BTC/USDT:USDT"""
    if not ensemble.active:
        ensemble.load_models()
        if not ensemble.active:
            return {"error": "Ensemble nie zaladowany - brak modeli .pkl"}

    try:
        from ..engine import engine

        prices_1h = engine._price_history_1h.get(symbol, [])
        if len(prices_1h) < 50:
            return {
                "error": f"Za malo danych dla {symbol} "
                         f"({len(prices_1h)} swiec, minimum 50)"
            }

        features = _build_features_from_engine(engine, symbol)
        if not features:
            return {"error": "Nie udalo sie zbudowac features"}

        ensemble_decision = ensemble.predict(features)

        # FIX v6.0: ModelThought ma 'reason' i 'weight', NIE ma 'key_features'
        return {
            "symbol": symbol,
            "current_price": prices_1h[-1]["close"] if prices_1h else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ensemble_decision": ensemble_decision,
            "features_used": features,
            "individual_thoughts": ensemble_decision.get("thoughts", []),
            "active_models": list(ensemble.models.keys()),
            "feature_names": FEATURE_NAMES,
        }
    except Exception as e:
        logger.error(f"Thoughts endpoint error {symbol}: {e}", exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# ENSEMBLE STATUS / RELOAD
# ─────────────────────────────────────────────────────────────

@router.get("/ensemble/status")
async def ensemble_status():
    """Status ensemble - ktore modele zaladowane, wagi, cache."""
    if not ensemble.active:
        ensemble.load_models()
    return ensemble.status()


@router.post("/ensemble/reload")
async def ensemble_reload():
    """Przeladuj modele ensemble (np. po treningu)."""
    ensemble.load_models()
    return {
        "status": "ok",
        "models": list(ensemble.models.keys()),
        "active": ensemble.active,
        "weights": ensemble.weights,
        "accuracies": {k: round(v, 4) for k, v in ensemble.accuracies.items()},
    }


@router.get("/models/available")
async def models_available():
    """Lista WSZYSTKICH .pkl w data/models/ (aktywne + staging _NEW/_backup +
    archiwalne warianty), z metadanymi - do panelu 'Zaladuj do slotow' w AI
    Learning (audyt 2026-07-05, na wyrazna prosbe). Pozwala recznie zaladowac
    NAWET odrzucone warianty jako dodatkowe glosy do testu (np. rf_sharp6x
    jako 8. glos obok reszty, mimo ze solo/WFV wypadl slabo)."""
    import joblib
    from ..ensemble import MODELS_DIR, MAX_ENSEMBLE_MODELS
    currently_loaded = set(ensemble.models.keys())
    items = []
    for path in sorted(MODELS_DIR.glob("*.pkl")):
        name = path.stem
        if name == "regime_hmm":
            continue
        try:
            data = joblib.load(path)
            feats = data.get("features") or data.get("feature_names") or []
            acc = data.get("accuracy", 0.0)
            prec_long = data.get("precision_long")
        except Exception:
            feats, acc, prec_long = [], 0.0, None
        items.append({
            "name": name, "loaded": name in currently_loaded,
            "n_features": len(feats), "accuracy": round(acc, 4) if acc else 0.0,
            "precision_long": round(prec_long, 4) if prec_long is not None else None,
            "is_staging": name.endswith("_NEW") or name.endswith("_backup") or "backup_" in name,
        })
    return {"status": "ok", "models": items, "max_slots": MAX_ENSEMBLE_MODELS,
            "currently_loaded": sorted(currently_loaded)}


@router.post("/models/load-slots")
async def models_load_slots(names: str):
    """Zaladuj RECZNIE WYBRANY zestaw modeli do slotow ensemble (nadpisuje
    dynamiczne skanowanie z load_models()). `names` = lista oddzielona
    przecinkami, np. 'lgb,rf,cat_sharp6x,rf_sharp6x_NEW'. Max 10 (MAX_ENSEMBLE_MODELS)."""
    from ..ensemble import MAX_ENSEMBLE_MODELS
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if not name_list:
        return {"status": "error", "message": "Brak wybranych modeli"}
    if len(name_list) > MAX_ENSEMBLE_MODELS:
        return {"status": "error", "message": f"Za duzo modeli ({len(name_list)}), limit {MAX_ENSEMBLE_MODELS}"}

    ensemble._cache.clear()
    ensemble.models = {}
    ensemble.scalers = {}
    ensemble.accuracies = {}
    ensemble.f1_scores = {}
    ensemble.precision_scores = {}
    ensemble.feature_names = {}
    failed = []
    for name in name_list:
        if not ensemble._load_one(name):
            failed.append(name)
    ensemble.active = len(ensemble.models) > 0
    if ensemble.active:
        ensemble._recalc_weights()
    state.add_log("ai", "INFO", event="MODEL_LOAD", model_name="ensemble",
                  message=f"Reczny load slotow: {list(ensemble.models.keys())} | nieudane: {failed}")
    return {"status": "ok" if not failed else "partial", "loaded": list(ensemble.models.keys()),
            "failed": failed, "weights": ensemble.weights}


# ─────────────────────────────────────────────────────────────
# TRAINING - ml_trainer w background
# ─────────────────────────────────────────────────────────────

def _train_in_thread():
    """Synchronous trening w osobnym threadzie (ml_trainer jest sync, ciezki CPU)."""
    global _train_state
    handler = _StateLogHandler()
    handler.setLevel(logging.INFO)
    trainer_logger = logging.getLogger("core.ml_trainer")
    prev_level = trainer_logger.level
    trainer_logger.addHandler(handler)
    trainer_logger.setLevel(logging.INFO)
    try:
        _train_state["status"] = "running"
        _train_state["stage"] = "dataset"
        _train_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["finished_at"] = None
        _train_state["summary"] = None
        _train_state["error"] = None
        state.add_log("ai", "INFO", event="TRAINING", model_name="ml_trainer",
                      message="Trening wystartowal (background thread)")

        from ..ml_trainer import train_and_save
        summary = train_and_save(suffix="_NEW")

        _train_state["status"] = "done"
        _train_state["stage"] = "done"
        _train_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["summary"] = summary
        logger.info(f"Trening zakonczony: {summary}")
        state.add_log("ai", "INFO", event="TRAINING", model_name="ml_trainer",
                      message=f"Trening zakonczony: {summary}")
    except Exception as e:
        _train_state["status"] = "error"
        _train_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["error"] = str(e)
        logger.error(f"Trening blad: {e}", exc_info=True)
        try:
            state.add_log("ai", "ERROR", event="TRAINING", model_name="ml_trainer",
                          message=f"[ERROR] Trening blad: {e}")
        except Exception:
            pass
    finally:
        trainer_logger.removeHandler(handler)
        trainer_logger.setLevel(prev_level)


def _train_in_thread_custom(only: list, custom_features: list = None, label_scheme: str = "3class",
                             class_weights: dict = None, val_ratio: float = None):
    """Wariant _train_in_thread dla treningu niestandardowego (audyt 2026-07-05,
    panel 'Trening niestandardowy' w AI Learning) - te sam _train_state (widget
    Postep dziala bez zmian), ale train_and_save(only=...) zamiast pelnych 5.
    `custom_features` (opcjonalnie) - jesli podane, TYMCZASOWO nadpisuje
    MODEL_FEATURES[name] dla kazdej nazwy w `only` na ta liste (zamiast
    domyslnej specjalizacji), przywracane w finally - dziala bez zmian w
    train_models() bo ta funkcja i tak zawsze czyta globalny MODEL_FEATURES.
    `label_scheme` - '3class' (domyslny, NEUTRAL/LONG/SHORT) albo 'binary'
    (stary schemat jak DEV/LAB przed kalka EPV - empirycznie GORSZY realny PF
    mimo ladniejszej accuracy walidacyjnej, dostepny do eksperymentu).
    `class_weights` (opcjonalnie) - dict {0,1,2} nadpisujacy domyslne wagi klas.
    `val_ratio` (opcjonalnie) - nadpisuje domyslny split 80/20 (core.ml_trainer.VAL_RATIO)."""
    global _train_state
    handler = _StateLogHandler()
    handler.setLevel(logging.INFO)
    trainer_logger = logging.getLogger("core.ml_trainer")
    prev_level = trainer_logger.level
    trainer_logger.addHandler(handler)
    trainer_logger.setLevel(logging.INFO)
    from ..ml_trainer import ml_trainer as _mt
    from ..ml_trainer import MODEL_FEATURES, MODEL_LABEL_COLUMN
    _saved_features = {}
    _saved_labels = {}
    _saved_val_ratio = _mt.VAL_RATIO
    try:
        if custom_features:
            for name in only:
                _saved_features[name] = MODEL_FEATURES.get(name)
                MODEL_FEATURES[name] = custom_features
        if label_scheme == "binary":
            for name in only:
                _saved_labels[name] = MODEL_LABEL_COLUMN.get(name)
                MODEL_LABEL_COLUMN[name] = "label_binary"
        if val_ratio is not None:
            _mt.VAL_RATIO = val_ratio

        _train_state["status"] = "running"
        _train_state["stage"] = "dataset"
        _train_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["finished_at"] = None
        _train_state["summary"] = None
        _train_state["error"] = None
        state.add_log("ai", "INFO", event="TRAINING", model_name="ml_trainer",
                      message=f"Trening niestandardowy wystartowal: {only}"
                              + (f" | cechy: {custom_features}" if custom_features else "")
                              + (f" | label: {label_scheme}" if label_scheme != "3class" else ""))

        from ..ml_trainer import train_and_save
        summary = train_and_save(suffix="_NEW", only=only, class_weights=class_weights)

        _train_state["status"] = "done"
        _train_state["stage"] = "done"
        _train_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["summary"] = summary
        logger.info(f"Trening niestandardowy zakonczony: {summary}")
        state.add_log("ai", "INFO", event="TRAINING", model_name="ml_trainer",
                      message=f"Trening niestandardowy zakonczony: {only}")
    except Exception as e:
        _train_state["status"] = "error"
        _train_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _train_state["error"] = str(e)
        logger.error(f"Trening niestandardowy blad: {e}", exc_info=True)
        try:
            state.add_log("ai", "ERROR", event="TRAINING", model_name="ml_trainer",
                          message=f"[ERROR] Trening niestandardowy blad: {e}")
        except Exception:
            pass
    finally:
        trainer_logger.removeHandler(handler)
        for name, feats in _saved_features.items():
            if feats is not None:
                MODEL_FEATURES[name] = feats
            else:
                MODEL_FEATURES.pop(name, None)
        for name, lbl in _saved_labels.items():
            if lbl is not None:
                MODEL_LABEL_COLUMN[name] = lbl
            else:
                MODEL_LABEL_COLUMN.pop(name, None)
        _mt.VAL_RATIO = _saved_val_ratio


def _swap_in_new_variant(base_name: str) -> bool:
    """Podmienia POJEDYNCZY slot ensemble na jego swiezo wytrenowany
    _NEW.pkl (staging, NIE promowany do produkcji) - zeby auto-backtest/WFV
    po treningu testowaly NOWE wagi w KONTEKSCIE calego istniejacego
    ensemble, a nie tylko wytrenowany model solo. Nie dodaje nowego slotu -
    podmienia w miejscu pod tym samym kluczem `base_name`."""
    from ..ensemble import MODELS_DIR
    import joblib
    path = MODELS_DIR / f"{base_name}_NEW.pkl"
    if not path.exists():
        return False
    try:
        data = joblib.load(path)
        ensemble.models[base_name] = data["model"]
        ensemble.scalers[base_name] = data.get("scaler")
        ensemble.accuracies[base_name] = data.get("accuracy", 0.0)
        ensemble.f1_scores[base_name] = data.get("f1") or data.get("f1_long", 0.0)
        ensemble.precision_scores[base_name] = data.get("precision") or data.get("wf_precision") or 0.0
        ensemble.feature_names[base_name] = data.get("features") or data.get("feature_names", [])
        return True
    except Exception as e:
        logger.warning(f"Nie podmieniono {base_name} na _NEW: {e}")
        return False


async def _train_pipeline(train_fn, train_args: tuple, trained_names: list,
                           then_backtest: bool, then_wfv: bool):
    """Odpal trening (sync, w threadpool) i - jesli zaznaczono ptaszki -
    doczekaj zakonczenia, podmien PRODUKCYJNY ensemble na swiezo wytrenowane
    warianty (staging _NEW, bez trwalej promocji) i dolacz backtest(365d)/WFV
    (audyt 2026-07-05, na wyrazna prosbe 'polacz WFV z trenuj modele, ptaszek
    po treningu odpal backtest, ptaszek odpal WFV'). Fire-and-forget z
    perspektywy HTTP (wolane przez asyncio.create_task)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, train_fn, *train_args)
    if _train_state.get("status") != "done" or not (then_backtest or then_wfv):
        return
    if not ensemble.active:
        ensemble.load_models()
    swapped = [n for n in trained_names if _swap_in_new_variant(n)]
    if swapped:
        ensemble._cache.clear()
        ensemble._recalc_weights()
        state.add_log("ai", "INFO", event="MODEL_LOAD", model_name="ensemble",
                      message=f"Auto-test po treningu: podmieniono staging _NEW dla {swapped}")
    from ..backtester import Backtester
    if then_backtest:
        try:
            bt = Backtester()
            await bt.run_full_ai(days=365)
        except Exception as e:
            logger.error(f"Auto-backtest po treningu blad: {e}", exc_info=True)
    if then_wfv:
        try:
            bt = Backtester()
            await bt.run_wfv(n_windows=6, window_days=90, embargo_days=7, mode="neutral")
        except Exception as e:
            logger.error(f"Auto-WFV po treningu blad: {e}", exc_info=True)


@router.get("/features/correlation-check")
async def features_correlation_check(features: str):
    """Sprawdza czy WYBRANE cechy (panel 'Trening niestandardowy') zawieraja
    silnie skorelowane pary (audyt 2026-07-05, na wyrazna prosbe - live-check
    zeby nie powtorzyc bledu z raportu: ema_slow_r+ema_fast_r razem w XGB,
    korelacja 0.917, lamalo wlasna zasade raportu 'korelacja<0.7')."""
    import json as _json
    from pathlib import Path
    corr_path = Path(__file__).resolve().parent.parent / "data" / "feature_correlation.json"
    if not corr_path.exists():
        return {"status": "error", "message": "Brak pliku korelacji - policz najpierw"}
    data = _json.loads(corr_path.read_text())
    selected = set(f.strip() for f in features.split(",") if f.strip())
    warnings = [p for p in data["pairs"] if p["a"] in selected and p["b"] in selected and abs(p["corr"]) >= 0.7]
    return {"status": "ok", "warnings": warnings}


@router.get("/train/horizons")
async def ai_train_horizons():
    """Lista wyborow horyzontu + WSZYSTKICH dostepnych cech (unia z 7
    produkcyjnych modeli) do panelu 'Trening niestandardowy' (audyt
    2026-07-05) - pozwala reczny wybor cech zamiast tylko domyslnej
    specjalizacji per algorytm."""
    from ..ml_trainer import HORIZON_CHOICES, MODEL_FEATURES
    all_feats = sorted(set().union(*[
        MODEL_FEATURES[m] for m in ("lgb", "rf", "xgb", "cat", "histgb")
        if m in MODEL_FEATURES
    ]))
    return {"status": "ok", "horizons": HORIZON_CHOICES,
            "algos": ["lgb", "rf", "xgb", "cat", "histgb", "et", "gb", "ada"],
            "all_features": all_feats}


@router.post("/train/custom")
async def ai_train_custom(algos: str, horizon: str = "main", features: str = "",
                          label_scheme: str = "3class",
                          class_weight_long: float = 2.5, class_weight_short: float = 2.5,
                          val_ratio: float = 0.20,
                          then_backtest: bool = False, then_wfv: bool = False):
    """Trening niestandardowy: wybrane algorytmy x wybrany horyzont x (opcjonalnie)
    reczny wybor cech (audyt 2026-07-05, panel w AI Learning). `algos` = lista
    oddzielona przecinkami np. 'lgb,rf,cat'. `horizon` = klucz z HORIZON_CHOICES
    (main/fast24h/sharp6x/fast6h/h72/wide96h). `features` = opcjonalna lista cech
    oddzielona przecinkami - jesli PUSTA, uzywa domyslnej specjalizacji per
    algorytm (bez zmian). `label_scheme` = '3class' (domyslny) albo 'binary'
    (stary schemat LONG/nie-LONG, empirycznie gorszy realny PF - dostepny do
    eksperymentu). `then_backtest`/`then_wfv` - ptaszki 'po treningu odpal
    backtest(365d)/WFV' (testuje staging _NEW w kontekscie calego ensemble,
    bez trwalej promocji)."""
    from ..ml_trainer import HORIZON_CHOICES
    if _train_state.get("status") == "running":
        return {"status": "already_running", "message": "Trening juz trwa",
                "started_at": _train_state.get("started_at")}
    if horizon not in HORIZON_CHOICES:
        return {"status": "error", "message": f"Nieznany horyzont: {horizon}"}
    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    valid_algos = {"lgb", "rf", "xgb", "cat", "histgb", "et", "gb", "ada"}
    bad = [a for a in algo_list if a not in valid_algos]
    if bad:
        return {"status": "error", "message": f"Nieznane algorytmy: {bad}"}
    if not algo_list:
        return {"status": "error", "message": "Brak wybranych algorytmow"}
    suffix = HORIZON_CHOICES[horizon]["suffix"]
    only = [f"{a}{suffix}" for a in algo_list]
    custom_features = [f.strip() for f in features.split(",") if f.strip()] or None
    if label_scheme not in ("3class", "binary"):
        return {"status": "error", "message": f"Nieznany label_scheme: {label_scheme}"}
    if not (0.05 <= val_ratio <= 0.40):
        return {"status": "error", "message": "val_ratio poza rozsadnym zakresem 0.05-0.40"}
    if not (0.5 <= class_weight_long <= 10) or not (0.5 <= class_weight_short <= 10):
        return {"status": "error", "message": "class_weight poza rozsadnym zakresem 0.5-10"}
    class_weights = {0: 1.0, 1: class_weight_long, 2: class_weight_short}

    asyncio.create_task(_train_pipeline(
        _train_in_thread_custom, (only, custom_features, label_scheme, class_weights, val_ratio), only,
        then_backtest, then_wfv))
    return {"status": "started", "message": f"Trening niestandardowy wystartowal: {only}",
            "models": only, "features": custom_features, "label_scheme": label_scheme,
            "class_weights": class_weights, "val_ratio": val_ratio,
            "check_status_endpoint": "GET /ai/train/status"}


@router.post("/train")
async def ai_train(then_backtest: bool = False, then_wfv: bool = False, models: str = None):
    """Start treningu ml_trainer (warehouse, 16 features, Triple Barrier).
    Trening leci w background thread. Sprawdz status: GET /ai/train/status.
    Po treningu wywolaj POST /ai/promote zeby aktywowac nowe modele NA STALE.
    `then_backtest`/`then_wfv` - ptaszki 'po treningu odpal backtest(365d)/WFV'
    (audyt 2026-07-05) - testuje staging _NEW w kontekscie calego ensemble
    (podmienia sloty tymczasowo), BEZ trwalej promocji - decyzja o promote
    nadal nalezy do uzytkownika po zobaczeniu wyniku.
    `models` (opcjonalnie, CSV) - audyt 2026-07-05 na wyrazna prosbe 'wybor
    modelu/modeli, cala chain na domyslnych cechach, wszystkich ile mamy' -
    zamiast sztywnych 5 bazowych, dowolny podzbior calej biblioteki
    (lgb/rf/xgb/cat/histgb/et/gb/ada) trenowany KAZDY na swojej wlasnej
    domyslnej specjalizacji cech (custom_features=None w _train_in_thread_custom)."""
    if _train_state.get("status") == "running":
        return {
            "status": "already_running",
            "message": "Trening juz trwa",
            "started_at": _train_state.get("started_at"),
        }

    if models:
        selected = [m.strip() for m in models.split(",") if m.strip()]
        asyncio.create_task(_train_pipeline(_train_in_thread_custom, (selected, None),
                                             selected, then_backtest, then_wfv))
    else:
        _BASE_5 = ["lgb", "rf", "xgb", "cat", "histgb"]
        asyncio.create_task(_train_pipeline(_train_in_thread, (), _BASE_5,
                                             then_backtest, then_wfv))

    return {
        "status": "started",
        "message": "Trening ml_trainer wystartowal w background",
        "started_at": _train_state.get("started_at"),
        "check_status_endpoint": "GET /ai/train/status",
        "promote_endpoint": "POST /ai/promote",
    }


@router.get("/train/status")
async def ai_train_status():
    """Status ostatniego treningu (idle/running/done/error)."""
    out = dict(_train_state)
    out["stage_label"] = _STAGE_LABELS.get(out.get("stage"), out.get("stage") or "—")
    return out


# ─────────────────────────────────────────────────────────────
# PROMOTE / ROLLBACK - atomic swap _NEW.pkl -> aktywne
# ─────────────────────────────────────────────────────────────

@router.post("/promote")
async def promote_models():
    """Atomic swap modeli _NEW.pkl -> aktywne. Backup poprzednich jako _OLD.pkl.
    Wywoluje sie PO treningu (GET /ai/train/status == 'done')."""
    promoted = []
    backed_up = []
    missing = []

    for name in ["rf", "lgb", "xgb"]:
        new_p = MODELS_DIR / f"{name}_NEW.pkl"
        cur_p = MODELS_DIR / f"{name}.pkl"
        if not new_p.exists():
            missing.append(name)
            continue
        if cur_p.exists():
            bak = MODELS_DIR / f"{name}_OLD.pkl"
            shutil.copy2(cur_p, bak)
            backed_up.append(name)
        shutil.move(str(new_p), str(cur_p))
        promoted.append(name)

    ensemble.load_models()

    return {
        "status": "ok" if promoted else "no_new_models",
        "promoted": promoted,
        "backed_up": backed_up,
        "missing": missing,
        "ensemble_active": ensemble.active,
        "models": list(ensemble.models.keys()),
        "weights": ensemble.weights,
        "accuracies": {k: round(v, 4) for k, v in ensemble.accuracies.items()},
    }


@router.post("/rollback")
async def rollback_models():
    """Cofnij promote - przywroc _OLD.pkl jako aktywne."""
    rolled = []
    missing = []

    for name in ["rf", "lgb", "xgb"]:
        old_p = MODELS_DIR / f"{name}_OLD.pkl"
        cur_p = MODELS_DIR / f"{name}.pkl"
        if not old_p.exists():
            missing.append(name)
            continue
        shutil.move(str(old_p), str(cur_p))
        rolled.append(name)

    ensemble.load_models()
    return {
        "status": "ok" if rolled else "no_backup",
        "rolled_back": rolled,
        "missing": missing,
        "ensemble_active": ensemble.active,
        "models": list(ensemble.models.keys()),
    }


# ?????????????????????????????????????????????????????????????
# FEATURE IMPORTANCE (przeniesione z trading.py)
# ?????????????????????????????????????????????????????????????

_LABEL_DESC = {
    "label_long": "48h, TP=4×ATR / SL=1×ATR (główny)",
    "label_fast24h": "24h, TP=4×ATR / SL=1×ATR",
    "label_sharp6x": "48h, TP=6×ATR / SL=1×ATR",
    "label_fast6h": "6h, TP=4×ATR / SL=1×ATR",
    "label_wide96h": "96h, TP=4×ATR / SL=1×ATR",
    "label_h72": "72h, TP=4×ATR / SL=1×ATR",
    "label_binary": "48h, TP=4×ATR (binarny LONG/nie-LONG)",
}


@router.get("/system-config")
async def system_config():
    """Pelna konfiguracja systemu na jakiej dziala aktualny ensemble (audyt
    2026-07-05, na wyrazna prosbe - 'pelny raport systemu na jakim byl
    robiony bt') - modele/cechy/horyzonty/ATR, do raportu w AI Learning."""
    from ..ml_trainer import MODEL_LABEL_COLUMN, TP_ATR_MULT, SL_ATR_MULT, LOOKAHEAD_BARS
    if not ensemble.active:
        ensemble.load_models()
    models = []
    for name in ensemble.models:
        label_col = MODEL_LABEL_COLUMN.get(name, "label_long")
        models.append({
            "name": name,
            "n_features": len(ensemble.feature_names.get(name, [])),
            "features": ensemble.feature_names.get(name, []),
            "label_column": label_col,
            "horizon_desc": _LABEL_DESC.get(label_col, label_col),
            "weight": round(ensemble.weights.get(name, 0.0), 4),
            "accuracy": round(ensemble.accuracies.get(name, 0.0), 4),
        })
    return {
        "status": "ok",
        "model_count": len(models),
        "max_slots": 10,
        "default_tp_atr_mult": TP_ATR_MULT,
        "default_sl_atr_mult": SL_ATR_MULT,
        "default_lookahead_bars": LOOKAHEAD_BARS,
        "models": models,
    }


@router.get("/features")
async def feature_importance():
    """Feature importance z 3 modeli ensemble.
    Pokazuje ktore z 16 wskaznikow maja najwiekszy wplyw na decyzje AI."""
    if not ensemble.active:
        ensemble.load_models()
    if not ensemble.active:
        return {"error": "Ensemble nie zaladowany - najpierw POST /ai/train + /ai/promote"}

    result = {}
    for name, model in ensemble.models.items():
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                result[name] = {"error": "model nie ma feature_importances_"}
                continue
            paired = sorted(
                zip(FEATURE_NAMES, importances.tolist()),
                key=lambda x: x[1], reverse=True
            )
            result[name] = [
                {"feature": f, "importance": round(float(i), 4)}
                for f, i in paired
            ]
        except Exception as e:
            result[name] = {"error": str(e)}

    return {
        "models": list(result.keys()),
        "weights": ensemble.weights,
        "accuracies": {k: round(v, 4) for k, v in ensemble.accuracies.items()},
        "importance": result,
        "feature_names": FEATURE_NAMES,
    }


@router.get("/walkforward/status")
async def walkforward_status():
    """Info o walk-forward.
    Uwaga: ml_trainer.train_models() ma walk-forward WBUDOWANY
    jako chronological train/val split (val_ratio=0.20, NO shuffle).
    Stary endpoint /ai/walkforward dziala teraz przez POST /ai/train."""
    return {
        "info": "Walk-Forward jest wbudowany w ml_trainer.train_models() "
                "(chronological train/val split, no shuffle).",
        "val_ratio": 0.20,
        "use_endpoint": "POST /ai/train  + GET /ai/train/status",
        "last_training": _train_state,
    }

@router.get("/status")
async def ai_status_facade():
    """Fasada agregujaca ensemble.status() + _train_state.
    Format zgodny ze starym dashboardem (hai_v4.html)."""
    if not ensemble.active:
        ensemble.load_models()

    accs = ensemble.accuracies or {}
    models_dict = {k: round(v, 4) for k, v in accs.items()}

    if accs:
        best_name = max(accs.items(), key=lambda x: x[1])[0]
        best_acc = accs[best_name]
    else:
        best_name = "none"
        best_acc = 0.0

    summary = _train_state.get("summary") or {}
    samples = 0
    if summary:
        models_summary = summary.get("models", {})
        if models_summary:
            first_model = next(iter(models_summary.values()), {})
            samples = first_model.get("samples", 0)

    last_train = _train_state.get("finished_at")
    training_in_progress = _train_state.get("status") == "running"

    return {
        "accuracy": round(best_acc, 4),
        "best_model": best_name,
        "samples": samples,
        "last_train": last_train,
        "training_in_progress": training_in_progress,
        "models": models_dict,
        "weights": ensemble.weights,
        "active_count": len(ensemble.models),
        "train_status": _train_state.get("status", "idle"),
    }



@router.get("/predict/{symbol:path}")
async def predict_from_warehouse(symbol: str):
    """Predykcja AI z magazynu (guzik Predict Live). Single source: build_features_live."""
    if not ensemble.active:
        ensemble.load_models()
        if not ensemble.active:
            return {"error": "Ensemble nie zaladowany"}
    try:
        from ..backtester import Backtester
        from ..features import build_features_live
        from ..strategies.ai_strategy import AIStrategy
        from ..engine import engine
        bt = Backtester()
        c1h = bt.load_candles_from_warehouse(symbol, "1h")
        c4h = bt.load_candles_from_warehouse(symbol, "4h")
        c1d = bt.load_candles_from_warehouse(symbol, "1d")
        if not c1h or len(c1h) < 50:
            return {"error": f"Za malo danych ({len(c1h) if c1h else 0} swiec, min 50)"}
        prices_1h = [c["close"] for c in c1h]
        prices_4h = [c["close"] for c in c4h] if c4h else prices_1h
        prices_1d = [c["close"] for c in c1d] if c1d else prices_1h
        volumes_1h = [c.get("volume", 0) for c in c1h]
        last = c1h[-1]
        dk = {
            "funding_rate": last.get("funding_rate", 0.0) or 0.0,
            "funding_change_24h": last.get("funding_change_24h", 0.0) or 0.0,
            "oi_total_log": last.get("oi_total_log", 0.0) or 0.0,
            "oi_change_24h": last.get("oi_change_24h", 0.0) or 0.0,
            "oi_zscore_30d": last.get("oi_zscore_30d", 0.0) or 0.0,
        }
        strategy = engine._strategy or AIStrategy()
        features = build_features_live(
            strategy=strategy, prices_1h=prices_1h, prices_4h=prices_4h,
            prices_1d=prices_1d, volumes_1h=volumes_1h, **dk,
        )
        if not features:
            return {"error": "Nie udalo sie zbudowac features"}
        decision = ensemble.predict(features)
        return {"symbol": symbol.replace("/USDT:USDT","").replace("/USDT",""),
                "source": "warehouse", "current_price": prices_1h[-1],
                "ensemble_decision": decision}
    except Exception as e:
        return {"error": str(e)[:200]}



@router.post("/config/confidence")
async def set_confidence(value: float):
    """Ustawia min_confidence na aktywnej strategii (0.40–0.95)."""
    if not (0.40 <= value <= 0.95):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Wartość poza zakresem 0.40–0.95")
    from ..engine import engine
    from ..state import state
    if engine._strategy and hasattr(engine._strategy, "min_confidence"):
        engine._strategy.min_confidence = value
    state.add_log("ai", "INFO", event="CONFIG",
                  message=f"ai_confidence_min → {value:.2f}")
    return {"status": "ok", "confidence_min": value}


@router.get("/market_bias")
async def market_bias():
    """Agregat sentymentu AI (LONG/SHORT/NEUTRAL ze wszystkich coinow w magazynie).
    Czyta gotowy market_bias_{INSTANCE}.json (liczony przez cron co ~20min)."""
    import json, os, glob
    try:
        files = glob.glob("market_bias_*.json")
        if not files:
            return {"error": "Brak danych - market_bias jeszcze nie policzony"}
        path = max(files, key=os.path.getmtime)
        data = json.load(open(path))
        # zwracamy details (front potrzebuje per-coin: action/consensus/models)
        return data
    except Exception as e:
        return {"error": str(e)[:150]}

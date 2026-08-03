# ===========================================
# HAI_EPV Engine ver.10 Final — core/meta_label.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - train_meta_label(trade_log_path, out_path) - trenuje meta-klasyfikator
#   (Lopez de Prado meta-labeling) na trade_log istniejacego backtestu:
#   drugi, maly model uczy sie WYLACZNIE "czy warto wziac ten sygnal"
#   (trade/skip), NIE kierunku. Cechy: zgodnosc/rozrzut glosow ensemble,
#   confidence, kontekst rynkowy (bb_pos/regime/adx/atr_pct/rsi_4h/
#   sr_node_strength). Chronologiczny split (nie losowy!) do oceny -
#   losowy split na danych czasowych zawyza wynik przez wyciek informacji.
# - load_meta_label(path) - wczytuje wytrenowany meta-model + scaler
# - score_trade(meta, side, confidence, model_votes, bb_pos, regime,
#   feature_snapshot) - zwraca P(transakcja zyskowna) dla pojedynczego
#   sygnalu, do uzycia jako dodatkowy filtr NAD istniejacym ensemble
#   (bez retrenowania modeli bazowych).
# ===========================================
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
META_LABEL_PATH = MODELS_DIR / "meta_label.pkl"

FEATURE_COLS = [
    "confidence", "vote_mean", "vote_std", "vote_max", "vote_min",
    "bb_pos", "regime", "adx_14", "atr_pct", "sr_node_strength", "rsi_4h", "is_long",
]

DEFAULT_THRESHOLD = 0.5


def _trade_to_row(t: Dict) -> Optional[Dict]:
    mv = t.get("model_votes") or {}
    side = t.get("side")
    if not mv or side not in ("LONG", "SHORT"):
        return None
    key = "long" if side == "LONG" else "short"
    votes = [v.get(key, 0.0) for v in mv.values()]
    fs = t.get("feature_snapshot") or {}
    return {
        "open_ts": t.get("open_ts", 0),
        "confidence": t.get("confidence", 0.0),
        "vote_mean": float(np.mean(votes)) if votes else 0.0,
        "vote_std": float(np.std(votes)) if votes else 0.0,
        "vote_max": float(np.max(votes)) if votes else 0.0,
        "vote_min": float(np.min(votes)) if votes else 0.0,
        "bb_pos": t.get("bb_pos", 0.5),
        "regime": t.get("regime", -1),
        "adx_14": fs.get("adx_14", 0.0),
        "atr_pct": fs.get("atr_pct", 0.0),
        "sr_node_strength": fs.get("sr_node_strength", 0.0),
        "rsi_4h": fs.get("rsi_4h", 50.0),
        "is_long": 1 if side == "LONG" else 0,
        "label": 1 if t.get("pnl_usdt", 0) > 0 else 0,
        "pnl_usdt": t.get("pnl_usdt", 0.0),
    }


def train_meta_label(trade_log_path: str, out_path: Path = META_LABEL_PATH,
                      chrono_test_frac: float = 0.3) -> Dict:
    """Trenuje meta-label classifier na trade_log zapisanego backtestu.
    Zwraca metryki oceny (chronologiczny hold-out) + zapisuje finalny
    model (wytrenowany na WSZYSTKICH danych) do out_path."""
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    d = json.load(open(trade_log_path))
    r = d.get("result", d)
    tl = r.get("trade_log", [])
    rows = [row for row in (_trade_to_row(t) for t in tl) if row is not None]
    if len(rows) < 100:
        raise ValueError(f"Za malo transakcji z model_votes ({len(rows)}) do treningu meta-label")

    df = pd.DataFrame(rows).sort_values("open_ts").reset_index(drop=True)
    n = len(df)
    split_i = int(n * (1 - chrono_test_frac))
    train, test = df.iloc[:split_i], df.iloc[split_i:]

    X_train, y_train = train[FEATURE_COLS].fillna(0.0), train["label"]
    X_test, y_test, pnl_test = test[FEATURE_COLS].fillna(0.0), test["label"], test["pnl_usdt"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    eval_model = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    eval_model.fit(X_train_s, y_train)
    proba_test = eval_model.predict_proba(X_test_s)[:, 1]

    metrics = {"baseline": _pf_wr(pnl_test.values, np.ones(len(pnl_test), dtype=bool))}
    for thr in (0.4, 0.45, 0.5, 0.55, 0.6):
        mask = proba_test >= thr
        metrics[f"thr_{thr}"] = _pf_wr(pnl_test.values, mask)

    # Finalny model: trenowany na WSZYSTKICH dostepnych danych (test set juz
    # jest "przeszloscia" wzgledem przyszlych sygnalow) - dla produkcji.
    X_all_s_scaler = StandardScaler()
    X_all = df[FEATURE_COLS].fillna(0.0)
    X_all_s = X_all_s_scaler.fit_transform(X_all)
    final_model = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    final_model.fit(X_all_s, df["label"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "scaler": X_all_s_scaler,
        "feature_cols": FEATURE_COLS,
        "trained_on": str(trade_log_path),
        "n_trades": n,
        "default_threshold": DEFAULT_THRESHOLD,
        "chrono_eval_metrics": metrics,
        "feature_importance": dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist())),
    }, out_path)

    logger.info(f"meta_label wytrenowany na {n} transakcjach z {trade_log_path} -> {out_path}")
    return metrics


def _pf_wr(pnl: np.ndarray, mask: np.ndarray) -> Dict:
    sub = pnl[mask]
    if len(sub) == 0:
        return {"n": 0, "pf": None, "wr": None}
    wins = sub[sub > 0].sum()
    losses = -sub[sub < 0].sum()
    pf = float(wins / losses) if losses > 0 else None
    wr = float((sub > 0).mean() * 100)
    return {"n": int(len(sub)), "pf": round(pf, 2) if pf else None, "wr": round(wr, 1)}


def load_meta_label(path: Path = META_LABEL_PATH) -> Optional[Dict]:
    import joblib
    if not path.exists():
        return None
    return joblib.load(path)


def score_trade(meta: Dict, side: str, confidence: float, model_votes: Dict,
                bb_pos: float, regime: int, feature_snapshot: Dict) -> float:
    """Zwraca P(transakcja zyskowna) wg meta-modelu dla pojedynczego sygnalu
    z istniejacego ensemble - dodatkowy filtr NAD, nie retrening modeli bazowych."""
    key = "long" if side == "LONG" else "short"
    votes = [v.get(key, 0.0) for v in model_votes.values()]
    row = {
        "confidence": confidence,
        "vote_mean": float(np.mean(votes)) if votes else 0.0,
        "vote_std": float(np.std(votes)) if votes else 0.0,
        "vote_max": float(np.max(votes)) if votes else 0.0,
        "vote_min": float(np.min(votes)) if votes else 0.0,
        "bb_pos": bb_pos,
        "regime": regime if regime is not None else -1,
        "adx_14": feature_snapshot.get("adx_14", 0.0),
        "atr_pct": feature_snapshot.get("atr_pct", 0.0),
        "sr_node_strength": feature_snapshot.get("sr_node_strength", 0.0),
        "rsi_4h": feature_snapshot.get("rsi_4h", 50.0),
        "is_long": 1 if side == "LONG" else 0,
    }
    X = np.array([[row[c] for c in meta["feature_cols"]]])
    X_s = meta["scaler"].transform(X)
    return float(meta["model"].predict_proba(X_s)[0, 1])

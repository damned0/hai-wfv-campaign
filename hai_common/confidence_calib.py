# ===========================================
# HAI_EPV Engine ver.10 Final — core/confidence_calib.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - train_calibration(trade_log_path, out_path) - kalibracja pewnosci per
#   model (Platt scaling - LogisticRegression 1D) - audyt 2026-07-05, na
#   wyrazna prosbe "inna metoda vote labelingu" (kontrast do meta_label.py).
#   ROZNICA vs meta-labeling: meta-labeling dokłada NOWY, DODATKOWY model
#   decyzyjny "czy wziac trade" (post-hoc filtr NAD ensemble). Kalibracja
#   NIE dodaje zadnego nowego kroku - PRZESKALOWUJE istniejace surowe
#   wyjscie KAZDEGO modelu z osobna, zeby liczba faktycznie odzwierciedlala
#   prawdziwe prawdopodobienstwo wygranej (model mowiacy "0.70" nie zawsze
#   wygrywa 70% razy - Platt scaling to koryguje per model).
# - load_calibration(path) - wczytuje wytrenowane kalibratory
# - calibrate(calib, model_name, raw_vote) - zwraca skalibrowane P(win)
#   dla surowego glosu danego modelu.
# ===========================================
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
CALIB_PATH = MODELS_DIR / "confidence_calib.pkl"


def train_calibration(trade_log_path: str, out_path: Path = CALIB_PATH,
                       chrono_test_frac: float = 0.3) -> Dict:
    """Dla kazdego modelu z osobna: Platt scaling (LogisticRegression na 1
    cesze - surowy glos modelu w kierunku faktycznie wzietej strony) ucza
    sie mapowac surowy glos -> skalibrowane P(transakcja zyskowna).
    Chronologiczny split do oceny (nie losowy - patrz meta_label.py)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib

    d = json.load(open(trade_log_path))
    r = d.get("result", d)
    tl = r.get("trade_log", [])

    per_model_rows: Dict[str, list] = {}
    for t in tl:
        mv = t.get("model_votes") or {}
        side = t.get("side")
        if side not in ("LONG", "SHORT"):
            continue
        key = "long" if side == "LONG" else "short"
        label = 1 if t.get("pnl_usdt", 0) > 0 else 0
        ts = t.get("open_ts", 0)
        for mname, v in mv.items():
            raw = v.get(key, 0.0)
            per_model_rows.setdefault(mname, []).append({"open_ts": ts, "raw": raw, "label": label})

    calibrators: Dict[str, Dict] = {}
    metrics: Dict[str, Dict] = {}
    for mname, rows in per_model_rows.items():
        if len(rows) < 30:
            logger.info(f"calib {mname}: za malo danych ({len(rows)}) - pominiety")
            continue
        df = pd.DataFrame(rows).sort_values("open_ts").reset_index(drop=True)
        n = len(df)
        split_i = int(n * (1 - chrono_test_frac))
        train, test = df.iloc[:split_i], df.iloc[split_i:]

        X_train = train[["raw"]].values
        y_train = train["label"].values
        if len(set(y_train.tolist())) < 2:
            continue

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        X_test = test[["raw"]].values
        y_test = test["label"].values
        raw_test = test["raw"].values
        calib_proba = clf.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])

        # Ocena: czy skalibrowana proba jest blizej realnej WR w binach niz surowy glos
        def _calib_error(pred, actual):
            if len(pred) == 0:
                return None
            bins = np.linspace(0, 1, 6)
            errs = []
            for i in range(len(bins) - 1):
                m = (pred >= bins[i]) & (pred < bins[i + 1])
                if m.sum() < 5:
                    continue
                errs.append(abs(pred[m].mean() - actual[m].mean()))
            return round(float(np.mean(errs)), 4) if errs else None

        raw_err = _calib_error(raw_test, y_test)
        calib_err = _calib_error(calib_proba, y_test) if len(calib_proba) else None

        calibrators[mname] = {"coef": float(clf.coef_[0][0]), "intercept": float(clf.intercept_[0])}
        metrics[mname] = {"n_train": len(train), "n_test": len(test),
                           "raw_calibration_error": raw_err, "calibrated_error": calib_err}
        logger.info(f"calib {mname}: n={n} raw_err={raw_err} calib_err={calib_err}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "calibrators": calibrators,
        "trained_on": str(trade_log_path),
        "metrics": metrics,
    }, out_path)
    return metrics


def load_calibration(path: Path = CALIB_PATH) -> Optional[Dict]:
    import joblib
    if not path.exists():
        return None
    return joblib.load(path)


def calibrate(calib: Dict, model_name: str, raw_vote: float) -> float:
    """Zwraca skalibrowane P(win) dla surowego glosu modelu. Jesli model
    nie ma kalibratora (za malo danych przy treningu), zwraca raw_vote
    bez zmian (fallback bezpieczny). Dopasowanie po nazwie ignoruje sufiks
    _NEW (audyt 2026-07-05) - load-slots laduje sloty z nazwami plikow typu
    'rf_NEW', kalibrator byl trenowany na 'rf' (nazwa bez stagingu)."""
    cals = calib.get("calibrators", {})
    c = cals.get(model_name) or cals.get(model_name.replace("_NEW", ""))
    if c is None:
        return raw_vote
    z = c["coef"] * raw_vote + c["intercept"]
    return float(1.0 / (1.0 + np.exp(-z)))

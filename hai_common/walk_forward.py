# ===========================================
# HAI_EPV Engine ver.10 Final — core/walk_forward.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: walk_forward_eval() (3-fold expanding evaluation, wybor
# najlepszego algorytmu przed finalnym treningiem), select_best_model().
# Importowane w ml_trainer.py.
# ===========================================
import numpy as np
import time
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

N_WF_FOLDS = 3
WF_SELECT_METRIC = 'wf_precision'
WF_MIN_RECALL = 0.10


_CW3 = {0: 1.0, 1: 2.5, 2: 2.5}  # neutral/long/short — jak w ml_trainer.py


def _make_model(name):
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=-1, class_weight=_CW3)
    if name == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05,
            min_child_samples=20, num_leaves=31, objective='multiclass', num_class=3,
            random_state=42, n_jobs=-1, verbose=-1, class_weight=_CW3)
    if name == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            objective='multi:softprob', num_class=3,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0)
    if name == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05,
            random_state=42, verbose=False, loss_function='MultiClass', classes_count=3,
            class_weights=[1.0, 2.5, 2.5])
    if name == 'histgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight=_CW3)
    raise ValueError(name)


def walk_forward_eval(df, model_name, feature_names, label_col='label_long', n_folds=N_WF_FOLDS):
    n = len(df)
    bounds = [int(n * i / (n_folds + 1)) for i in range(n_folds + 2)]
    accs, precs, recs, f1s, precs_l, precs_s, fold_details = [], [], [], [], [], [], []
    for k in range(n_folds):
        train_end = bounds[k + 1]
        test_start = bounds[k + 1]
        test_end = bounds[k + 2]
        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]
        if len(train) < 100 or len(test) < 50:
            continue
        X_tr = train[feature_names].values
        y_tr = train[label_col].values
        X_te = test[feature_names].values
        y_te = test[label_col].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = _make_model(model_name)
        if model_name == 'xgb':
            sw = np.array([_CW3[int(v)] for v in y_tr])
            model.fit(X_tr_s, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        if model_name == 'cat':
            y_pred = y_pred.astype(int).ravel()
        pc_prec = precision_score(y_te, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        accs.append(accuracy_score(y_te, y_pred))
        precs.append(precision_score(y_te, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_te, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_te, y_pred, average='macro', zero_division=0))
        precs_l.append(float(pc_prec[1]))
        precs_s.append(float(pc_prec[2]))
        fold_details.append({'fold': k + 1, 'train_size': len(train),
            'test_size': len(test), 'acc': round(accs[-1], 4), 'f1': round(f1s[-1], 4)})
    return {
        'wf_accuracy': round(float(np.mean(accs)), 4) if accs else 0.0,
        'wf_acc_std': round(float(np.std(accs)), 4) if accs else 0.0,
        'wf_precision': round(float(np.mean(precs)), 4) if precs else 0.0,
        'wf_recall': round(float(np.mean(recs)), 4) if recs else 0.0,
        'wf_f1': round(float(np.mean(f1s)), 4) if f1s else 0.0,
        'wf_precision_long': round(float(np.mean(precs_l)), 4) if precs_l else 0.0,
        'wf_precision_short': round(float(np.mean(precs_s)), 4) if precs_s else 0.0,
        'wf_folds': fold_details, 'wf_n_folds': len(accs),
    }


def select_best_model(wf_results, logger=None):
    def log(msg):
        if logger: logger.info(msg)
        else: print(msg)
    log("=== WALK-FORWARD: porownanie modeli ===")
    log(f"  kryterium: {WF_SELECT_METRIC}, prog recall >= {WF_MIN_RECALL}")
    for name, r in wf_results.items():
        log(f"  {name.upper():4} | prec={r['wf_precision']:.4f} rec={r['wf_recall']:.4f} "
            f"f1={r['wf_f1']:.4f} acc={r['wf_accuracy']:.4f} (acc_std={r['wf_acc_std']:.4f}) "
            f"| foldow={r['wf_n_folds']}")
    eligible = {n: r for n, r in wf_results.items() if r['wf_recall'] >= WF_MIN_RECALL}
    if not eligible:
        best = max(wf_results.items(), key=lambda kv: kv[1]['wf_recall'])
        reason = f"FALLBACK: zaden model recall>={WF_MIN_RECALL}, wybrano najwyzszy recall"
        log(f"  -> {best[0].upper()} ({reason})")
        return best[0], reason
    best = max(eligible.items(), key=lambda kv: kv[1][WF_SELECT_METRIC])
    reason = f"najlepszy {WF_SELECT_METRIC}={best[1][WF_SELECT_METRIC]:.4f} (recall={best[1]['wf_recall']:.4f} OK)"
    log(f"  -> ZWYCIEZCA: {best[0].upper()} ({reason})")
    return best[0], reason

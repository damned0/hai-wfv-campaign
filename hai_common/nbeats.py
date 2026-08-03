"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) — LAB v9.0
Global model trained on all 105 symbols, CPU-only.

Architecture:
  - 2 stacks × 2 blocks × 4 FC layers (256 units, ReLU)
  - Generic basis (identity projection = "N-BEATS-G")
  - Lookback: 168h (7 days of 1h bars)
  - Forecast: 4h ahead → nbeats_pred_return_4h

Training:
  - Input: normalized log-returns of last 168 closes
  - Target: sum of next 4 log-returns (= 4h cumulative return)
  - Loss: Huber (robust to outliers in crypto)
  - Global model: one model for ALL symbols

Usage:
  train_nbeats(symbols, wh_base, save_path, days=365)
  pred = predict_nbeats_series(closes_1h, model, scaler)
  val  = predict_nbeats_scalar(closes_168, model, scaler)  # live inference
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

LOOKBACK  = 168   # 7 days × 24h
FORECAST  = 4     # 4h ahead
LAYER_SIZE = 256
N_BLOCKS   = 2
N_STACKS   = 2
EPOCHS     = 25
BATCH_SIZE = 1024
LR         = 3e-4
CLIP_NORM  = 1.0


# ─── Model ────────────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,  LAYER_SIZE), nn.ReLU(),
            nn.Linear(LAYER_SIZE,  LAYER_SIZE), nn.ReLU(),
            nn.Linear(LAYER_SIZE,  LAYER_SIZE), nn.ReLU(),
            nn.Linear(LAYER_SIZE,  LAYER_SIZE), nn.ReLU(),
        )
        self.theta_b = nn.Linear(LAYER_SIZE, backcast_size,  bias=False)
        self.theta_f = nn.Linear(LAYER_SIZE, forecast_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBeatsForecast(nn.Module):
    """
    N-BEATS-G (generic basis).
    Each block produces backcast (subtract from input) and forecast (add to output).
    """
    def __init__(self):
        super().__init__()
        self.stacks = nn.ModuleList([
            nn.ModuleList([
                NBeatsBlock(LOOKBACK, LOOKBACK, FORECAST)
                for _ in range(N_BLOCKS)
            ])
            for _ in range(N_STACKS)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast  = torch.zeros(x.size(0), FORECAST, device=x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual  = residual  - backcast
                forecast  = forecast  + block_forecast
        return forecast


# ─── Data prep ────────────────────────────────────────────────────────────────

def _extract_windows(closes: np.ndarray, days: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) windows from 1h close price series.

    X shape: (n_windows, LOOKBACK)  — normalized log-returns
    y shape: (n_windows,)           — 4h cumulative log-return
    """
    n_keep = days * 24
    if len(closes) > n_keep:
        closes = closes[-n_keep:]

    log_r = np.diff(np.log(np.clip(closes, 1e-10, None))).astype(np.float32)
    n = len(log_r)
    min_len = LOOKBACK + FORECAST
    if n < min_len:
        return np.empty((0, LOOKBACK), np.float32), np.empty(0, np.float32)

    step = 2
    starts = range(0, n - min_len + 1, step)
    Xs, ys = [], []
    for s in starts:
        window = log_r[s: s + LOOKBACK]
        target = log_r[s + LOOKBACK: s + LOOKBACK + FORECAST].sum()
        std = window.std()
        if std < 1e-8:
            continue
        Xs.append(window / (std + 1e-8))
        ys.append(target / (std + 1e-8))

    if not Xs:
        return np.empty((0, LOOKBACK), np.float32), np.empty(0, np.float32)
    return np.array(Xs, np.float32), np.array(ys, np.float32)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_nbeats(wh_base: Path, save_path: Path, days: int = 365,
                 max_windows: int = 400_000) -> dict:
    """Train global N-BEATS model on all symbols in warehouse.

    Args:
        wh_base:     Path to /data_warehouse/ohlcv/binance
        save_path:   Where to save {'model': state_dict, 'meta': ...}
        days:        How many days of history to use
        max_windows: Cap total training windows (memory/speed)

    Returns:
        dict with training metrics
    """
    import pandas as pd

    symbols = sorted(f.stem for f in (wh_base / '1h').glob('*.parquet'))
    logger.info(f'N-BEATS training: {len(symbols)} symbols | {days}d | max_windows={max_windows:,}')

    all_X, all_y = [], []
    for sym in symbols:
        p = wh_base / '1h' / f'{sym}.parquet'
        try:
            closes = pd.read_parquet(p, columns=['close'])['close'].values.astype(np.float32)
            X, y = _extract_windows(closes, days)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            logger.debug(f'  {sym}: {e}')

    if not all_X:
        raise RuntimeError('No training data')

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Cap + shuffle
    if len(X_all) > max_windows:
        idx = np.random.default_rng(42).choice(len(X_all), max_windows, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    # Clip extreme targets (>10 std)
    y_std = y_all.std()
    y_all = np.clip(y_all, -10 * y_std, 10 * y_std)

    logger.info(f'  Windows: {len(X_all):,} | y mean={y_all.mean():.4f} std={y_all.std():.4f}')

    # Split 90/10
    split = int(len(X_all) * 0.9)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    model = NBeatsForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_dl:
            optimizer.zero_grad()
            pred = model(Xb)
            # pred shape (B, FORECAST) → sum across horizon
            pred_sum = pred.sum(dim=1)
            loss = loss_fn(pred_sum, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in va_dl:
                pred = model(Xb).sum(dim=1)
                va_loss += loss_fn(pred, yb).item() * len(Xb)

        tr_loss /= len(tr_ds)
        va_loss /= len(va_ds)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f'  Epoch {epoch:2d}/{EPOCHS} | tr={tr_loss:.4f} | va={va_loss:.4f} | lr={lr_now:.6f}')

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    save_data = {
        'model_state': best_state,
        'val_loss':    best_val,
        'windows':     len(X_all),
        'elapsed_sec': round(elapsed, 1),
        'lookback':    LOOKBACK,
        'forecast':    FORECAST,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(save_data, save_path)
    logger.info(f'N-BEATS saved → {save_path} | val_loss={best_val:.4f} | {elapsed:.0f}s')

    return save_data


# ─── Inference ────────────────────────────────────────────────────────────────

_cached_model: Optional[NBeatsForecast] = None
_cached_path:  Optional[Path] = None


def _load_model(model_path: Path) -> Optional[NBeatsForecast]:
    global _cached_model, _cached_path
    if _cached_model is not None and _cached_path == model_path:
        return _cached_model
    if not model_path.exists():
        return None
    try:
        data = joblib.load(model_path)
        m = NBeatsForecast()
        m.load_state_dict(data['model_state'])
        m.eval()
        _cached_model = m
        _cached_path  = model_path
        logger.info(f'N-BEATS loaded from {model_path} (val_loss={data.get("val_loss", "?"):.4f})')
        return m
    except Exception as e:
        logger.warning(f'N-BEATS load failed: {e}')
        return None


def predict_nbeats_series(closes: np.ndarray, model_path: Path) -> np.ndarray:
    """Generate N-BEATS predictions for ALL bars in a 1h close series.

    Used in ml_trainer to add nbeats_pred_return_4h to training dataset.
    Returns array of shape (len(closes),) with 0.0 for first LOOKBACK bars.
    """
    model = _load_model(model_path)
    if model is None:
        return np.zeros(len(closes), dtype=np.float32)

    log_r = np.diff(np.log(np.clip(closes, 1e-10, None))).astype(np.float32)
    n = len(log_r)
    result = np.zeros(len(closes), dtype=np.float32)

    if n < LOOKBACK:
        return result

    # Batch all valid windows
    starts  = np.arange(0, n - LOOKBACK + 1)
    windows = np.stack([log_r[s: s + LOOKBACK] for s in starts])  # (N, 168)
    stds    = windows.std(axis=1, keepdims=True)
    stds    = np.where(stds < 1e-8, 1.0, stds)
    windows_norm = (windows / stds).astype(np.float32)

    with torch.no_grad():
        X_t = torch.from_numpy(windows_norm)
        preds = []
        for i in range(0, len(X_t), 4096):
            batch = X_t[i: i + 4096]
            out   = model(batch).sum(dim=1).numpy()  # (B,) summed over 4 forecast steps
            preds.append(out)
        pred_arr = np.concatenate(preds)

    # Re-scale back to log-return units
    pred_arr = pred_arr * stds.squeeze() * FORECAST / FORECAST

    # Align: prediction at position s+LOOKBACK (i.e. the bar after the window)
    for s in range(len(starts)):
        pos = s + LOOKBACK  # index in closes array (+1 for np.diff offset)
        if pos < len(result):
            result[pos] = float(pred_arr[s])

    return result


def predict_nbeats_scalar(recent_closes: np.ndarray, model_path: Path) -> float:
    """Single-step inference for live feature generation.

    Args:
        recent_closes: last LOOKBACK+1 closes (need +1 to compute LOOKBACK log-returns)
        model_path:    path to saved N-BEATS model

    Returns:
        Predicted 4h cumulative log-return (float)
    """
    model = _load_model(model_path)
    if model is None:
        return 0.0

    closes = np.array(recent_closes, dtype=np.float32)
    if len(closes) < LOOKBACK + 1:
        return 0.0

    log_r = np.diff(np.log(np.clip(closes[-(LOOKBACK + 1):], 1e-10, None)))
    std = log_r.std()
    if std < 1e-8:
        return 0.0

    x_norm = torch.from_numpy((log_r / std).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x_norm).sum(dim=1).item()

    return float(np.clip(pred * std, -0.20, 0.20))


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    WH = Path('/root/ProjektHAI/data_warehouse/ohlcv/binance')
    OUT = Path('/root/ProjektHAI/HAIs/HAI_LIV/data/models/nbeats.pkl')

    result = train_nbeats(WH, OUT, days=365)
    print(f"Done: val_loss={result['val_loss']:.4f} | windows={result['windows']:,} | {result['elapsed_sec']}s")

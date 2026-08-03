"""
Transformer Time Series Forecaster — LAB v10.0
Global model trained on all 105 symbols, CPU inference / GPU training.

Architecture:
  - Input: normalized log-returns, lookback 168h (7 days × 1h)
  - Positional encoding (learned)
  - 2× Transformer encoder layers (d_model=128, nhead=4, dim_ff=256)
  - Global average pooling → Linear → 4h cumulative log-return
  - Forecast: 4h ahead → transformer_pred_return_4h

Training: identical pipeline to nbeats.py
  - Huber loss, Adam, CosineAnnealingLR
  - Global model: one model for ALL symbols
  - Input normalized per-window (std normalization)
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

LOOKBACK   = 168
FORECAST   = 4
D_MODEL    = 128
NHEAD      = 4
DIM_FF     = 256
N_LAYERS   = 2
DROPOUT    = 0.1
EPOCHS     = 25
BATCH_SIZE = 2048
LR         = 3e-4
CLIP_NORM  = 1.0


# ─── Model ────────────────────────────────────────────────────────────────────

class TransformerForecast(nn.Module):
    """Encoder-only Transformer for time series regression."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1, D_MODEL)
        self.pos_emb = nn.Embedding(LOOKBACK, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dim_feedforward=DIM_FF, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, LOOKBACK)
        x = x.unsqueeze(-1)                           # (B, L, 1)
        x = self.input_proj(x)                        # (B, L, D_MODEL)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)                     # add positional encoding
        x = self.encoder(x)                           # (B, L, D_MODEL)
        x = x.mean(dim=1)                             # global average pool → (B, D_MODEL)
        return self.head(x).squeeze(-1)               # (B,)


# ─── Data prep (shared with nbeats) ───────────────────────────────────────────

def _extract_windows(closes: np.ndarray, days: int):
    n_keep = days * 24
    if len(closes) > n_keep:
        closes = closes[-n_keep:]
    log_r = np.diff(np.log(np.clip(closes, 1e-10, None))).astype(np.float32)
    n = len(log_r)
    if n < LOOKBACK + FORECAST:
        return np.empty((0, LOOKBACK), np.float32), np.empty(0, np.float32)
    step = 2
    Xs, ys = [], []
    for s in range(0, n - LOOKBACK - FORECAST + 1, step):
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

def train_transformer(wh_base: Path, save_path: Path, days: int = 365,
                      max_windows: int = 400_000) -> dict:
    """Train global Transformer on all symbols in warehouse.

    Args:
        wh_base:     Path to /data_warehouse/ohlcv/binance (or /workspace/data)
        save_path:   Where to save {'model_state': ..., 'meta': ...}
        days:        Days of history to use
        max_windows: Cap total windows

    Returns:
        dict with training metrics
    """
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Transformer training on {device}')

    symbols = sorted(f.stem for f in (wh_base / '1h').glob('*.parquet'))
    logger.info(f'Transformer training: {len(symbols)} symbols | {days}d | device={device}')

    all_X, all_y = [], []
    for sym in symbols:
        p = wh_base / '1h' / f'{sym}.parquet'
        try:
            closes = pd.read_parquet(p, columns=['close'])['close'].values.astype(np.float32)
            X, y = _extract_windows(closes, days)
            if len(X) > 0:
                all_X.append(X); all_y.append(y)
        except Exception as e:
            logger.debug(f'  {sym}: {e}')

    if not all_X:
        raise RuntimeError('No training data')

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    if len(X_all) > max_windows:
        idx = np.random.default_rng(42).choice(len(X_all), max_windows, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    y_std = y_all.std()
    y_all = np.clip(y_all, -10 * y_std, 10 * y_std)
    logger.info(f'  Windows: {len(X_all):,} | y mean={y_all.mean():.4f} std={y_all.std():.4f}')

    split = int(len(X_all) * 0.9)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    model = TransformerForecast().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val, best_state = float('inf'), None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in va_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                va_loss += loss_fn(model(Xb), yb).item() * len(Xb)

        tr_loss /= len(tr_dl.dataset)
        va_loss /= len(va_dl.dataset)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f'  Epoch {epoch:2d}/{EPOCHS} | tr={tr_loss:.4f} | va={va_loss:.4f} | lr={lr_now:.6f}')

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
    logger.info(f'Transformer saved → {save_path} | val_loss={best_val:.4f} | {elapsed:.0f}s')
    return save_data


# ─── Inference ────────────────────────────────────────────────────────────────

_cached_model: Optional[TransformerForecast] = None
_cached_path:  Optional[Path] = None


def _load_model(model_path: Path) -> Optional[TransformerForecast]:
    global _cached_model, _cached_path
    if _cached_model is not None and _cached_path == model_path:
        return _cached_model
    if not model_path.exists():
        return None
    try:
        data = joblib.load(model_path)
        m = TransformerForecast()
        m.load_state_dict(data['model_state'])
        m.eval()
        _cached_model = m
        _cached_path  = model_path
        logger.info(f'Transformer loaded from {model_path} (val_loss={data.get("val_loss", "?"):.4f})')
        return m
    except Exception as e:
        logger.warning(f'Transformer load failed: {e}')
        return None


def predict_transformer_series(closes: np.ndarray, model_path: Path) -> np.ndarray:
    """Batch inference for entire 1h close series (used in ml_trainer)."""
    model = _load_model(model_path)
    if model is None:
        return np.zeros(len(closes), dtype=np.float32)

    log_r = np.diff(np.log(np.clip(closes, 1e-10, None))).astype(np.float32)
    n = len(log_r)
    result = np.zeros(len(closes), dtype=np.float32)
    if n < LOOKBACK:
        return result

    starts  = np.arange(0, n - LOOKBACK + 1)
    windows = np.stack([log_r[s: s + LOOKBACK] for s in starts])
    stds    = windows.std(axis=1, keepdims=True)
    stds    = np.where(stds < 1e-8, 1.0, stds)
    windows_norm = (windows / stds).astype(np.float32)

    with torch.no_grad():
        X_t = torch.from_numpy(windows_norm)
        preds = []
        # batch=2048 (nie 4096): self-attention pamięć skaluje się jak
        # batch × seq_len² × nhead. LIV ma mniejszy model niż LAB (D_MODEL=128,
        # N_LAYERS=2 vs 256/4) więc jest z natury lżejszy, ale przy
        # MAX_CONCURRENT_SYMBOLS>1 warto zachować ten sam margines co w LAB.
        for i in range(0, len(X_t), 2048):
            out = model(X_t[i: i + 2048]).numpy()
            preds.append(out)
        pred_arr = np.concatenate(preds)

    pred_arr = pred_arr * stds.squeeze()

    for s in range(len(starts)):
        pos = s + LOOKBACK
        if pos < len(result):
            result[pos] = float(pred_arr[s])

    return result


def predict_transformer_scalar(recent_closes: np.ndarray, model_path: Path) -> float:
    """Single-step live inference from last LOOKBACK+1 closes."""
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

    x = torch.from_numpy((log_r / std).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).item()

    return float(np.clip(pred * std, -0.20, 0.20))


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    WH  = Path('/workspace/data')
    OUT = Path('/workspace/transformer.pkl')

    result = train_transformer(WH, OUT, days=365)
    print(f"Done: val_loss={result['val_loss']:.4f} | windows={result['windows']:,} | {result['elapsed_sec']}s")

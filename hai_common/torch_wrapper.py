"""
HAI_EPV Engine ver.10 Final — core/torch_wrapper.py
Created by Hauzer | Coded & produced by Claude Sonnet 5

Neural model architectures (LSTM/TCN/Transformer) + sklearn-compatible
wrapper. Defined here (not in __main__) so joblib pkl files reference
core.torch_wrapper.* i moga byc wczytane z dowolnego kontekstu (ensemble,
backtest, API). Ablowane z aktywnego ensemble EPV (szkodzily PF jako
glosujace) - kod zostaje dla LIV/TST ktore ich uzywaja.
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "MLP", "LSTM", "TCNBlock", "TCN", "TransformerClassifier", "TorchWrapper",
]

N_FEAT_DEFAULT = 24
N_CLASSES = 3


# ── model architectures ──────────────────────────────────────────────────────

class MLP(nn.Module):
    """Tabular MLP — 22/24→512→256→128→64→3 with residual skip."""
    def __init__(self, n_feat: int = N_FEAT_DEFAULT, n_cls: int = N_CLASSES):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(n_feat)
        self.fc1   = nn.Linear(n_feat, 512);  self.bn1 = nn.BatchNorm1d(512)
        self.fc2   = nn.Linear(512, 256);     self.bn2 = nn.BatchNorm1d(256)
        self.fc3   = nn.Linear(256, 128);     self.bn3 = nn.BatchNorm1d(128)
        self.fc4   = nn.Linear(128, 64)
        self.head  = nn.Linear(64, n_cls)
        self.res1  = nn.Linear(n_feat, 256)
        self.drop  = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.act   = nn.GELU()

    def forward(self, x):
        x = self.bn_in(x)
        skip = self.res1(x)
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x)))) + skip
        x = self.drop2(self.act(self.bn3(self.fc3(x))))
        x = self.act(self.fc4(x))
        return self.head(x)


class LSTM(nn.Module):
    """LSTM seq classifier — hidden=256, 3 layers, seq_len=48."""
    def __init__(self, n_feat: int = N_FEAT_DEFAULT, hidden: int = 256, layers: int = 3, n_cls: int = N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=0.25)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.shape[-1]]  # causal trim
        return self.act(self.bn(out)) + self.res(x)


class TCN(nn.Module):
    """TCN — 256 channels, 6 dilated blocks, receptive field ~192 bars."""
    def __init__(self, n_feat: int = N_FEAT_DEFAULT, channels: int = 256, n_cls: int = N_CLASSES):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(n_feat,   channels, kernel=3, dilation=1),
            TCNBlock(channels, channels, kernel=3, dilation=2),
            TCNBlock(channels, channels, kernel=3, dilation=4),
            TCNBlock(channels, channels, kernel=3, dilation=8),
            TCNBlock(channels, channels, kernel=3, dilation=16),
            TCNBlock(channels, channels, kernel=3, dilation=32),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(channels, n_cls)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, seq, feat) → (B, feat, seq)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier — d_model=128, 4 heads, 4 layers."""
    def __init__(self, n_feat: int = N_FEAT_DEFAULT, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, n_cls: int = N_CLASSES, max_seq: int = 64):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.proj(x) + self.pos_emb(pos)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


# ── sklearn-compatible wrapper ───────────────────────────────────────────────

class TorchWrapper:
    """Wraps a PyTorch model with sklearn-style predict_proba / predict.

    Saved via joblib as core.torch_wrapper.TorchWrapper — importable from any
    context (backtest, ensemble, API) without requiring __main__ to be the
    training script.
    """

    def __init__(self, model: nn.Module, scaler, feature_names,
                 model_type: str, seq_len: int = 0, device: str = "cpu",
                 accuracy: float = 0.0, precision: float = 0.0, recall: float = 0.0,
                 f1: float = 0.0, trained_at: str = ""):
        self.model         = model.to("cpu")
        self.scaler        = scaler
        self.feature_names = feature_names
        self.model_type    = model_type
        self.seq_len       = seq_len
        self.device_str    = device
        self.accuracy      = accuracy
        self.precision     = precision
        self.recall        = recall
        self.f1            = f1
        self.trained_at    = trained_at

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (n, n_feat) for MLP, (n, seq_len, n_feat) for LSTM/TCN/Transformer."""
        self.model.eval()
        with torch.no_grad():
            if self.scaler is not None:
                if self.model_type == "mlp":
                    X = self.scaler.transform(X)
                else:
                    sh = X.shape
                    X = self.scaler.transform(X.reshape(-1, sh[-1])).reshape(sh)
            t = torch.FloatTensor(X)
            logits = self.model(t)
            probs  = torch.softmax(logits, dim=-1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

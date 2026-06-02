"""Deep-learning baselines, trained on the same ARIMA-augmented features and the
same expanding-window folds as every other method. Reported metrics average
over several random seeds to reduce initialisation variance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

from .evaluation import augmented_fold, metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LSTM(nn.Module):
    def __init__(self, input_size, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, input_size, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


class _BiLSTM(nn.Module):
    def __init__(self, input_size, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


class _AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        out, _ = self.rnn(x)
        attn, _ = self.attn(out, out, out)
        return self.fc(self.drop(attn[:, -1, :])).squeeze(-1)


class _CNN(nn.Module):
    def __init__(self, input_size, filters=32):
        super().__init__()
        self.c1 = nn.Conv1d(1, filters, 3, padding=1)
        self.c2 = nn.Conv1d(filters, filters, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.drop(self.act(self.c1(x)))
        x = self.act(self.c2(x))
        return self.fc(self.pool(x).squeeze(-1)).squeeze(-1)


class _Transformer(nn.Module):
    def __init__(self, input_size, d_model=32):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.enc(self.proj(x))
        return self.fc(self.drop(x[:, -1, :])).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation):
        super().__init__()
        pad = 2 * dilation
        self.c1 = nn.Conv1d(c_in, c_out, 3, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(c_out, c_out, 3, padding=pad, dilation=dilation)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        out = self.drop(self.act(self.c1(x)))[:, :, : x.size(2)]
        out = self.drop(self.act(self.c2(out)))[:, :, : x.size(2)]
        return self.act(out + res)


class _TCN(nn.Module):
    def __init__(self, input_size, channels=32, layers=2):
        super().__init__()
        blocks = [_TCNBlock(1 if i == 0 else channels, channels, 2 ** i) for i in range(layers)]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        return self.fc(self.pool(self.net(x)).squeeze(-1)).squeeze(-1)


class _CNNLSTM(nn.Module):
    def __init__(self, input_size, filters=16, hidden=16):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, 3, padding=1)
        self.act = nn.ReLU()
        self.rnn = nn.LSTM(filters, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.act(self.conv(x)).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


ARCHITECTURES = {
    "LSTM": _LSTM, "GRU": _GRU, "BiLSTM": _BiLSTM, "Attention LSTM": _AttentionLSTM,
    "1D CNN": _CNN, "Transformer": _Transformer, "TCN": _TCN, "CNN-LSTM": _CNNLSTM,
}


def _train_predict(model_cls, x_tr, y_tr, x_te, epochs=500, lr=1e-3, patience=50):
    model = model_cls(x_tr.shape[1]).to(DEVICE)
    y_mean, y_std = float(np.mean(y_tr)), float(np.std(y_tr)) or 1.0
    xt = torch.tensor(x_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((y_tr - y_mean) / y_std, dtype=torch.float32, device=DEVICE)
    xe = torch.tensor(x_te, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())
        if loss.item() < best:
            best, best_state, bad = loss.item(), {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(xe).cpu().numpy() * y_std + y_mean


def evaluate_deep_models(X, y, folds, seeds=5, k=8):
    """Return {name: metrics} for all architectures, averaged over seeds."""
    results = {}
    for name, cls in ARCHITECTURES.items():
        seed_preds, actuals = [], None
        for seed in range(seeds):
            torch.manual_seed(seed * 42)
            np.random.seed(seed * 42)
            preds, act = [], []
            for train_idx, test_idx in folds:
                x_tr, x_te = augmented_fold(X, y, train_idx, test_idx, k)
                scaler = RobustScaler()
                try:
                    pred = _train_predict(cls, scaler.fit_transform(x_tr), y[train_idx],
                                          scaler.transform(x_te))
                except Exception:
                    pred = np.array([y[train_idx][-1]])
                preds.append(float(np.atleast_1d(pred)[0]))
                act.append(float(y[test_idx][0]))
            seed_preds.append(preds)
            actuals = act
        results[name] = metrics(actuals, np.mean(seed_preds, axis=0))
    return results

"""
lstm_common.py
Everything the LSTM steps share (05 walk-forward, 06 frozen models, 07 SHAP): data preparation,
the network, training with early stopping, and forecasting. Keeping it in one place guarantees that
every step uses the identical, pre-registered set-up.

Leakage rules (see PREREGISTRATION.md):
  * scalers are fitted on the training part of each window only;
  * the last HORIZON days before a boundary are dropped, because their 5-day targets cross it;
  * validation = the last 12 months of the training window, used only for early stopping and the
    back-transformation (smearing) factor.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation import SAMPLE_END, HORIZON

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ["ret", "r2", "us_ret", "us_r2", "euro_ret", "euro_r2"]
LOG_INPUTS = ["r2", "us_r2", "euro_r2"]
CHANNEL = {"ret": "domestic", "r2": "domestic", "us_ret": "US", "us_r2": "US", "euro_ret": "euro", "euro_r2": "euro"}
FLOOR = 1e-8


@dataclass(frozen=True)
class Config:
    lookback: int = 22
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 64
    max_epochs: int = 200
    patience: int = 15


def load_frame(inputs=INPUTS) -> pd.DataFrame:
    """Model-ready frame: inputs (squared returns in logs) and the log target, main sample only."""
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True)
    d = d.loc[:SAMPLE_END].copy()
    for c in LOG_INPUTS + [c for c in inputs if c.endswith("_r2") and c not in LOG_INPUTS]:
        if c in d:
            d[c] = np.log(d[c].clip(lower=FLOOR))
    return d.dropna(subset=list(inputs))


class LSTMVol(nn.Module):
    def __init__(self, n_in, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(n_in, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                            dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def windows(X: np.ndarray, positions: np.ndarray, lookback: int) -> np.ndarray:
    """Stack the look-back window ending at each position: shape (n, lookback, n_inputs)."""
    return np.stack([X[p - lookback + 1: p + 1] for p in positions]).astype(np.float32)


def split_positions(d: pd.DataFrame, train_end, lookback: int):
    """Positions for fitting and validation inside the training window ending at train_end."""
    dates = d.index
    has_y = d["log_rv5_fwd"].notna().to_numpy()
    ok = np.arange(len(d)) >= lookback - 1
    train_end = pd.Timestamp(train_end)
    val_start = train_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    fit = np.where(ok & has_y & (dates < val_start))[0][:-HORIZON]
    val = np.where(ok & has_y & (dates >= val_start) & (dates <= train_end))[0][:-HORIZON]
    return fit, val


def fit_model(d: pd.DataFrame, train_end, cfg: Config, seed: int, inputs=INPUTS):
    """Train one LSTM on data up to train_end. Returns a bundle with everything needed to forecast."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_pos, val_pos = split_positions(d, train_end, cfg.lookback)
    fit_dates = d.index[fit_pos]
    mu, sd = d.loc[:fit_dates[-1], inputs].mean(), d.loc[:fit_dates[-1], inputs].std()
    y_mu, y_sd = d["log_rv5_fwd"].iloc[fit_pos].mean(), d["log_rv5_fwd"].iloc[fit_pos].std()
    X = ((d[inputs] - mu) / sd).to_numpy(np.float32)
    y = ((d["log_rv5_fwd"] - y_mu) / y_sd).to_numpy(np.float32)
    Xf, yf = torch.tensor(windows(X, fit_pos, cfg.lookback)), torch.tensor(y[fit_pos])
    Xv, yv = torch.tensor(windows(X, val_pos, cfg.lookback)), torch.tensor(y[val_pos])

    model = LSTMVol(len(inputs), cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, epoch = np.inf, None, 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xf))
        for b in range(0, len(Xf), cfg.batch):
            j = perm[b:b + cfg.batch]
            opt.zero_grad()
            loss_fn(model(Xf[j]), yf[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
        if v < best - 1e-4:
            best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(Xv).numpy() * y_sd + y_mu
    smear = float(np.mean(np.exp(d["log_rv5_fwd"].iloc[val_pos].to_numpy() - pv)))   # Duan smearing on validation
    return {"model": model, "cfg": cfg, "inputs": list(inputs), "mu": mu, "sd": sd, "y_mu": y_mu, "y_sd": y_sd,
            "smear": smear, "epochs": epoch, "val_loss": best, "seed": seed, "train_end": str(train_end),
            "n_fit": len(fit_pos), "n_val": len(val_pos)}


def forecast(bundle, d: pd.DataFrame, dates) -> pd.Series:
    """Variance forecasts (decimal units) for the given dates, using data up to each date only."""
    cfg = bundle["cfg"]
    X = ((d[bundle["inputs"]] - bundle["mu"]) / bundle["sd"]).to_numpy(np.float32)
    pos = d.index.get_indexer(pd.DatetimeIndex(dates))
    with torch.no_grad():
        z = bundle["model"](torch.tensor(windows(X, pos, cfg.lookback))).numpy()
    return pd.Series(np.exp(z * bundle["y_sd"] + bundle["y_mu"]) * bundle["smear"], index=pd.DatetimeIndex(dates))


def save_bundle(bundle, path: Path):
    torch.save({k: (v.state_dict() if k == "model" else v) for k, v in bundle.items()}, path)


def load_bundle(path: Path):
    b = torch.load(path, weights_only=False)
    model = LSTMVol(len(b["inputs"]), b["cfg"])
    model.load_state_dict(b["model"])
    model.eval()
    b["model"] = model
    return b

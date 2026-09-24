"""
03_shap_smoke_test.py
SMOKE TEST ONLY - NOT RESULTS. Checks that the tooling works end to end:
LSTM on the 6 model inputs -> Captum GradientShap -> attributions that add up -> channel, lag and
input-type summaries.

To protect the pre-registration, this script never explains a crisis or validation window:
it trains on 2003-2015, early-stops on 2016 and explains only calm days in 2017.
Single seed, fixed split, no tuning. Nothing here may be reported as a finding.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap, IntegratedGradients
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
torch.manual_seed(1)
np.random.seed(1)

INPUTS = ["ret", "r2", "us_ret", "us_r2", "euro_ret", "euro_r2"]
CHANNEL = {"ret": "domestic", "r2": "domestic", "us_ret": "US", "us_r2": "US", "euro_ret": "euro", "euro_r2": "euro"}
KIND = {k: ("size (squared return)" if k.endswith("r2") else "direction (return)") for k in INPUTS}
LOOKBACK = 22

# ---- data ------------------------------------------------------------------------
d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True)
d = d.loc[:"2023-08-31", INPUTS + ["log_rv5_fwd", "crisis"]].dropna()
for c in ("r2", "us_r2", "euro_r2"):                      # heavy-tailed: model the log of squared returns
    d[c] = np.log(d[c].clip(lower=1e-8))

train_end, val_end, explain = "2015-12-31", "2016-12-31", ("2017-01-01", "2017-12-31")
mu = d.loc[:train_end, INPUTS].mean()
sd = d.loc[:train_end, INPUTS].std()                     # scaler fitted on training data only
X_all = ((d[INPUTS] - mu) / sd).to_numpy(np.float32)
y_mu, y_sd = d.loc[:train_end, "log_rv5_fwd"].mean(), d.loc[:train_end, "log_rv5_fwd"].std()
y_all = ((d["log_rv5_fwd"] - y_mu) / y_sd).to_numpy(np.float32)

idx = np.arange(LOOKBACK - 1, len(d))                    # window ending at day t -> target at t
dates = d.index[idx]
X = np.stack([X_all[i - LOOKBACK + 1: i + 1] for i in idx])
y = y_all[idx]
tr = dates <= pd.Timestamp(train_end)
va = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))
ex = (dates >= pd.Timestamp(explain[0])) & (dates <= pd.Timestamp(explain[1])) & (d["crisis"].iloc[idx] == "calm").to_numpy()
# The 5-day target overlaps the next period, so drop the last 5 training days (no leakage into validation).
tr[np.where(tr)[0][-5:]] = False
Xt, yt, Xv, yv = map(torch.tensor, (X[tr], y[tr], X[va], y[va]))


class LSTMVol(nn.Module):
    def __init__(self, n_in=6, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


model = LSTMVol()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
best, best_state, patience = np.inf, None, 0
for epoch in range(200):
    model.train()
    perm = torch.randperm(len(Xt))
    for b in range(0, len(Xt), 64):
        j = perm[b:b + 64]
        opt.zero_grad()
        loss_fn(model(Xt[j]), yt[j]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        v = loss_fn(model(Xv), yv).item()
    if v < best - 1e-4:
        best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 15:
            break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pv = model(Xv).numpy()
naive = float(np.mean((yv.numpy() - yt.numpy().mean()) ** 2))
print(f"Stopped after {epoch + 1} epochs. Validation 2016 MSE (standardised log RV): LSTM {best:.3f} vs "
      f"predict-the-training-mean {naive:.3f}; corr(pred, actual) = {np.corrcoef(pv, yv.numpy())[0, 1]:.2f}")

# ---- GradientShap on calm 2017 days -----------------------------------------------
gs = GradientShap(model)
background = Xt[torch.randperm(len(Xt))[:200]]
Xe = torch.tensor(X[ex])
with torch.no_grad():
    f_x = model(Xe).numpy()
    f_base = model(background).numpy().mean()
# GradientShap (expected gradients) is additive only on average over random baselines, so the
# sampling error shrinks as n_samples grows. Check convergence before trusting the shares.
def gradient_shap(x, n, chunk=8):
    """Explain a few days at a time: n_samples x days x 22 x 6 gradients do not fit in memory at once."""
    return np.concatenate([gs.attribute(x[i:i + chunk], baselines=background, n_samples=n, stdevs=0.0)
                           .detach().numpy() for i in range(0, len(x), chunk)])


for n in (64, 256, 1024):
    torch.manual_seed(2)
    a_n = gradient_shap(Xe, n)
    gap = a_n.sum(axis=(1, 2)) - (f_x - f_base)
    print(f"GradientShap n_samples={n:4d}: mean |sum of attributions - (forecast - baseline)| = "
          f"{np.abs(gap).mean():.3f}; corr = {np.corrcoef(a_n.sum(axis=(1, 2)), f_x - f_base)[0, 1]:.3f}")
attr = a_n                                                        # (N, 22, 6), largest n_samples
# Exact cross-check: Integrated Gradients against one baseline (mean training window) is additive
# up to numerical integration error.
ig = IntegratedGradients(model)
ig_attr, delta = ig.attribute(Xe, baselines=Xt.mean(0, keepdim=True), n_steps=64,
                              return_convergence_delta=True)
print(f"Integrated Gradients: max additivity error = {delta.abs().max().item():.5f}")
print(f"Explained {len(Xe)} calm days in 2017 (forecast spread {f_x.std():.3f}).")

a = np.abs(attr)
tot = a.sum()
by_feat = pd.Series(a.sum(axis=(0, 1)) / tot, index=INPUTS)
by_channel = by_feat.groupby(CHANNEL).sum().reindex(["domestic", "US", "euro"])
by_kind = by_feat.groupby(KIND).sum()
by_lag = pd.Series(a.sum(axis=(0, 2)) / tot, index=[f"t-{LOOKBACK - 1 - k}" for k in range(LOOKBACK)])
print("\nChannel shares (calm 2017, smoke test):\n" + by_channel.round(3).to_string())
print("\nInput-type shares:\n" + by_kind.round(3).to_string())
print("\nMost important lags:\n" + by_lag.sort_values(ascending=False).head(5).round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
by_channel.plot.bar(ax=axes[0], color=["#1b1b1b", "#1f77b4", "#2ca02c"])
axes[0].set_title("By channel")
by_kind.plot.bar(ax=axes[1], color=["#9e9e9e", "#1f4e79"])
axes[1].set_title("By input type")
axes[1].tick_params(axis="x", labelrotation=0, labelsize=7)
axes[2].bar(range(LOOKBACK), by_lag.values[::-1], color="#1f4e79")
axes[2].set_xticks(range(0, LOOKBACK, 3))
axes[2].set_xticklabels([f"t-{k}" for k in range(0, LOOKBACK, 3)])
axes[2].set_title("By look-back lag (t-0 = most recent day)")
for ax in axes:
    ax.set_ylabel("share of |attribution|")
fig.suptitle("SMOKE TEST ONLY — calm 2017 days, single seed, not a result", x=0.01, ha="left", color="#d62728")
fig.tight_layout()
fig.savefig(FIG / "smoke_test_shap.png", dpi=140)
print(f"\nFigure: {FIG / 'smoke_test_shap.png'}")

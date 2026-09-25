"""
09_report_figures.py
Report figures built only from saved results (no model is re-run):
  figures/report_crisis_forecasts.png - LSTM vs GARCH vs HAR vs realised volatility in each window
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import HORIZON

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"

bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
fc = bench.join(lstm[["LSTM"]], how="inner")
ann = lambda s: np.sqrt(s * 252 / HORIZON)

panels = [("Global Financial Crisis", "2007-06-01", "2009-06-30"),
          ("Irish sovereign debt crisis", "2010-03-01", "2012-09-30"),
          ("COVID-19", "2020-01-01", "2020-09-30"),
          ("War / energy shock 2022", "2022-01-01", "2022-12-31")]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (title, a, b) in zip(axes.flat, panels):
    g = fc.loc[a:b]
    ax.plot(ann(g["rv5_fwd"]), color="#c9c9c9", lw=0.7, label="realised (next 5 days)")
    ax.plot(ann(g["HAR"]), color="#1f77b4", lw=1.0, label="HAR")
    ax.plot(ann(g["GARCH"]), color="#d62728", lw=1.0, label="GARCH(1,1)")
    ax.plot(ann(g["LSTM"]), color="#2ca02c", lw=1.4, label="LSTM (mean of 5 seeds)")
    w = g.index[g["window"] == title]
    if len(w):
        ax.axvspan(w[0], w[-1], color="#fff2cc", alpha=0.6, zorder=0)
    ax.set_title(title, loc="left", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(labelsize=8)
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 0].set_ylabel("annualised volatility")
axes[1, 0].set_ylabel("annualised volatility")
fig.suptitle("Out-of-sample forecasts of ISEQ volatility in each window (shaded = evaluation window)",
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "report_crisis_forecasts.png", dpi=150)
print("Saved figures/report_crisis_forecasts.png")

"""
02_handbook_figures.py
Original figures from the dissertation dataset that illustrate concepts from the
literature (Minsky, Cont, Forbes & Rigobon, Hamao et al., Corsi, event studies).

Run after 01_data_pipeline.py:  py -3.13 02_handbook_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
CRISES = {"GFC": ("2007-08-09", "2009-03-09"), "Irish debt crisis": ("2010-04-23", "2012-07-26"),
          "COVID-19": ("2020-02-19", "2020-06-30")}
COLS = {"ISEQ": "#1b1b1b", "S&P 500": "#1f77b4", "FTSE 100": "#d62728", "DAX": "#2ca02c"}


def close(name):
    return pd.read_csv(RAW / f"{name}.csv", index_col=0, parse_dates=True)["Close"].dropna().loc["2003-01-01":]


px = pd.DataFrame({"ISEQ": close("ISEQ"), "S&P 500": close("GSPC"), "FTSE 100": close("FTSE"),
                   "DAX": close("GDAXI")})
# Put every market on the Irish trading calendar, carrying forward the last known close.
px = px.ffill().loc[close("ISEQ").index]
ret = np.log(px).diff()
iseq_r = ret["ISEQ"].dropna()


def shade(ax):
    for (a, b), c in zip(CRISES.values(), ["#1f77b4", "#2ca02c", "#9467bd"]):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.12)


# 1. Event windows: how each shock reached Dublin -------------------------------
events = [("Lehman Brothers fails", "2008-09-15"), ("Irish bank guarantee", "2008-09-30"),
          ("EU/IMF programme for Ireland", "2010-11-29"), ("Brexit vote result", "2016-06-24"),
          ("COVID-19 sell-off", "2020-03-09"), ("Russia invades Ukraine", "2022-02-24")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
for ax, (title, day) in zip(axes.flat, events):
    d = pd.Timestamp(day)
    idx = px.index
    i0 = idx.get_indexer([d], method="bfill")[0]
    win = px.iloc[max(i0 - 10, 0): i0 + 21]
    base = px.iloc[i0 - 1]
    norm = win / base * 100
    x = np.arange(len(win)) - (i0 - max(i0 - 10, 0))
    for col, colr in COLS.items():
        ax.plot(x, norm[col].ffill().values, color=colr, lw=1.6 if col == "ISEQ" else 1.1, label=col)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.axhline(100, color="grey", lw=0.5)
    ax.set_title(f"{title}\n({idx[i0].date()})", fontsize=9.5)
    ax.set_xlabel("trading days from event", fontsize=8)
    ax.tick_params(labelsize=8)
axes[0, 0].set_ylabel("index level (day before event = 100)", fontsize=8)
axes[1, 0].set_ylabel("index level (day before event = 100)", fontsize=8)
axes[0, 0].legend(fontsize=7.5, frameon=False)
fig.suptitle("How six shocks reached the Irish market (your data, Yahoo Finance closes)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "hb_event_windows.png", dpi=150)
plt.close(fig)

# 2. Rolling correlations: interdependence rises in crises ---------------------
# ISEQ day t is matched with the US close of day t-1 (New York closes after Dublin).
pairs = {"S&P 500 (previous US day)": ret["S&P 500"].shift(1), "FTSE 100 (same day)": ret["FTSE 100"],
         "DAX (same day)": ret["DAX"]}
fig, ax = plt.subplots(figsize=(13, 4.2))
for (lab, s), colr in zip(pairs.items(), ["#1f77b4", "#d62728", "#2ca02c"]):
    ax.plot(ret["ISEQ"].rolling(250, min_periods=200).corr(s), color=colr, lw=1.1, label=lab)
shade(ax)
ax.set_ylabel("250-day correlation with ISEQ")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Links between Dublin and foreign markets over time (crisis windows shaded)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "hb_rolling_correlation.png", dpi=150)
plt.close(fig)

# 3. Stylised facts: returns vs squared returns autocorrelation (Cont, 2001) -----
lags = np.arange(1, 61)
acf_r = [iseq_r.autocorr(k) for k in lags]
acf_r2 = [(iseq_r ** 2).autocorr(k) for k in lags]
band = 1.96 / np.sqrt(len(iseq_r))
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(lags - 0.2, acf_r, width=0.4, color="#9e9e9e", label="daily returns")
ax.bar(lags + 0.2, acf_r2, width=0.4, color="#1f4e79", label="squared returns (volatility)")
ax.axhspan(-band, band, color="#f4cccc", alpha=0.6, label="95% band for no autocorrelation")
ax.set_xlabel("lag (trading days)")
ax.set_ylabel("autocorrelation")
ax.legend(frameon=False, fontsize=8)
ax.set_title("ISEQ 2003-2025: returns are close to unpredictable, volatility is persistent", loc="left")
fig.tight_layout()
fig.savefig(FIG / "hb_volatility_clustering.png", dpi=150)
plt.close(fig)

# 4. HAR components around the GFC (Corsi, 2009) -------------------------------
r2 = iseq_r ** 2
har = pd.DataFrame({"daily": np.sqrt(r2 * 252), "weekly (5-day)": np.sqrt(r2.rolling(5).mean() * 252),
                    "monthly (22-day)": np.sqrt(r2.rolling(22).mean() * 252)}).loc["2008-01-01":"2009-06-30"]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(har["daily"], color="#c9c9c9", lw=0.7, label="daily")
ax.plot(har["weekly (5-day)"], color="#1f77b4", lw=1.3, label="weekly (5-day average)")
ax.plot(har["monthly (22-day)"], color="#d62728", lw=1.8, label="monthly (22-day average)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0, 2.0)
ax.set_ylabel("annualised volatility")
ax.legend(frameon=False, fontsize=8)
ax.set_title("The three HAR ingredients for the ISEQ during the GFC (2008 - mid-2009)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "hb_har_components.png", dpi=150)
plt.close(fig)

# 5. Minsky: calm boom, then crash ----------------------------------------------
vol1y = np.sqrt(r2.rolling(250).mean() * 252).loc[:"2010-12-31"]
lvl = px["ISEQ"].loc[:"2010-12-31"]
fig, ax1 = plt.subplots(figsize=(10, 3.8))
ax1.plot(lvl, color="#1b1b1b", lw=1.1)
ax1.set_ylabel("ISEQ level")
ax2 = ax1.twinx()
ax2.plot(vol1y, color="#d62728", lw=1.3)
ax2.set_ylabel("1-year realised volatility", color="#d62728")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.axvspan(pd.Timestamp("2003-01-01"), pd.Timestamp("2007-02-20"), color="#fff2cc", alpha=0.6)
ax1.text(pd.Timestamp("2003-06-01"), lvl.max() * 0.93, "calm boom: low volatility\n(your pre-GFC training data)",
         fontsize=8)
ax1.set_title("Minsky in the data: stability before the storm (ISEQ 2003-2010)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "hb_minsky_boom.png", dpi=150)
plt.close(fig)

# 6. Trading hours and information flow (Hamao et al.; Engle et al.) ------------
fig, ax = plt.subplots(figsize=(11, 3.0))
sessions = [("Tokyo (day t)", 0, 6, "#9467bd"), ("Dublin / London (day t)", 8, 16.5, "#1b1b1b"),
            ("Frankfurt (day t)", 8, 16.5, "#2ca02c"), ("New York (day t)", 14.5, 21, "#1f77b4")]
for i, (name, a, b, c) in enumerate(sessions):
    ax.barh(i, b - a, left=a, color=c, alpha=0.75)
    ax.text(a + 0.15, i, name, va="center", color="white", fontsize=8, fontweight="bold")
ax.barh(1, 2, left=32, color="#1b1b1b", alpha=0.35)
ax.text(32.1, 1, "Dublin\nt+1", va="center", fontsize=7)
ax.axvline(31.9, color="#d62728", lw=1.4)
ax.text(31.7, 3.45, "08:00 day t+1: forecast issued\nusing every close up to day t", ha="right", va="top",
        fontsize=8, color="#d62728")
ax.set_yticks([])
ax.set_xlim(0, 34)
ax.set_ylim(-0.6, 3.6)
ax.set_xticks(range(0, 34, 3))
ax.set_xticklabels([f"{h % 24:02d}:00" for h in range(0, 34, 3)], fontsize=8)
ax.set_xlabel("Irish time, from midnight of trading day t to the next morning (approximate)")
ax.set_title("Why New York’s day-t close is known before Dublin’s day t+1 opens", loc="left")
fig.tight_layout()
fig.savefig(FIG / "hb_trading_hours.png", dpi=150)
plt.close(fig)
print("figures written to", FIG)

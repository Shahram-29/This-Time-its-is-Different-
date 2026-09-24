"""
01_data_pipeline.py
Dissertation: "This Time Is Different?" - explainable deep learning of crisis
transmission into Irish equity volatility: the Global Financial Crisis, the Irish
sovereign debt crisis and COVID-19 (sample January 2003 - August 2023).

Step 1 of the pipeline:
  1. Download (and cache) daily data for the target and all feature markets.
  2. Align everything to the Irish (ISEQ) trading calendar without look-ahead.
  3. Build the target: forward 5-day realised variance from close-to-close returns.
  4. Build HAR regressors and foreign-market features.
  5. Write a data-quality report and the crisis chart.

Run:  py -3.13 01_data_pipeline.py
Needs: pandas, numpy, yfinance, matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
for p in (RAW, PROCESSED, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

# Sample starts after the dot-com crash ended (Nasdaq trough 9 Oct 2002). The download
# begins earlier only so the 22-day HAR averages are complete on the first sample day.
START, END = "2002-10-01", "2025-12-31"   # fixed end date = reproducible sample
SAMPLE_START = "2003-01-01"

TARGET = "^ISEQ"
# Three markets, three transmission channels: domestic (ISEQ), US (S&P 500), euro area (DAX).
# Nasdaq (0.95 with S&P 500), VIX (-0.73 with S&P 500) and the US 10-year yield (0.05 with
# ISEQ) were dropped as redundant or uninformative. FTSE overlaps DAX at 0.84, so it is kept
# only for the robustness run that swaps it in for DAX (and for the Brexit check).
FEATURES = {
    "^GSPC":  ("us",   "S&P 500"),
    "^GDAXI": ("euro", "DAX (euro-area core; Euro Stoxx 50 only on Yahoo from 2007)"),
    "^FTSE":  ("uk",   "FTSE 100 (robustness only - not a main model input)"),
}
MODEL_INPUTS = ["ret", "r2", "us_ret", "us_r2", "euro_ret", "euro_r2"]
ROBUSTNESS_INPUTS = ["ret", "r2", "us_ret", "us_r2", "uk_ret", "uk_r2"]   # FTSE in place of DAX

# Evaluation windows only. These are defined ex post and must NEVER be model inputs.
# Core crises: three different origins (US credit, Irish/euro sovereign-bank, exogenous global).
CRISES = {
    "Global Financial Crisis": ("2007-08-09", "2009-03-09"),  # BNP Paribas fund freeze -> S&P 500 trough
    "Irish sovereign debt crisis": ("2010-04-23", "2012-07-26"),  # Greece requests aid -> Draghi "whatever it takes"
    "COVID-19":                ("2020-02-19", "2020-06-30"),  # S&P 500 pre-crash peak -> end of Q2 2020
}
# Single-origin validation events: each has one obvious transmission channel.
VALIDATION = {
    "Brexit referendum":       ("2016-06-01", "2016-07-31"),  # UK channel expected (FTSE robustness run only)
    "Ukraine war / rate shock": ("2022-02-10", "2022-10-31"),  # euro channel expected
}
ISEQ_BREAK = "2023-09-01"  # CRH (Sep 2023), Flutter (Jan 2024), Smurfit (Jul 2024) leave the ISEQ


def load(ticker: str) -> pd.DataFrame:
    """Download once, then read from the local cache (keeps the study reproducible)."""
    cache = RAW / f"{ticker.replace('^', '')}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df.index = pd.to_datetime(df.index.date)          # drop exchange time zone, keep trading date
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(cache)
    return df


def build():
    report = []
    iseq = load(TARGET).dropna(subset=["Close"])
    iseq = iseq[iseq["Close"] > 0]

    # ---- Target: forward 5-day realised variance (close-to-close only) ----------
    r = np.log(iseq["Close"]).diff()
    r2 = r ** 2
    data = pd.DataFrame(index=iseq.index)
    data["ret"] = r
    data["r2"] = r2
    # Forecast issued before Dublin open on day t+1, targeting days t+1..t+5.
    data["rv5_fwd"] = sum(r2.shift(-k) for k in range(1, 6))
    data["log_rv5_fwd"] = np.log(data["rv5_fwd"])

    # ---- HAR regressors (information up to and including day t) ---------------
    data["rv_d"] = r2
    data["rv_w"] = r2.rolling(5).mean()
    data["rv_m"] = r2.rolling(22).mean()

    # ---- Foreign features: as-of join, backward, on calendar date --------------
    # US and European closes on day t happen before a forecast issued at Dublin
    # open on t+1, so values dated <= t are legitimate. Nothing dated t+1 is used.
    for ticker, (group, name) in FEATURES.items():
        f = load(ticker).dropna(subset=["Close"])
        f = f[f["Close"] > 0]
        fr = np.log(f["Close"]).diff()
        feat = pd.DataFrame({f"{group}_ret": fr, f"{group}_r2": fr ** 2}).sort_index()
        data = pd.merge_asof(data.sort_index(), feat, left_index=True, right_index=True,
                             direction="backward")
        stale = (~data.index.isin(f.index)).mean()
        report.append(f"{ticker:7s} {name:60s} first={f.index.min().date()}  "
                      f"rows={len(f):5d}  Irish days carried forward={stale:.1%}")

    c = data.loc[SAMPLE_START:"2023-08-31", ["ret", "us_ret", "euro_ret", "uk_ret"]].corr()
    report.append(f"Daily-return correlations 2003-2023: ISEQ-US {c.loc['ret', 'us_ret']:.2f}, "
                  f"ISEQ-euro {c.loc['ret', 'euro_ret']:.2f}, ISEQ-UK {c.loc['ret', 'uk_ret']:.2f}, "
                  f"euro-UK {c.loc['euro_ret', 'uk_ret']:.2f}")
    report.append(f"Main model inputs: {MODEL_INPUTS}")
    report.append(f"Robustness inputs (FTSE replaces DAX): {ROBUSTNESS_INPUTS}")

    data = data.loc[SAMPLE_START:]
    iseq = iseq.loc[SAMPLE_START:]
    r2 = r2.loc[SAMPLE_START:]
    data["crisis"] = "calm"
    for label, (a, b) in {**CRISES, **VALIDATION}.items():
        data.loc[a:b, "crisis"] = label   # evaluation label only - never a feature

    # ---- Data-quality checks that justify methodological choices ---------------
    o, h, l, c = (iseq[k] for k in ("Open", "High", "Low", "Close"))
    stale_open = (o - c.shift(1)).abs() < 1e-9
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    park = np.log(h / l) ** 2 / (4 * np.log(2))
    q = pd.DataFrame({"stale": stale_open, "gk": gk, "park": park, "r2": r2}).dropna()
    report.append("\nWhy Garman-Klass is rejected (Yahoo ^ISEQ open is often yesterday's close):")
    for a, b in [("2003", "2006"), ("2007", "2012"), ("2013", "2019"), ("2020", "2025")]:
        s = q.loc[a:b]
        m = s.resample("ME").mean()
        report.append(f"  {a}-{b}: stale opens={s['stale'].mean():5.1%}  "
                      f"monthly corr(GK, r^2)={m['gk'].corr(m['r2']):.2f}  "
                      f"monthly corr(Parkinson, r^2)={m['park'].corr(m['r2']):.2f}")

    report.append("\nISEQ by evaluation window (target = forward 5-day realised variance):")
    ann_vol = np.sqrt(data["rv5_fwd"] * 252 / 5)
    calm_vol = ann_vol[data["crisis"] == "calm"].median()
    report.append(f"  calm periods: median annualised vol {calm_vol:.1%}")
    for label, (a, b) in {**CRISES, **VALIDATION}.items():
        px = iseq.loc[a:b, "Close"]
        dd = (px / px.cummax() - 1).min()
        v = ann_vol.loc[a:b]
        report.append(f"  {label:30s} days={len(px):4d}  max drawdown={dd:6.1%}  median vol={v.median():5.1%}  "
                      f"peak vol={v.max():6.1%}  days above 2x calm vol={(v > 2 * calm_vol).mean():5.1%}")

    pre = data.loc[:"2007-08-08"].dropna(subset=["rv5_fwd"])
    report.append(f"\nTraining history before the GFC window: {len(pre)} days ({len(pre) / 252:.1f} years), "
                  f"all from the calm 2003-07 boom.")
    main = data.loc[:"2023-08-31"].dropna()
    report.append(f"Main sample (Jan 2003 - Aug 2023), complete rows: {len(main)}; "
                  f"zero-return days: {(main['r2'] == 0).sum()}")
    post = data.loc[ISEQ_BREAK:]
    report.append(f"Days after ISEQ composition break ({ISEQ_BREAK}): {len(post)} "
                  f"-> robustness only, not in the main sample.")

    data.to_csv(PROCESSED / "dataset.csv")
    (ROOT / "data" / "data_quality_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    plot(iseq, ann_vol)


def plot(iseq, ann_vol):
    colours = ["#1f77b4", "#2ca02c", "#9467bd"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    ax1.plot(iseq.index, iseq["Close"], color="#333", lw=0.9)
    ax1.set_yscale("log")
    ax1.set_ylabel("ISEQ Overall (log scale)")
    ax2.plot(ann_vol.index, ann_vol, color="#333", lw=0.6)
    ax2.set_ylabel("Forward 5-day realised vol\n(annualised)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for ax in (ax1, ax2):
        first = ax is ax1
        for (label, (a, b)), col in zip(CRISES.items(), colours):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.18,
                       label=label if first else None)
        for i, (label, (a, b)) in enumerate(VALIDATION.items()):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), facecolor="none", edgecolor="#d62728",
                       hatch="///", lw=0.0, label=("Validation events (Brexit, 2022)" if first and i == 0 else None))
        ax.axvline(pd.Timestamp(ISEQ_BREAK), color="grey", ls="--", lw=0.8)
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax2.text(pd.Timestamp(ISEQ_BREAK), ax2.get_ylim()[1] * 0.9, " ISEQ composition\n break (CRH exit)",
             fontsize=7, color="grey", va="top")
    ax1.set_title("Irish equity market, 2003-2025: three crises of different origin, two validation events",
                  loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "crisis_chart.png", dpi=160)


if __name__ == "__main__":
    build()

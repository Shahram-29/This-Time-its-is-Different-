"""
evaluation.py
Shared definitions for the whole study: evaluation windows, loss functions and the
Diebold-Mariano test with the Harvey-Leybourne-Newbold correction. Imported by 04+ scripts
so every model is judged in exactly the same way.
"""

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_END = "2023-08-31"
TEST_YEARS = range(2007, 2024)          # walk-forward test years (2023 ends on SAMPLE_END)
HORIZON = 5                             # target = realised variance over the next 5 trading days

WINDOWS = {                              # pre-registered evaluation windows (never model inputs)
    "Global Financial Crisis": ("2007-08-09", "2009-03-09"),
    "Irish sovereign debt crisis": ("2010-04-23", "2012-07-26"),
    "COVID-19": ("2020-02-19", "2020-06-30"),
    "War / energy shock 2022": ("2022-02-10", "2022-10-31"),
    "Brexit referendum": ("2016-06-01", "2016-07-31"),
}


def window_of(dates: pd.DatetimeIndex) -> pd.Series:
    """Label each date with its evaluation window ('calm' if none)."""
    lab = pd.Series("calm", index=dates)
    for name, (a, b) in WINDOWS.items():
        lab[(dates >= a) & (dates <= b)] = name
    return lab


def qlike(realised, forecast):
    """Patton (2011) QLIKE: RV/F - ln(RV/F) - 1. Zero for a perfect forecast; punishes under-prediction more."""
    ratio = np.asarray(realised) / np.asarray(forecast)
    return ratio - np.log(ratio) - 1


def mse(realised, forecast):
    return (np.asarray(realised) - np.asarray(forecast)) ** 2


def dm_hln(loss_a, loss_b, h=HORIZON):
    """
    Diebold-Mariano test of equal expected loss, with the Harvey-Leybourne-Newbold (1997)
    small-sample correction and Student-t critical values.
    d = loss_a - loss_b; a negative mean means model A is more accurate.
    Overlapping h-step forecasts: long-run variance uses autocovariances up to lag h-1.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:                                        # fall back to Bartlett weights if needed
        lrv = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    dm = dbar / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"n": n, "mean_diff": dbar, "stat": hln, "p_value": p}


def loss_table(df: pd.DataFrame, models, realised="rv5_fwd") -> pd.DataFrame:
    """Mean QLIKE and MSE per model, for the full test sample and each window."""
    rows = []
    groups = [("All test days (2007 - Aug 2023)", df)] + [(w, g) for w, g in df.groupby("window")]
    for name, g in groups:
        row = {"window": name, "days": len(g)}
        for m in models:
            row[f"QLIKE {m}"] = qlike(g[realised], g[m]).mean()
            row[f"MSE {m} (x1e6)"] = mse(g[realised], g[m]).mean() * 1e6
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")

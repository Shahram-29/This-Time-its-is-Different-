"""
04_benchmarks.py
Benchmark forecasts of ISEQ volatility: HAR (log) and GARCH(1,1) with Student-t errors,
walk-forward with annual refits on an expanding window (pre-registered design).

For each test year Y in 2007-2023:
  * models are estimated on data up to 31 Dec of Y-1;
  * the last 5 training days are dropped from HAR estimation, because their 5-day targets
    reach into year Y (no leakage);
  * parameters stay fixed through year Y while forecasts use data up to each day t.
Target: rv5_fwd(t) = r^2(t+1) + ... + r^2(t+5).

A naive forecast (last week's realised variance) is reported as a reference only.
Outputs: results/benchmark_forecasts.csv, benchmark_losses.csv, benchmark_parameters.csv,
figures/benchmark_forecasts.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from evaluation import SAMPLE_END, TEST_YEARS, HORIZON, WINDOWS, window_of, loss_table, dm_hln, qlike

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True)
d = d.loc[:SAMPLE_END]
FLOOR = 1e-8
har_X = np.log(d[["rv_d", "rv_w", "rv_m"]].clip(lower=FLOOR))
har_y = np.log(d["rv5_fwd"])
returns = d["ret"].dropna() * 100                     # percent returns for numerical stability

forecasts, params = [], []
for year in TEST_YEARS:
    start = pd.Timestamp(f"{year}-01-01")
    end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(SAMPLE_END))
    test = d.loc[start:end].dropna(subset=["rv5_fwd"]).index

    # ---- HAR (log), OLS on the expanding window ----------------------------------------
    tr = d.loc[:start - pd.Timedelta(days=1)].dropna(subset=["rv5_fwd", "rv_m"]).index[:-HORIZON]
    X = sm.add_constant(har_X.loc[tr])
    fit = sm.OLS(har_y.loc[tr], X).fit()
    smear = np.mean(np.exp(fit.resid))               # Duan smearing: unbiased back-transformation
    har = np.exp(fit.predict(sm.add_constant(har_X.loc[test], has_constant="add"))) * smear

    # ---- GARCH(1,1), Student-t, constant mean --------------------------------------------
    am = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(last_obs=start, disp="off")
    fc = res.forecast(horizon=HORIZON, start=test[0], reindex=False)
    garch = fc.variance.loc[test].sum(axis=1) / 1e4    # sum of 1..5-step variances, back to decimal units

    # ---- naive reference: last week's realised variance ---------------------------------
    naive = d.loc[test, "rv_w"] * HORIZON

    forecasts.append(pd.DataFrame({"rv5_fwd": d.loc[test, "rv5_fwd"], "HAR": har, "GARCH": garch,
                                   "Naive": naive}))
    p = res.params
    params.append({"test_year": year, "train_days": len(tr),
                   "HAR_const": fit.params["const"], "HAR_daily": fit.params["rv_d"],
                   "HAR_weekly": fit.params["rv_w"], "HAR_monthly": fit.params["rv_m"],
                   "HAR_R2": fit.rsquared, "GARCH_omega": p["omega"], "GARCH_alpha": p["alpha[1]"],
                   "GARCH_beta": p["beta[1]"], "GARCH_persistence": p["alpha[1]"] + p["beta[1]"],
                   "GARCH_nu": p["nu"]})
    print(f"{year}: {len(test):3d} test days | HAR R2 {fit.rsquared:.2f} | GARCH alpha+beta "
          f"{p['alpha[1]'] + p['beta[1]']:.3f}, nu {p['nu']:.1f}")

fc = pd.concat(forecasts)
fc["window"] = window_of(fc.index)
fc.to_csv(RES / "benchmark_forecasts.csv")
pd.DataFrame(params).set_index("test_year").to_csv(RES / "benchmark_parameters.csv")

losses = loss_table(fc, ["HAR", "GARCH", "Naive"])
order = ["All test days (2007 - Aug 2023)", "calm", *WINDOWS]
losses = losses.reindex([w for w in order if w in losses.index])
losses.to_csv(RES / "benchmark_losses.csv")
pd.set_option("display.width", 200)
print("\nMean losses (lower is better):\n" + losses.round(3).to_string())

t = dm_hln(qlike(fc["rv5_fwd"], fc["HAR"]), qlike(fc["rv5_fwd"], fc["GARCH"]))
print(f"\nDM-HLN, QLIKE, HAR vs GARCH (all test days): mean difference {t['mean_diff']:.4f}, "
      f"stat {t['stat']:.2f}, p = {t['p_value']:.4f}  (negative = HAR more accurate)")

# ---- figure: annualised forecast vs realised volatility --------------------------------
ann = lambda s: np.sqrt(s * 252 / HORIZON)
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(ann(fc["rv5_fwd"]), color="#c9c9c9", lw=0.6, label="realised (next 5 days)")
ax.plot(ann(fc["HAR"]), color="#1f77b4", lw=1.0, label="HAR forecast")
ax.plot(ann(fc["GARCH"]), color="#d62728", lw=1.0, label="GARCH(1,1) forecast")
for (a, b), c in zip([WINDOWS[k] for k in list(WINDOWS)[:3]], ["#1f77b4", "#2ca02c", "#9467bd"]):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.08)
ax.set_ylim(0, 1.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylabel("annualised volatility")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Benchmark forecasts of ISEQ volatility, walk-forward 2007 - Aug 2023", loc="left")
fig.tight_layout()
fig.savefig(FIG / "benchmark_forecasts.png", dpi=150)
print(f"\nSaved: {RES / 'benchmark_forecasts.csv'}, benchmark_losses.csv, benchmark_parameters.csv, "
      f"{FIG / 'benchmark_forecasts.png'}")

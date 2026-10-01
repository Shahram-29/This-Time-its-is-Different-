"""
13_ml_benchmarks.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A2, E3)
Classical machine-learning benchmarks from the MSc module: Random Forest and Support Vector Regression, tuned with
GridSearchCV, forecasting the same target on the same days as HAR, GARCH and the LSTM.

Specification (fixed in Amendment A2 before this script was run on any test year):
  * inputs (9): logs of daily, weekly (5-day mean) and monthly (22-day mean) squared returns of ISEQ, S&P 500, DAX;
    target: log forward 5-day realised variance;
  * walk-forward as the LSTM: for test year Y, fitting days to 31 Dec Y-2, validation year Y-1 (smearing only),
    the last 5 days of each part dropped; SVR inputs standardised on the fitting days;
  * tuning once on the first window (fitting days 2003-2005) with GridSearchCV, TimeSeriesSplit(5, gap=5),
    negative MSE; settings then frozen;
  * evaluation: QLIKE and MSE by window, DM-HLN (QLIKE) against GARCH, HAR and the LSTM on all test days.
Run from the code folder:  py -3.13 13_ml_benchmarks.py   (about 2 minutes)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from evaluation import SAMPLE_END, TEST_YEARS, HORIZON, window_of, qlike, loss_table, dm_hln

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
COLUMNS = ["rv_d", "rv_w", "rv_m", "us_r2", "us_rv_w", "us_rv_m", "euro_r2", "euro_rv_w", "euro_rv_m"]
CHANNEL = {"rv_d": "domestic", "rv_w": "domestic", "rv_m": "domestic", "us_r2": "US", "us_rv_w": "US",
           "us_rv_m": "US", "euro_r2": "euro", "euro_rv_w": "euro", "euro_rv_m": "euro"}
RF_GRID = {"n_estimators": [200, 500], "max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20, 50]}
SVR_GRID = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "epsilon": [0.05, 0.1, 0.3], "gamma": ["scale", 0.01]}
FOLDS = TimeSeriesSplit(n_splits=5, gap=HORIZON)


def load():
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True).loc[:SAMPLE_END]
    for m in ["us", "euro"]:
        d[f"{m}_rv_w"] = d[f"{m}_r2"].rolling(5).mean()
        d[f"{m}_rv_m"] = d[f"{m}_r2"].rolling(22).mean()
    x = np.log(d[COLUMNS].clip(lower=1e-8))
    y = np.log(d["rv5_fwd"])
    ok = d.dropna(subset=["rv5_fwd", "rv_m", "us_rv_m", "euro_rv_m"]).index
    return d, x, y, ok


def days_for(d, ok, year):
    fit = ok[ok <= f"{year - 2}-12-31"][:-HORIZON]
    val = ok[(ok >= f"{year - 1}-01-01") & (ok <= f"{year - 1}-12-31")][:-HORIZON]
    test = d.loc[f"{year}-01-01":f"{year}-12-31"].dropna(subset=["rv5_fwd"]).index
    return fit, val, test


def tune(x, y, fit):
    scaled = pd.DataFrame(StandardScaler().fit(x.loc[fit]).transform(x), index=x.index, columns=COLUMNS)
    rf = GridSearchCV(RandomForestRegressor(max_features="sqrt", random_state=100, n_jobs=-1), RF_GRID,
                      scoring="neg_mean_squared_error", cv=FOLDS).fit(x.loc[fit], y.loc[fit])
    sv = GridSearchCV(SVR(), SVR_GRID, scoring="neg_mean_squared_error", cv=FOLDS).fit(scaled.loc[fit], y.loc[fit])
    rows = []
    for name, g in [("RF", rf), ("SVR", sv)]:
        for p, s in zip(g.cv_results_["params"], g.cv_results_["mean_test_score"]):
            rows.append({"model": name, **{k: str(v) for k, v in p.items()}, "mean_cv_mse": -s})
    return rf.best_params_, sv.best_params_, pd.DataFrame(rows)


def main():
    d, x, y, ok = load()
    fit, _, _ = days_for(d, ok, TEST_YEARS[0])
    rf_best, svr_best, tuning = tune(x, y, fit)
    tuning.to_csv(RES / "ml_tuning.csv", index=False)
    print(f"Tuned on {len(fit)} fitting days ({fit[0].date()} - {fit[-1].date()}):")
    print("  Random Forest:", rf_best)
    print("  SVR          :", svr_best)

    fc, imp, log = [], [], []
    for year in TEST_YEARS:
        fit, val, test = days_for(d, ok, year)
        rf = RandomForestRegressor(max_features="sqrt", random_state=100, n_jobs=-1, **rf_best)
        rf.fit(x.loc[fit], y.loc[fit])
        sc = StandardScaler().fit(x.loc[fit])
        xs = pd.DataFrame(sc.transform(x), index=x.index, columns=COLUMNS)
        sv = SVR(**svr_best).fit(xs.loc[fit], y.loc[fit])
        out = {"rv5_fwd": d.loc[test, "rv5_fwd"]}
        for name, model, xx in [("RF", rf, x), ("SVR", sv, xs)]:
            smear = float(np.mean(np.exp(y.loc[val] - model.predict(xx.loc[val]))))
            out[name] = pd.Series(np.exp(model.predict(xx.loc[test])) * smear, index=test)
            log.append({"test_year": year, "model": name, "n_fit": len(fit), "smear": smear,
                        "max_forecast": out[name].max(), "max_actual": d.loc[test, "rv5_fwd"].max()})
        fc.append(pd.DataFrame(out))
        imp.append(pd.Series(rf.feature_importances_, index=COLUMNS).groupby(CHANNEL).sum().rename(year))
        print(f"  {year}: {len(fit)} fitting days, smearing RF {log[-2]['smear']:.2f} / SVR {log[-1]['smear']:.2f}")

    fc = pd.concat(fc)
    fc["window"] = window_of(fc.index)
    fc.to_csv(RES / "ml_forecasts.csv")
    pd.DataFrame(log).to_csv(RES / "ml_training_log.csv", index=False)
    pd.DataFrame(imp).round(4).to_csv(RES / "ml_importance_by_channel.csv", index_label="test_year")

    # compare with the saved forecasts of the main study, on identical days
    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    allm = fc[["rv5_fwd", "RF", "SVR", "window"]].join(bench[["GARCH", "HAR"]]).join(lstm[["LSTM"]])
    assert allm.notna().all().all() and len(allm) == len(lstm), "forecast days do not line up"
    models = ["LSTM", "GARCH", "HAR", "RF", "SVR"]
    losses = loss_table(allm, models)
    losses.to_csv(RES / "ml_losses.csv")
    tests = []
    for a in ["RF", "SVR"]:
        for b in ["GARCH", "HAR", "LSTM"]:
            t = dm_hln(qlike(allm.rv5_fwd, allm[a]), qlike(allm.rv5_fwd, allm[b]))
            tests.append({"comparison": f"{a} vs {b}", **t})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "ml_dm_tests.csv", index=False)

    q = losses[[f"QLIKE {m}" for m in models]].round(3)
    q.columns = models
    print("\nQLIKE (lower is better):")
    print(q.to_string())
    print("\nDM-HLN, QLIKE, all test days (negative = first model better):")
    print(tests.round(4).to_string(index=False))
    gfc = q.loc["Global Financial Crisis"]
    print("\nExpectation (i)  RF worse than LSTM in the GFC:", "met" if gfc.RF > gfc.LSTM else "NOT met")
    a = q.loc["All test days (2007 - Aug 2023)"]
    print("Expectation (ii) neither RF nor SVR beats GARCH on all days:",
          "met" if min(a.RF, a.SVR) >= a.GARCH else "NOT met")

    # figure: the GFC year, all five models
    w = allm.loc["2008-01-01":"2009-06-30"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(w.index, w.rv5_fwd * 1e4, color="lightgrey", lw=1, label="Realised (next 5 days)")
    for m, c in [("GARCH", "tab:blue"), ("HAR", "tab:green"), ("LSTM", "tab:red"), ("RF", "tab:orange"),
                 ("SVR", "tab:purple")]:
        ax.plot(w.index, w[m] * 1e4, color=c, lw=1.3, label=m)
    ax.set_ylabel("5-day variance (x 10^4)")
    ax.set_title("Forecasts through the Global Financial Crisis, 2008 - mid 2009 (exploratory, Amendment A2)")
    ax.legend(ncol=6, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "ml_benchmarks_gfc.png", dpi=150)
    print("\nSaved results/ml_*.csv and figures/ml_benchmarks_gfc.png")


if __name__ == "__main__":
    main()

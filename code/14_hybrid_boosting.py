"""
14_hybrid_boosting.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A3, E4)
Does any of these do better than GARCH, and does the "unseen crisis" weakness of the learners remain?
  1. XGBoost with the 9 inputs of E3;
  2. GARCH hybrids: Random Forest, SVR and XGBoost learn only the correction log(RV5) - log(GARCH forecast);
  3. extra inputs: log VIX and the leverage terms of Corsi & Reno (2012), in HAR-X (OLS) and in the three learners;
  4. the equal-weight average of the saved GARCH and LSTM forecasts.
Walk-forward, tuning and smearing rules as in E3 (13_ml_benchmarks.py); HAR-X follows the main HAR (04).
Primary comparison: each hybrid vs GARCH, all test days, QLIKE, DM-HLN with a Holm adjustment over three tests.
Run from the code folder:  py -3.13 14_hybrid_boosting.py   (about 10 minutes; downloads the VIX once)
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from evaluation import TEST_YEARS, HORIZON, window_of, qlike, loss_table, dm_hln

ROOT = Path(__file__).resolve().parent.parent
RAW, RES, FIG = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
_spec = importlib.util.spec_from_file_location("e3", Path(__file__).parent / "13_ml_benchmarks.py")
e3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e3)                     # reuse E3's data, day splits, grids and folds

XGB_GRID = {"n_estimators": [200, 500], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 10]}
EXTRA = ["vix", "lev_d", "lev_w", "lev_m"]


def learner(kind, params=None):
    params = params or {}
    if kind == "RF":
        return RandomForestRegressor(max_features="sqrt", random_state=100, n_jobs=-1, **params)
    if kind == "SVR":
        return SVR(**params)
    return XGBRegressor(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8, random_state=100,
                        n_jobs=4, **params)


GRIDS = {"RF": e3.RF_GRID, "SVR": e3.SVR_GRID, "XGB": XGB_GRID}


def load_vix(index):
    cache = RAW / "VIX.csv"
    if not cache.exists():
        v = yf.Ticker("^VIX").history(start="2002-10-01", end="2025-12-31", auto_adjust=False)
        v.index = pd.to_datetime(v.index.date)            # trading date, as in 01_data_pipeline.py
        v[["Open", "High", "Low", "Close", "Volume"]].to_csv(cache)
    v = pd.read_csv(cache, index_col=0, parse_dates=True)["Close"].dropna()
    v = v[v > 0].rename("vix_close").sort_index()
    # backward as-of join on calendar date, like the S&P 500 and DAX series
    out = pd.merge_asof(pd.DataFrame(index=index).sort_index(), v.to_frame(), left_index=True, right_index=True,
                        direction="backward")
    return out["vix_close"]


def garch_path(d, year):
    """The main study's GARCH(1,1)-t for test year `year` (data to 31 Dec of year-1); 5-day forecasts for every day."""
    returns = d["ret"].dropna() * 100
    am = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(last_obs=pd.Timestamp(f"{year}-01-01"), disp="off")
    fc = res.forecast(horizon=HORIZON, start=returns.index[0], reindex=False)
    return fc.variance.sum(axis=1) / 1e4


def fit_predict(kind, params, X, target, fit, val, test, scale):
    if scale:
        sc = StandardScaler().fit(X.loc[fit])
        X = pd.DataFrame(sc.transform(X), index=X.index, columns=X.columns)
    m = learner(kind, params).fit(X.loc[fit], target.loc[fit])
    smear = float(np.mean(np.exp(target.loc[val] - m.predict(X.loc[val]))))
    return pd.Series(m.predict(X.loc[test]), index=test), smear


def tune(kind, X, target, fit, scale):
    Xf = X.loc[fit]
    if scale:
        Xf = pd.DataFrame(StandardScaler().fit(Xf).transform(Xf), index=Xf.index, columns=Xf.columns)
    g = GridSearchCV(learner(kind), GRIDS[kind], scoring="neg_mean_squared_error", cv=e3.FOLDS)
    g.fit(Xf, target.loc[fit])
    rows = [{"params": str(p), "mean_cv_mse": -s} for p, s in zip(g.cv_results_["params"],
                                                                 g.cv_results_["mean_test_score"])]
    return g.best_params_, rows


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    d, x9, y, ok = e3.load()
    d["vix"] = np.log(load_vix(d.index))
    d["lev_d"] = d["ret"].clip(upper=0)
    d["lev_w"] = d["lev_d"].rolling(5).mean()
    d["lev_m"] = d["lev_d"].rolling(22).mean()
    x13 = x9.join(d[EXTRA])
    ok = ok[d.loc[ok, EXTRA].notna().all(axis=1).to_numpy()]
    print(f"VIX joined: {d['vix'].notna().sum()} days; model days {len(ok)}")

    print("GARCH paths for each refit year ...")
    gpath = {year: garch_path(d, year) for year in TEST_YEARS}
    saved = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    gap = max((gpath[yr].reindex(saved.index[saved.index.year == yr]) /
               saved.loc[saved.index.year == yr, "GARCH"] - 1).abs().max() for yr in TEST_YEARS)
    print(f"  test-day GARCH vs saved main-study GARCH: largest relative difference {gap:.1e}")

    # specifications: name -> (learner, inputs, hybrid?)
    specs = {"XGB": ("XGB", "x9", False),
             "RF-hybrid": ("RF", "x9g", True), "SVR-hybrid": ("SVR", "x9g", True), "XGB-hybrid": ("XGB", "x9g", True),
             "RF-X": ("RF", "x13", False), "SVR-X": ("SVR", "x13", False), "XGB-X": ("XGB", "x13", False)}

    def inputs(name, year):
        if name == "x9":
            return x9
        if name == "x13":
            return x13
        return x9.assign(log_garch=np.log(gpath[year]).reindex(x9.index))

    def target(hybrid, year):
        return y - np.log(gpath[year]).reindex(y.index) if hybrid else y

    # ---- tuning once on the first window --------------------------------------------------------
    first = TEST_YEARS[0]
    fit, _, _ = e3.days_for(d, ok, first)
    best, tuning = {}, []
    for name, (kind, inp, hyb) in specs.items():
        best[name], rows = tune(kind, inputs(inp, first), target(hyb, first), fit, scale=(kind == "SVR"))
        tuning += [{"model": name, **r} for r in rows]
        print(f"  tuned {name:10s}: {best[name]}")
    pd.DataFrame(tuning).to_csv(RES / "e4_tuning.csv", index=False)

    # ---- walk-forward ---------------------------------------------------------------------------
    fc, log = [], []
    for year in TEST_YEARS:
        fit, val, test = e3.days_for(d, ok, year)
        out = {"rv5_fwd": d.loc[test, "rv5_fwd"]}
        for name, (kind, inp, hyb) in specs.items():
            pred, smear = fit_predict(kind, best[name], inputs(inp, year), target(hyb, year), fit, val, test,
                                      scale=(kind == "SVR"))
            base = gpath[year].reindex(test) if hyb else 1.0
            out[name] = base * np.exp(pred) * smear
            log.append({"test_year": year, "model": name, "n_fit": len(fit), "smear": smear,
                        "max_forecast": out[name].max(), "max_actual": out["rv5_fwd"].max()})
        # HAR-X: as the main HAR, OLS on all days before the test year (last 5 dropped), in-sample smearing
        tr = ok[ok < f"{year}-01-01"][:-HORIZON]
        hx = np.log(d[["rv_d", "rv_w", "rv_m"]].clip(lower=1e-8)).join(d[EXTRA])
        ols = sm.OLS(y.loc[tr], sm.add_constant(hx.loc[tr])).fit()
        out["HAR-X"] = np.exp(ols.predict(sm.add_constant(hx.loc[test], has_constant="add"))) * np.mean(np.exp(ols.resid))
        log.append({"test_year": year, "model": "HAR-X", "n_fit": len(tr), "smear": float(np.mean(np.exp(ols.resid))),
                    "max_forecast": out["HAR-X"].max(), "max_actual": out["rv5_fwd"].max(),
                    "coef_vix": ols.params["vix"], "p_vix": ols.pvalues["vix"],
                    "coef_lev_w": ols.params["lev_w"], "p_lev_w": ols.pvalues["lev_w"]})
        fc.append(pd.DataFrame(out))
        print(f"  {year}: done")

    fc = pd.concat(fc)
    lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    ml = pd.read_csv(RES / "ml_forecasts.csv", index_col=0, parse_dates=True)
    allm = fc.join(saved[["GARCH", "HAR"]]).join(lstm[["LSTM"]]).join(ml[["RF", "SVR"]])
    allm["GARCH+LSTM"] = (allm["GARCH"] + allm["LSTM"]) / 2
    allm["window"] = window_of(allm.index)
    assert allm.notna().all().all() and len(allm) == len(lstm), "forecast days do not line up"
    allm.to_csv(RES / "e4_forecasts.csv")
    pd.DataFrame(log).to_csv(RES / "e4_training_log.csv", index=False)

    models = ["GARCH", "HAR", "LSTM", "RF", "SVR", "XGB", "RF-hybrid", "SVR-hybrid", "XGB-hybrid",
              "HAR-X", "RF-X", "SVR-X", "XGB-X", "GARCH+LSTM"]
    losses = loss_table(allm, models)
    losses.to_csv(RES / "e4_losses.csv")
    q = losses[[f"QLIKE {m}" for m in models]].round(3)
    q.columns = models

    tests = []
    for a in ["RF-hybrid", "SVR-hybrid", "XGB-hybrid"]:
        t = dm_hln(qlike(allm.rv5_fwd, allm[a]), qlike(allm.rv5_fwd, allm.GARCH))
        tests.append({"family": "primary", "comparison": f"{a} vs GARCH", **t})
    adj = holm([t["p_value"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = a
    for a in ["XGB", "HAR-X", "RF-X", "SVR-X", "XGB-X", "GARCH+LSTM", "RF-hybrid", "SVR-hybrid", "XGB-hybrid"]:
        for b in ["GARCH", "HAR", "LSTM"]:
            if b == "GARCH" and a.endswith("hybrid"):
                continue
            t = dm_hln(qlike(allm.rv5_fwd, allm[a]), qlike(allm.rv5_fwd, allm[b]))
            tests.append({"family": "secondary", "comparison": f"{a} vs {b}", **t, "p_holm": np.nan})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "e4_dm_tests.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nQLIKE (lower is better):")
    print(q.T.to_string())
    print("\nDM-HLN, QLIKE, all test days (negative mean_diff = first model better):")
    print(tests.round(4).to_string(index=False))

    a, g = q.loc["All test days (2007 - Aug 2023)"], q.loc["Global Financial Crisis"]
    print("\nExpectations (Amendment A3):")
    print(" (i)   XGB not better than GARCH on all days, GFC above HAR's:", "met" if a.XGB >= a.GARCH and g.XGB > g.HAR else "NOT met")
    for k in ["RF", "SVR", "XGB"]:
        print(f" (ii)  {k}-hybrid GFC below {k}:", "met" if g[f"{k}-hybrid"] < g[k] else "NOT met")
    print(" (iii) HAR-X all days below HAR:", "met" if a["HAR-X"] < a.HAR else "NOT met")
    for k in ["RF-X", "SVR-X", "XGB-X"]:
        print(f" (iv)  {k} GFC above HAR's:", "met" if g[k] > g.HAR else "NOT met")
    print(" (v)   GARCH+LSTM all days below LSTM:", "met" if a["GARCH+LSTM"] < a.LSTM else "NOT met")

    plot(allm)
    print("\nSaved results/e4_*.csv and figures/e4_hybrids_gfc.png")


def plot(allm):
    """Top: GARCH, the hybrids and XGBoost on a normal scale. Bottom: log scale, for HAR-X's October 2008 spikes."""
    w = allm.loc["2008-01-01":"2009-06-30"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for a in (ax, bx):
        a.plot(w.index, w.rv5_fwd * 1e4, color="lightgrey", lw=1, label="Realised (next 5 days)")
        a.plot(w.index, w["GARCH"] * 1e4, color="tab:blue", lw=1.2, label="GARCH")
    for m, c in [("RF-hybrid", "tab:orange"), ("SVR-hybrid", "tab:purple"), ("XGB-hybrid", "tab:brown"),
                 ("XGB", "tab:red")]:
        ax.plot(w.index, w[m] * 1e4, color=c, lw=1.2, label=m)
    bx.plot(w.index, w["HAR-X"] * 1e4, color="tab:green", lw=1.2, label="HAR-X")
    bx.set_yscale("log")
    ax.set_ylabel("5-day variance (x 10^4)")
    bx.set_ylabel("5-day variance (x 10^4), log scale")
    ax.set_title("GARCH hybrids and XGBoost through the GFC, 2008 - mid 2009 (exploratory, Amendment A3)")
    bx.set_title("HAR-X (VIX + leverage): forecasts far above the realised peak in October 2008")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    bx.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "e4_hybrids_gfc.png", dpi=150)


if __name__ == "__main__":
    main()

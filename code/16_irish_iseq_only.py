"""
16_irish_iseq_only.py  -  EXPLORATORY, POST-HOC (PREREGISTRATION.md, Amendment A5, E7)
Which model is best in the Irish sovereign debt crisis when only ISEQ data is used?
The 14 models of E4, each in an ISEQ-only version, walk-forward for the test years 2010-2012 (each refit uses only
data before its test year), evaluated on the Irish window (23 Apr 2010 - 26 Jul 2012).
Primary: 90% Model Confidence Set (QLIKE, E6 settings). Secondary: DM-HLN vs GARCH; ISEQ-only vs three-market.
Run from the code folder:  py -3.13 16_irish_iseq_only.py   (about 5 minutes)
"""

import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS

from evaluation import HORIZON, window_of, qlike, mse, dm_hln
from lstm_common import Config, load_frame, fit_model, forecast

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
WINDOW = "Irish sovereign debt crisis"
YEARS = [2010, 2011, 2012]
SEEDS = [1, 2, 3, 4, 5]
HAR3 = ["rv_d", "rv_w", "rv_m"]
LEV = ["lev_d", "lev_w", "lev_m"]
MODELS = ["GARCH", "HAR", "LSTM-ISEQ", "RF-ISEQ", "SVR-ISEQ", "XGB-ISEQ", "RF-hybrid-ISEQ", "SVR-hybrid-ISEQ",
          "XGB-hybrid-ISEQ", "HAR-L", "RF-L", "SVR-L", "XGB-L", "GARCH+LSTM-ISEQ"]
THREE_MARKET = {"LSTM-ISEQ": "LSTM", "RF-ISEQ": "RF", "SVR-ISEQ": "SVR", "XGB-ISEQ": "XGB",
                "RF-hybrid-ISEQ": "RF-hybrid", "SVR-hybrid-ISEQ": "SVR-hybrid", "XGB-hybrid-ISEQ": "XGB-hybrid",
                "HAR-L": "HAR-X", "RF-L": "RF-X", "SVR-L": "SVR-X", "XGB-L": "XGB-X", "GARCH+LSTM-ISEQ": "GARCH+LSTM"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lstm_job(args):
    """One ISEQ-only LSTM (inputs ret, r2; main-study settings) for one test year and seed."""
    year, seed = args
    d = load_frame(inputs=["ret", "r2"])
    b = fit_model(d, f"{year - 1}-12-31", Config(lookback=22, hidden=64, layers=2), seed, inputs=["ret", "r2"])
    test = d.loc[f"{year}-01-01":f"{year}-12-31"].dropna(subset=["log_rv5_fwd"]).index
    return year, seed, forecast(b, d, test)


def main():
    e3 = _load("e3", "13_ml_benchmarks.py")
    e4 = _load("e4", "14_hybrid_boosting.py")
    d, x9, y, ok = e3.load()
    d["lev_d"] = d["ret"].clip(upper=0)
    d["lev_w"] = d["lev_d"].rolling(5).mean()
    d["lev_m"] = d["lev_d"].rolling(22).mean()
    x3 = x9[HAR3]
    x6 = x3.join(d[LEV])
    gpath = {year: e4.garch_path(d, year) for year in [2007] + YEARS}

    specs = {"RF-ISEQ": ("RF", "x3", False), "SVR-ISEQ": ("SVR", "x3", False), "XGB-ISEQ": ("XGB", "x3", False),
             "RF-hybrid-ISEQ": ("RF", "x3g", True), "SVR-hybrid-ISEQ": ("SVR", "x3g", True),
             "XGB-hybrid-ISEQ": ("XGB", "x3g", True),
             "RF-L": ("RF", "x6", False), "SVR-L": ("SVR", "x6", False), "XGB-L": ("XGB", "x6", False)}

    def inputs(name, year):
        if name == "x3":
            return x3
        if name == "x6":
            return x6
        return x3.assign(log_garch=np.log(gpath[year]).reindex(x3.index))

    def target(hybrid, year):
        return y - np.log(gpath[year]).reindex(y.index) if hybrid else y

    # ---- tuning once on the first window (fitting days 2003-2005), as in A2-A3 --------------------------------
    fit, _, _ = e3.days_for(d, ok, 2007)
    best, tuning = {}, []
    for name, (kind, inp, hyb) in specs.items():
        best[name], rows = e4.tune(kind, inputs(inp, 2007), target(hyb, 2007), fit, scale=(kind == "SVR"))
        tuning += [{"model": name, **r} for r in rows]
        print(f"  tuned {name:16s}: {best[name]}", flush=True)
    pd.DataFrame(tuning).to_csv(RES / "e7_tuning.csv", index=False)

    # ---- walk-forward for 2010-2012 ---------------------------------------------------------------------------------
    fc = []
    for year in YEARS:
        fit, val, test = e3.days_for(d, ok, year)
        out = {"rv5_fwd": d.loc[test, "rv5_fwd"]}
        for name, (kind, inp, hyb) in specs.items():
            pred, smear = e4.fit_predict(kind, best[name], inputs(inp, year), target(hyb, year), fit, val, test,
                                         scale=(kind == "SVR"))
            base = gpath[year].reindex(test) if hyb else 1.0
            out[name] = base * np.exp(pred) * smear
        tr = ok[ok < f"{year}-01-01"][:-HORIZON]
        hx = np.log(d[HAR3].clip(lower=1e-8)).join(d[LEV])
        ols = sm.OLS(y.loc[tr], sm.add_constant(hx.loc[tr])).fit()
        out["HAR-L"] = np.exp(ols.predict(sm.add_constant(hx.loc[test], has_constant="add"))) * np.mean(np.exp(ols.resid))
        fc.append(pd.DataFrame(out))
        print(f"  {year}: learners done", flush=True)
    fc = pd.concat(fc)

    print("  training 15 ISEQ-only LSTMs (3 years x 5 seeds) ...", flush=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        res = list(pool.map(lstm_job, [(yr, s) for yr in YEARS for s in SEEDS]))
    lstm = pd.concat([pd.DataFrame({s: f for yr, s, f in res if yr == year}) for year in YEARS]).sort_index()
    fc["LSTM-ISEQ"] = lstm.mean(axis=1).reindex(fc.index)

    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    fc = fc.join(bench[["GARCH", "HAR"]])
    fc["GARCH+LSTM-ISEQ"] = (fc["GARCH"] + fc["LSTM-ISEQ"]) / 2
    fc["window"] = window_of(fc.index)
    assert fc[MODELS].notna().all().all(), "missing forecasts"
    fc.to_csv(RES / "e7_forecasts.csv")

    # ---- evaluation on the Irish window ---------------------------------------------------------------------------------
    w = fc[fc.window == WINDOW]
    three = pd.read_csv(RES / "e4_forecasts.csv", index_col=0, parse_dates=True).loc[w.index]
    q = pd.DataFrame({m: qlike(w.rv5_fwd, w[m]) for m in MODELS}, index=w.index)
    rows = []
    for m in MODELS:
        t = dm_hln(q[m], q["GARCH"]) if m != "GARCH" else {"mean_diff": 0.0, "p_value": np.nan}
        tm = THREE_MARKET.get(m)
        rows.append({"model": m, "QLIKE": q[m].mean(), "MSE (x1e6)": mse(w.rv5_fwd, w[m]).mean() * 1e6,
                     "vs GARCH: diff": t["mean_diff"], "vs GARCH: DM p": t["p_value"],
                     "three-market version": tm or "",
                     "three-market QLIKE": qlike(three.rv5_fwd, three[tm]).mean() if tm else np.nan})
    tab = pd.DataFrame(rows).set_index("model")
    mcs = MCS(q, size=0.10, reps=10000, block_size=10, method="R", bootstrap="stationary", seed=2026)
    mcs.compute()
    tab["MCS p"] = mcs.pvalues["Pvalue"].reindex(MODELS)
    tab["in 90% MCS"] = [m in mcs.included for m in MODELS]
    tab.to_csv(RES / "e7_irish_results.csv")
    pd.set_option("display.width", 220)
    print(f"\nIrish sovereign debt crisis, {len(w)} days, ISEQ-only models (lower QLIKE is better):")
    print(tab.round(3).to_string())

    print("\nExpectations (Amendment A5):")
    print(" (i)   GARCH in the 90% MCS:", "met" if "GARCH" in mcs.included else "NOT met")
    better = tab[(tab["vs GARCH: diff"] < 0) & (tab["vs GARCH: DM p"] < 0.05)].index.tolist()
    print(" (ii)  no model significantly better than GARCH:", "met" if not better else f"NOT met {better}")
    for m in ["LSTM-ISEQ", "RF-ISEQ", "SVR-ISEQ", "XGB-ISEQ"]:
        print(f" (iii) {m} QLIKE {tab.loc[m, 'QLIKE']:.3f} <= three-market {tab.loc[m, 'three-market QLIKE']:.3f}:",
              "met" if tab.loc[m, "QLIKE"] <= tab.loc[m, "three-market QLIKE"] else "NOT met")
    plot(tab)
    print("\nSaved results/e7_*.csv and figures/e7_irish_iseq_only.png")


def plot(tab):
    """Dot plot: small differences stay visible without a truncated bar axis."""
    t = tab.sort_values("QLIKE")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    yy = np.arange(len(t))
    ax.scatter(t["three-market QLIKE"], yy, s=45, facecolor="none", edgecolor="#555", zorder=3,
               label="three-market version (E3-E4)")
    ax.scatter(t["QLIKE"], yy, s=55, zorder=4, c=["tab:blue" if b else "#bbbbbb" for b in t["in 90% MCS"]],
               label="ISEQ only (blue = in the 90% MCS, grey = excluded)")
    ax.axvline(t.loc["GARCH", "QLIKE"], color="tab:blue", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(yy, t.index)
    ax.invert_yaxis()
    ax.grid(axis="y", color="#eee", zorder=0)
    ax.set_xlabel("QLIKE in the Irish sovereign debt crisis (lower is better; dashed line = GARCH)")
    ax.set_title("E7: all models, ISEQ data only, Irish sovereign debt crisis (Amendment A5, post-hoc)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "e7_irish_iseq_only.png", dpi=150)


if __name__ == "__main__":
    main()

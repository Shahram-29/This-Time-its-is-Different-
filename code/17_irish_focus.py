"""
17_irish_focus.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A6, E8)
One market (ISEQ), one crisis (Ireland 2007-2012) in Kindleberger-Minsky phases, Irish data only.

Data:     ISEQ daily closes (data/processed/dataset.csv) + two Irish banks from Yahoo Finance (Bank of Ireland BIRG.IR,
          AIB A5G.IR; adjusted closes; bank return = mean of the two daily log returns).
Features: Set A (ISEQ, 6)  - log daily/weekly/monthly squared returns; leverage min(r,0) on the day, 5- and 22-day means
          Set B (+banks, 9) - Set A + log daily/weekly/monthly squared bank returns
Models:   GARCH, HAR (saved forecasts); per set: LSTM, Random Forest, SVR, XGBoost, Random Forest GARCH hybrid.
          Walk-forward, test years 2007-2012; plus a frozen version of every model (estimated on data to 2006 only).
Outputs:  results/e8_*.csv, figures/e8_*.png; LSTM models in models/irish_focus/ (explained by 18_irish_focus_shap.py)
Run from the code folder:  py -3.13 17_irish_focus.py   (about 20 minutes; downloads the two bank series once)
"""

import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import yfinance as yf
from arch.bootstrap import MCS
from sklearn.preprocessing import StandardScaler

from evaluation import SAMPLE_END, HORIZON, qlike, mse, dm_hln
from lstm_common import Config, fit_model, forecast, save_bundle

ROOT = Path(__file__).resolve().parent.parent
RAW, RES, FIG = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
MODELS_DIR = ROOT / "models" / "irish_focus"
BANKS = {"BIRG.IR": "Bank of Ireland", "A5G.IR": "AIB"}
PHASES = {"P1 Distress": ("2007-02-20", "2008-09-29"),
          "P2 Panic and guarantee": ("2008-09-30", "2009-03-09"),
          "P3 Relief": ("2009-03-10", "2010-04-22"),
          "P4 Sovereign crisis": ("2010-04-23", "2012-07-26")}
WHOLE = ("Whole crisis", ("2007-02-20", "2012-07-26"))
YEARS = list(range(2007, 2013))
SEEDS = [1, 2, 3, 4, 5]
ISEQ_SIZE = ["rv_d", "rv_w", "rv_m"]
ISEQ_LEV = ["lev_d", "lev_w", "lev_m"]
BANK_SIZE = ["bank_rv_d", "bank_rv_w", "bank_rv_m"]
SETS = {"A": ISEQ_SIZE + ISEQ_LEV, "B": ISEQ_SIZE + ISEQ_LEV + BANK_SIZE}
LSTM_INPUTS = {"A": ["ret", "r2"], "B": ["ret", "r2", "bank_ret", "bank_r2"]}
LSTM_CFG = Config(lookback=22, hidden=64, layers=2)
GROUP = {**{c: "ISEQ size" for c in ISEQ_SIZE}, **{c: "ISEQ leverage" for c in ISEQ_LEV},
         **{c: "banks" for c in BANK_SIZE}, "log_garch": "GARCH input"}
REFIT = ["GARCH", "HAR"] + [f"{m}-{s}" for s in "AB" for m in ["LSTM", "RF", "SVR", "XGB", "RF-hybrid"]]


def phase_of(dates):
    lab = pd.Series("outside", index=dates)
    for name, (a, b) in PHASES.items():
        lab[(dates >= a) & (dates <= b)] = name
    return lab


def bank_returns():
    """Equal-weight daily log return of Bank of Ireland and AIB (adjusted closes), cached in data/raw."""
    rets = []
    for t in BANKS:
        cache = RAW / f"{t.replace('.', '_')}.csv"
        if not cache.exists():
            df = yf.Ticker(t).history(start="2002-10-01", end="2025-12-31", auto_adjust=False)
            df.index = pd.to_datetime(df.index.date)
            df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].to_csv(cache)
        a = pd.read_csv(cache, index_col=0, parse_dates=True)["Adj Close"].dropna()
        rets.append(np.log(a[a > 0]).diff().rename(t))
    return pd.concat(rets, axis=1).dropna().mean(axis=1).rename("bank_ret").sort_index()


def build():
    """The ISEQ dataset with leverage and bank features added (bank series joined backward on ISEQ dates)."""
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True).loc[:SAMPLE_END]
    d = pd.merge_asof(d.sort_index(), bank_returns().to_frame(), left_index=True, right_index=True,
                      direction="backward")
    d["bank_r2"] = d["bank_ret"] ** 2
    d["bank_rv_d"] = d["bank_r2"]
    d["bank_rv_w"] = d["bank_r2"].rolling(5).mean()
    d["bank_rv_m"] = d["bank_r2"].rolling(22).mean()
    d["lev_d"] = d["ret"].clip(upper=0)
    d["lev_w"] = d["lev_d"].rolling(5).mean()
    d["lev_m"] = d["lev_d"].rolling(22).mean()
    x = d[SETS["B"]].copy()
    for c in ISEQ_SIZE + BANK_SIZE:
        x[c] = np.log(x[c].clip(lower=1e-8))
    y = np.log(d["rv5_fwd"])
    ok = d.dropna(subset=["rv5_fwd"] + SETS["B"]).index
    return d, x, y, ok


def lstm_frame(d, inputs):
    """Like lstm_common.load_frame: squared-return inputs in logs; rows with all inputs present."""
    f = d.copy()
    for c in inputs:
        if c.endswith("r2"):
            f[c] = np.log(f[c].clip(lower=1e-8))
    return f.dropna(subset=inputs)


def days_for(d, ok, year):
    fit = ok[ok <= f"{year - 2}-12-31"][:-HORIZON]
    val = ok[(ok >= f"{year - 1}-01-01") & (ok <= f"{year - 1}-12-31")][:-HORIZON]
    test = d.loc[f"{year}-01-01":f"{year}-12-31"].dropna(subset=["rv5_fwd"]).index
    return fit, val, test


def lstm_job(args):
    """One LSTM (feature set, test year, seed); the 2007 models also give the frozen forecasts for 2007-2012."""
    import torch
    torch.set_num_threads(2)
    s, year, seed = args
    d, _, _, _ = build()
    f = lstm_frame(d, LSTM_INPUTS[s])
    b = fit_model(f, f"{year - 1}-12-31", LSTM_CFG, seed, inputs=LSTM_INPUTS[s])
    test = f.loc[f"{year}-01-01":f"{year}-12-31"].dropna(subset=["log_rv5_fwd"]).index
    out = {"refit": forecast(b, f, test)}
    if year == YEARS[0]:
        allt = f.loc[f"{YEARS[0]}-01-01":f"{YEARS[-1]}-12-31"].dropna(subset=["log_rv5_fwd"]).index
        out["frozen"] = forecast(b, f, allt)
    save_bundle(b, MODELS_DIR / f"{s}_{year}_seed{seed}.pt")
    return s, year, seed, out


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    e4 = _load("e4", "14_hybrid_boosting.py")      # learners, GARCH paths, tuning (E3-E4 rules)
    s07 = _load("s07", "07_shap.py")               # the step-07 block bootstrap
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    d, x, y, ok = build()
    print(f"Data: {len(d)} ISEQ days; bank data joined on {d['bank_ret'].notna().sum()} days; model days {len(ok)}")
    gpath = {year: e4.garch_path(d, year) for year in YEARS}

    specs = {f"{k}-{s}": (k, s, False) for s in "AB" for k in ["RF", "SVR", "XGB"]}
    specs.update({f"RF-hybrid-{s}": ("RF", s, True) for s in "AB"})

    def inputs(s, year, hybrid):
        X = x[SETS[s]]
        return X.assign(log_garch=np.log(gpath[year]).reindex(X.index)) if hybrid else X

    def target(hybrid, year):
        return y - np.log(gpath[year]).reindex(y.index) if hybrid else y

    # ---- tuning once on the first window (fitting days 2003-2005) ------------------------------------------------
    fit0, _, _ = days_for(d, ok, YEARS[0])
    best, tuning = {}, []
    for name, (kind, s, hyb) in specs.items():
        best[name], rows = e4.tune(kind, inputs(s, YEARS[0], hyb), target(hyb, YEARS[0]), fit0, scale=(kind == "SVR"))
        tuning += [{"model": name, **r} for r in rows]
        print(f"  tuned {name:12s}: {best[name]}", flush=True)
    pd.DataFrame(tuning).to_csv(RES / "e8_tuning.csv", index=False)

    # ---- walk-forward (refit every year) and frozen (2007 model kept) ------------------------------------------------
    all_test = d.loc[f"{YEARS[0]}-01-01":f"{YEARS[-1]}-12-31"].dropna(subset=["rv5_fwd"]).index
    refit, frozen, tree_models = {}, {}, {}
    for year in YEARS:
        fit, val, test = days_for(d, ok, year)
        for name, (kind, s, hyb) in specs.items():
            X, z = inputs(s, year, hyb), target(hyb, year)
            sc = StandardScaler().fit(X.loc[fit]) if kind == "SVR" else None
            tf = (lambda Z, sc=sc: pd.DataFrame(sc.transform(Z), index=Z.index, columns=Z.columns)) if sc else (lambda Z: Z)
            m = e4.learner(kind, best[name]).fit(tf(X.loc[fit]), z.loc[fit])
            smear = float(np.mean(np.exp(z.loc[val] - m.predict(tf(X.loc[val])))))

            def predict(days, m=m, tf=tf, X=X, smear=smear, hyb=hyb, base=gpath[year]):
                p = np.exp(m.predict(tf(X.loc[days]))) * smear
                return pd.Series(p * (base.reindex(days).to_numpy() if hyb else 1.0), index=days)

            refit.setdefault(name, []).append(predict(test))
            if year == YEARS[0]:
                frozen[name] = predict(all_test)
            if name in ("RF-B", "RF-hybrid-B"):
                tree_models[(name, year)] = (m, X, fit)
        print(f"  {year}: learners refitted", flush=True)
    refit = {k: pd.concat(v) for k, v in refit.items()}

    # HAR frozen: the 2007 coefficients (data to 2006), as in 04_benchmarks.py; GARCH frozen: the 2007 parameters
    har_X = np.log(d[ISEQ_SIZE].clip(lower=1e-8))
    tr = d.loc[:"2006-12-31"].dropna(subset=["rv5_fwd", "rv_m"]).index[:-HORIZON]
    ols = sm.OLS(np.log(d.loc[tr, "rv5_fwd"]), sm.add_constant(har_X.loc[tr])).fit()
    frozen["HAR"] = pd.Series(np.exp(ols.predict(sm.add_constant(har_X.loc[all_test], has_constant="add")))
                              * np.mean(np.exp(ols.resid)), index=all_test)
    frozen["GARCH"] = gpath[YEARS[0]].reindex(all_test)

    print("  training 60 LSTMs (2 sets x 6 years x 5 seeds) ...", flush=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        res = list(pool.map(lstm_job, [(s, yr, sd) for s in "AB" for yr in YEARS for sd in SEEDS]))
    for s in "AB":
        refit[f"LSTM-{s}"] = pd.concat([pd.concat([o["refit"] for ss, yr, sd, o in res if ss == s and yr == year],
                                                  axis=1).mean(axis=1) for year in YEARS])
        frozen[f"LSTM-{s}"] = pd.concat([o["frozen"] for ss, yr, sd, o in res if ss == s and yr == YEARS[0]],
                                        axis=1).mean(axis=1)

    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    fc = pd.DataFrame({"rv5_fwd": d.loc[all_test, "rv5_fwd"], "GARCH": bench["GARCH"].reindex(all_test),
                       "HAR": bench["HAR"].reindex(all_test)})
    for k in REFIT[2:]:
        fc[k] = refit[k].reindex(all_test)
    for k in REFIT:
        fc[f"{k} (frozen)"] = frozen[k].reindex(all_test)
    fc["phase"] = phase_of(all_test)
    assert fc.drop(columns="phase").notna().all().all(), "missing forecasts"
    fc.to_csv(RES / "e8_forecasts.csv")
    check = (gpath[2010].reindex(all_test[all_test.year == 2010]) / bench["GARCH"].reindex(all_test[all_test.year == 2010]) - 1).abs().max()
    print(f"  check: refitted GARCH path vs saved GARCH (2010), largest relative difference {check:.1e}")

    evaluate(fc, s07)
    explain_trees(tree_models, fc, s07)
    plot_phases(d, fc)
    print("\nSaved results/e8_*.csv and figures/e8_*.png; LSTM models in models/irish_focus/")


def samples(fc):
    out = [(p, fc[fc.phase == p]) for p in PHASES]
    a, b = WHOLE[1]
    return out + [(WHOLE[0], fc.loc[a:b])]


def evaluate(fc, s07):
    rows, mcs_rows, tests = [], [], []
    for name, g in samples(fc):
        q = pd.DataFrame({m: qlike(g.rv5_fwd, g[m]) for m in REFIT}, index=g.index)
        mcs = MCS(q, size=0.10, reps=10000, block_size=10, method="R", bootstrap="stationary", seed=2026)
        mcs.compute()
        for m in REFIT:
            t = dm_hln(q[m], q["GARCH"]) if m != "GARCH" else {"mean_diff": 0.0, "p_value": np.nan}
            fq = qlike(g.rv5_fwd, g[f"{m} (frozen)"]).mean()
            rows.append({"phase": name, "days": len(g), "model": m, "QLIKE": q[m].mean(),
                         "MSE (x1e6)": mse(g.rv5_fwd, g[m]).mean() * 1e6, "DM vs GARCH: diff": t["mean_diff"],
                         "DM vs GARCH: p": t["p_value"], "MCS p": mcs.pvalues["Pvalue"][m],
                         "in 90% MCS": m in mcs.included, "QLIKE frozen": fq, "frozen / refit": fq / q[m].mean()})
        for k in ["LSTM", "RF", "SVR", "XGB", "RF-hybrid"]:
            t = dm_hln(q[f"{k}-B"], q[f"{k}-A"])
            tests.append({"phase": name, "comparison": f"{k}: banks (B) vs ISEQ only (A)", **t})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "e8_results.csv", index=False)
    pd.DataFrame(tests).to_csv(RES / "e8_bank_tests.csv", index=False)
    pd.set_option("display.width", 250)
    for col, label in [("QLIKE", "QLIKE (lower is better; * = in the 90% MCS)"), ("frozen / refit", "Frozen / refitted QLIKE (Minsky test)")]:
        t = res.pivot(index="model", columns="phase", values=col).reindex(REFIT)[list(PHASES) + [WHOLE[0]]]
        if col == "QLIKE":
            star = res.pivot(index="model", columns="phase", values="in 90% MCS").reindex(REFIT)[t.columns]
            t = t.round(3).astype(str) + star.replace({True: "*", False: ""})
        print(f"\n{label}:\n" + (t if col == "QLIKE" else t.round(2)).to_string())
    print("\nBank inputs (DM-HLN, QLIKE; negative = banks help):\n"
          + pd.DataFrame(tests).pivot(index="comparison", columns="phase", values="mean_diff")[list(PHASES) + [WHOLE[0]]].round(3).to_string()
          + "\np-values:\n"
          + pd.DataFrame(tests).pivot(index="comparison", columns="phase", values="p_value")[list(PHASES) + [WHOLE[0]]].round(3).to_string())

    get = lambda p, m, c="QLIKE": res[(res.phase == p) & (res.model == m)][c].iloc[0]
    plain = [f"{k}-{s}" for s in "AB" for k in ["LSTM", "RF", "SVR", "XGB"]]
    early = list(PHASES)[:2]
    print("\nExpectations (Amendment A6):")
    print(" (i)   GARCH in the 90% MCS in P1 and P2:", "met" if all(get(p, "GARCH", "in 90% MCS") for p in early) else "NOT met")
    worse = [(p, m) for p in early for m in plain if get(p, m) <= get(p, "GARCH")]
    print(" (ii)  every plain learner above GARCH in P1 and P2:", "met" if not worse else f"NOT met {worse}")
    for p in early:
        n = sum(get(p, f"{k}-B") < get(p, f"{k}-A") for k in ["LSTM", "RF", "SVR", "XGB", "RF-hybrid"])
        print(f" (iii) {p}: banks lower QLIKE for {n} of 5 learners:", "met" if n >= 3 else "NOT met")
    worse = [(p, m) for p in early for m in plain if get(p, m, "frozen / refit") <= get(p, "GARCH", "frozen / refit")]
    print(" (iv)  frozen/refit ratio of every plain learner above GARCH's in P1 and P2:", "met" if not worse else f"NOT met {worse}")


def explain_trees(tree_models, fc, s07):
    """Interventional TreeSHAP (200 background fitting days) of RF-B and RF-hybrid-B on every crisis day."""
    days = fc.index[fc.phase != "outside"]
    group_rows, feat_rows = [], []
    for name in ["RF-B", "RF-hybrid-B"]:
        parts = []
        for year in YEARS:
            dd = days[days.year == year]
            if len(dd) == 0:
                continue
            m, X, fit = tree_models[(name, year)]
            bg = X.loc[fit].to_numpy()[np.random.default_rng(1000 * year).choice(len(fit), 200, replace=False)]
            ex = shap.TreeExplainer(m, data=shap.maskers.Independent(bg, max_samples=200),
                                    feature_perturbation="interventional")
            parts.append(pd.DataFrame(ex.shap_values(X.loc[dd].to_numpy(), check_additivity=False),
                                      index=dd, columns=X.columns))
        sv = pd.concat(parts).sort_index().abs()
        ph = fc.loc[sv.index, "phase"]
        groups = [g for g in ["ISEQ size", "ISEQ leverage", "banks", "GARCH input"] if any(GROUP[c] == g for c in sv.columns)]
        per_day = np.stack([sv[[c for c in sv.columns if GROUP[c] == g]].sum(axis=1) for g in groups], axis=1)
        for p in PHASES:
            msk = (ph == p).to_numpy()
            tot = per_day[msk].sum(axis=0)
            lo, hi = s07.block_bootstrap(per_day[msk])
            for i, g in enumerate(groups):
                group_rows.append({"model": name, "phase": p, "group": g, "share": tot[i] / tot.sum(),
                                   "ci_low": lo[i], "ci_high": hi[i]})
            for c in sv.columns:
                feat_rows.append({"model": name, "phase": p, "feature": c, "mean_abs_shap": sv[c][msk].mean()})
    gr = pd.DataFrame(group_rows)
    gr.to_csv(RES / "e8_treeshap_groups.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(RES / "e8_treeshap_features.csv", index=False)
    for name, g in gr.groupby("model", sort=False):
        t = g.assign(txt=g.apply(lambda r: f"{r.share:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
        print(f"\nTreeSHAP shares, {name} (95% block-bootstrap intervals):\n"
              + t.pivot(index="phase", columns="group", values="txt").reindex(list(PHASES)).to_string())


def plot_phases(d, fc):
    colours = {"P1 Distress": "#fde0c5", "P2 Panic and guarantee": "#f4a582", "P3 Relief": "#d1e5f0",
               "P4 Sovereign crisis": "#e7d4e8"}
    p = pd.read_csv(RAW / "ISEQ.csv", index_col=0, parse_dates=True)["Close"].loc["2003":"2012"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax.plot(p.index, p, color="black", lw=1)
    ax.axvspan(pd.Timestamp("2003-01-02"), pd.Timestamp("2007-02-19"), color="#e5f5e0", lw=0, label="Boom (training only)")
    for a in (ax, bx):
        for name, (s, e) in PHASES.items():
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=colours[name], lw=0,
                      label=name if a is ax else None)
    ax.set_ylabel("ISEQ Overall")
    ax.set_title("E8: the Irish crisis in Kindleberger-Minsky phases (Amendment A6)", fontsize=10)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    w = fc.loc[:"2012-12-31"]
    bx.plot(w.index, w.rv5_fwd * 1e4, color="#999", lw=0.8, label="Realised (next 5 days)")
    for m, c in [("GARCH", "tab:blue"), ("LSTM-B", "tab:red"), ("RF-B", "tab:orange"), ("RF-hybrid-B", "tab:purple")]:
        bx.plot(w.index, w[m] * 1e4, color=c, lw=1, label=m)
    bx.set_yscale("log")
    bx.set_ylabel("5-day variance (x 10^4, log scale)")
    bx.legend(fontsize=7, ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "e8_phases.png", dpi=150)


if __name__ == "__main__":
    main()

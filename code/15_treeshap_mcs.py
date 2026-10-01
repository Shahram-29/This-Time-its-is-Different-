"""
15_treeshap_mcs.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A4: E5 and E6)
E5 - What does GARCH miss? TreeSHAP (interventional) of the GARCH hybrids' predicted correction
     log(RV5) - log(GARCH), by channel (domestic / US / euro / GARCH input), on the same days as the LSTM SHAP
     analysis (step 07), with the same share definition and block bootstrap.
E6 - Which models are statistically best? Model Confidence Set (Hansen, Lunde & Nason, 2011) over the 14 models of
     E4, QLIKE (primary) and MSE, all test days; per window and sensitivity runs as secondary.
Run from the code folder:  py -3.13 15_treeshap_mcs.py   (about 10-20 minutes)
"""

import ast
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from arch.bootstrap import MCS

from evaluation import TEST_YEARS, qlike, mse

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e3 = _load("e3", "13_ml_benchmarks.py")       # data, day splits
e4 = _load("e4", "14_hybrid_boosting.py")     # GARCH paths, learners
s07 = _load("s07", "07_shap.py")              # the step-07 block bootstrap

CHANNEL = dict(e3.CHANNEL, log_garch="GARCH")
WINDOWS = ["calm", "Global Financial Crisis", "Irish sovereign debt crisis", "COVID-19", "War / energy shock 2022"]
CRISES = WINDOWS[1:]
N_BACKGROUND = 200
MCS_MODELS = ["GARCH", "HAR", "LSTM", "RF", "SVR", "XGB", "RF-hybrid", "SVR-hybrid", "XGB-hybrid",
              "HAR-X", "RF-X", "SVR-X", "XGB-X", "GARCH+LSTM"]


def best_params(model):
    t = pd.read_csv(RES / "e4_tuning.csv")
    t = t[t.model == model]
    return ast.literal_eval(t.loc[t.mean_cv_mse.idxmin(), "params"])


def shares_table(abs_shap, win, channels):
    """Step-07 shares: per window, channel sum of |SHAP| / total, with the step-07 block bootstrap."""
    rows = []
    per_day = np.stack([abs_shap[:, [i for i, c in enumerate(abs_shap_cols) if CHANNEL[c] == ch]].sum(axis=1)
                        for ch in channels], axis=1)
    for w in WINDOWS:
        m = (win == w).to_numpy()
        tot = per_day[m].sum(axis=0)
        sh = tot / tot.sum()
        lo, hi = s07.block_bootstrap(per_day[m])
        for i, ch in enumerate(channels):
            rows.append({"window": w, "channel": ch, "days": int(m.sum()), "share": sh[i], "ci_low": lo[i],
                         "ci_high": hi[i]})
    return pd.DataFrame(rows)


abs_shap_cols = e3.COLUMNS + ["log_garch"]


def explain(model_name):
    d, x9, y, ok = e3.load()
    saved = pd.read_csv(RES / "e4_forecasts.csv", index_col=0, parse_dates=True)
    lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    calm = lstm.index[lstm["window"] == "calm"][::10]                     # the step-07 day selection
    pick = lstm.index[lstm["window"].isin(CRISES)].union(calm)
    kind = model_name.split("-")[0]
    params = best_params(model_name)
    out, gaps, repro = [], [], []
    for year in TEST_YEARS:
        days = pick[pick.year == year]
        if len(days) == 0:
            continue
        g = e4.garch_path(d, year)
        X = x9.assign(log_garch=np.log(g).reindex(x9.index))
        z = y - np.log(g).reindex(y.index)
        fit, val, test = e3.days_for(d, ok, year)
        m = e4.learner(kind, params).fit(X.loc[fit], z.loc[fit])
        smear = float(np.mean(np.exp(z.loc[val] - m.predict(X.loc[val]))))
        f = g.reindex(test) * np.exp(m.predict(X.loc[test])) * smear
        repro.append(float((f / saved.loc[test, model_name] - 1).abs().max()))
        bg = X.loc[fit].to_numpy()[np.random.default_rng(1000 * year).choice(len(fit), N_BACKGROUND, replace=False)]
        # all 200 background days (shap's default would subsample to 100); XGBoost is passed as its booster,
        # because shap 0.52 wrongly reports categorical splits for an xgboost 3 XGBRegressor
        ex = shap.TreeExplainer(m.get_booster() if kind == "XGB" else m,
                                data=shap.maskers.Independent(bg, max_samples=N_BACKGROUND),
                                feature_perturbation="interventional")
        sv = ex.shap_values(X.loc[days].to_numpy(), check_additivity=False)
        gaps.append(np.abs(sv.sum(axis=1) + ex.expected_value - m.predict(X.loc[days])))
        out.append(pd.DataFrame(sv, index=days, columns=abs_shap_cols))
        print(f"  {model_name} {year}: {len(days)} days explained", flush=True)
    sv = pd.concat(out).sort_index()
    print(f"  {model_name}: refit vs saved E4 forecasts, largest relative difference {max(repro):.1e}; "
          f"additivity gap mean {np.concatenate(gaps).mean():.1e}, max {np.concatenate(gaps).max():.1e}")
    return sv, lstm.loc[sv.index, "window"]


def e5():
    lstm_sh = pd.read_csv(RES / "shap_channel_shares.csv")
    get_l = lambda w, c: lstm_sh[(lstm_sh.window == w) & (lstm_sh.channel == c)].share.iloc[0]
    tables, checks = [], []
    for name in ["RF-hybrid", "XGB-hybrid"]:
        sv, win = explain(name)
        a = np.abs(sv.to_numpy())
        for version, chans in [("4 channels", ["domestic", "US", "euro", "GARCH"]),
                               ("3 channels", ["domestic", "US", "euro"])]:
            t = shares_table(a, win, chans)
            t.insert(0, "version", version)
            t.insert(0, "model", name)
            tables.append(t)
        t3 = tables[-1]
        get = lambda w, c: t3[(t3.window == w) & (t3.channel == c)].iloc[0]
        rises = lambda w, c: get(w, c).ci_low > get("calm", c).share
        for w in CRISES:
            checks.append({"model": name, "check": f"(i) foreign share above LSTM's: {w}",
                           "value": f"{1 - get(w, 'domestic').share:.3f} vs {1 - get_l(w, 'domestic'):.3f}",
                           "met": bool(1 - get(w, "domestic").share > 1 - get_l(w, "domestic"))})
        for w, c in [("Global Financial Crisis", "US"), ("COVID-19", "US"), ("Irish sovereign debt crisis", "euro"),
                     ("War / energy shock 2022", "euro")]:
            checks.append({"model": name, "check": f"(ii) {c} share rises: {w}",
                           "value": f"{get(w, c).ci_low:.3f} (low) vs calm {get('calm', c).share:.3f}",
                           "met": bool(rises(w, c))})
        sign = sum(np.sign(get(w, c).share - get("calm", c).share) == np.sign(get_l(w, c) - get_l("calm", c))
                   for w in CRISES for c in ["US", "euro"])
        rank = sum((get(w, "US").share > get(w, "euro").share) == (get_l(w, "US") > get_l(w, "euro")) for w in WINDOWS)
        checks.append({"model": name, "check": "agreement with LSTM SHAP: sign of change from calm (of 8)",
                       "value": str(int(sign)), "met": np.nan})
        checks.append({"model": name, "check": "agreement with LSTM SHAP: US-vs-euro ranking (of 5)",
                       "value": str(int(rank)), "met": np.nan})
    shares = pd.concat(tables)
    shares.to_csv(RES / "e5_correction_shares.csv", index=False)
    checks = pd.DataFrame(checks)
    checks.to_csv(RES / "e5_checks.csv", index=False)
    pd.set_option("display.width", 220)
    for (m, v), g in shares.groupby(["model", "version"], sort=False):
        print(f"\nE5 {m}, {v}: share of |SHAP| of the correction [95% interval]")
        g = g.assign(txt=g.apply(lambda r: f"{r.share:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
        print(g.pivot(index="window", columns="channel", values="txt").reindex(WINDOWS).to_string())
    print("\nE5 checks:\n" + checks.to_string(index=False))
    plot_e5(shares, lstm_sh)


def plot_e5(shares, lstm_sh):
    t = shares[(shares.model == "RF-hybrid") & (shares.version == "3 channels")]
    colours = {"domestic": "#1b1b1b", "US": "#1f77b4", "euro": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(WINDOWS))
    for j, (label, src) in enumerate([("LSTM", lstm_sh), ("RF-hybrid correction", t)]):
        bottom = np.zeros(len(WINDOWS))
        for c in ["domestic", "US", "euro"]:
            v = np.array([src[(src.window == w) & (src.channel == c)].share.iloc[0] for w in WINDOWS])
            ax.bar(x + (j - 0.5) * 0.38, v, 0.36, bottom=bottom, color=colours[c], alpha=1.0 if j == 0 else 0.55,
                   label=f"{c} ({label})")
            bottom += v
    ax.set_xticks(x, ["Calm", "GFC", "Irish debt", "COVID-19", "2022"])
    ax.set_ylabel("share of |SHAP| (three information channels)")
    ax.set_title("E5: what the LSTM uses (dark, left) vs what GARCH misses (light, right: RF-hybrid correction)",
                 fontsize=10)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIG / "e5_correction_shares.png", dpi=150)


def e6():
    f = pd.read_csv(RES / "e4_forecasts.csv", index_col=0, parse_dates=True)
    q = pd.DataFrame({m: qlike(f.rv5_fwd, f[m]) for m in MCS_MODELS}, index=f.index)
    s = pd.DataFrame({m: mse(f.rv5_fwd, f[m]) * 1e6 for m in MCS_MODELS}, index=f.index)
    runs = [("QLIKE", "All test days", "main", q, 10, "R"), ("MSE", "All test days", "main", s, 10, "R"),
            ("QLIKE", "All test days", "block 5", q, 5, "R"), ("QLIKE", "All test days", "block 22", q, 22, "R"),
            ("QLIKE", "All test days", "max statistic", q, 10, "max")]
    runs += [("QLIKE", w, "window", q[f.window == w], 10, "R") for w in WINDOWS]
    rows = []
    for loss, sample, spec, L, block, method in runs:
        mcs = MCS(L, size=0.10, reps=10000, block_size=block, method=method, bootstrap="stationary", seed=2026)
        mcs.compute()
        p = mcs.pvalues["Pvalue"]
        for m in MCS_MODELS:
            rows.append({"loss": loss, "sample": sample, "spec": spec, "model": m, "mean_loss": L[m].mean(),
                         "mcs_pvalue": p[m], "in_90_mcs": m in mcs.included})
        print(f"  MCS {loss:5s} {sample:28s} {spec:13s}: {len(mcs.included)} models in the 90% set", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(RES / "e6_mcs.csv", index=False)
    main = r[(r.spec == "main")].pivot(index="model", columns="loss", values="mcs_pvalue").reindex(MCS_MODELS)
    main = main.join(r[(r.spec == "main") & (r.loss == "QLIKE")].set_index("model")["mean_loss"].rename("mean QLIKE"))
    print("\nE6 MCS p-values, all test days (in the 90% set if p >= 0.10):\n" + main.round(3).to_string())
    win = r[r.spec == "window"].pivot(index="model", columns="sample", values="mcs_pvalue").reindex(MCS_MODELS)[WINDOWS]
    print("\nE6 MCS p-values by window (QLIKE):\n" + win.round(3).to_string())
    sens = r[(r["sample"] == "All test days") & (r.loss == "QLIKE")].pivot(index="model", columns="spec",
                                                                         values="in_90_mcs").reindex(MCS_MODELS)
    print("\nE6 sensitivity (in the 90% set, QLIKE, all days):\n" + sens.to_string())
    inq = set(r[(r.spec == "main") & (r.loss == "QLIKE") & r.in_90_mcs].model)
    ins = set(r[(r.spec == "main") & (r.loss == "MSE") & r.in_90_mcs].model)
    print("\nExpectations (Amendment A4, E6):")
    print(" (i)   GARCH in the QLIKE 90% MCS:", "met" if "GARCH" in inq else "NOT met")
    print(" (ii)  LSTM, RF, SVR, XGB not in it:", "met" if not inq & {"LSTM", "RF", "SVR", "XGB"} else "NOT met")
    print(" (iii) HAR-X not in the MSE 90% MCS:", "met" if "HAR-X" not in ins else "NOT met")


if __name__ == "__main__":
    print("E5: TreeSHAP of the GARCH-hybrid correction")
    e5()
    print("\nE6: Model Confidence Set")
    e6()
    print("\nSaved results/e5_*.csv, results/e6_mcs.csv, figures/e5_correction_shares.png")

"""
08_robustness.py
Pre-registered robustness checks. Run one part at a time or all in order:

    py -3.13 08_robustness.py lookback66     # LSTM walk-forward with a 66-day look-back
    py -3.13 08_robustness.py ftse           # LSTM walk-forward with FTSE 100 in place of DAX
    py -3.13 08_robustness.py ftse_shap      # SHAP for the FTSE models (incl. Brexit, H2d) via 07_shap.py
    py -3.13 08_robustness.py parkinson      # re-score all forecasts against Parkinson range volatility
    py -3.13 08_robustness.py extension      # Sep 2023 - Dec 2025 (after the ISEQ composition break)
    py -3.13 08_robustness.py summary        # one table comparing every check with the main result
    py -3.13 08_robustness.py all

Nothing here changes the main results; every check is reported next to them.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from evaluation import TEST_YEARS, SAMPLE_END, HORIZON, WINDOWS, window_of, loss_table, dm_hln, qlike
from lstm_common import INPUTS, FTSE_INPUTS, Config, load_frame, fit_model, forecast, save_bundle, load_bundle

ROOT = Path(__file__).resolve().parent.parent
RES, CODE = ROOT / "results", ROOT / "code"
SEEDS = [1, 2, 3, 4, 5]
MAIN_CFG = Config(lookback=22, hidden=64, layers=2)          # frozen by the pre-registered tuning in 05
EXT_END = "2025-12-31"
WORKERS = 5


def _init():
    import torch
    torch.set_num_threads(2)


def _fit(args):
    cfg, inputs, year, seed, end = args
    return year, seed, fit_model(load_frame(inputs, end), f"{year - 1}-12-31", cfg, seed, inputs)


def walk_forward(cfg, inputs, tag, years=TEST_YEARS, end=SAMPLE_END):
    """Same protocol as 05 (annual refits, 5 seeds, mean forecast) for a robustness variant."""
    d = load_frame(inputs, end)
    mdir = ROOT / "models" / f"lstm_{tag}"
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as pool:
        fits = list(pool.map(_fit, [(cfg, inputs, y, s, end) for y in years for s in SEEDS]))
    per_seed = {s: [] for s in SEEDS}
    for year, seed, b in fits:
        last = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end))
        test = d.loc[f"{year}-01-01":last].dropna(subset=["log_rv5_fwd"]).index
        per_seed[seed].append(forecast(b, d, test))
        save_bundle(b, mdir / f"{year}_seed{seed}.pt")
    fc = pd.DataFrame({f"LSTM_seed{s}": pd.concat(per_seed[s]) for s in SEEDS}).sort_index()
    fc.insert(0, "LSTM", fc.mean(axis=1))
    fc.insert(0, "rv5_fwd", np.exp(d.loc[fc.index, "log_rv5_fwd"]))
    fc["window"] = window_of(fc.index)
    print(f"[{tag}] {len(fits)} models, {len(fc)} forecast days, {time.time() - t0:.0f} s")
    return fc


def compare(fc_variant, name):
    """QLIKE by window and DM-HLN vs GARCH/HAR, for a variant LSTM on the main test days."""
    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    main = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    fc = bench.join(main[["LSTM"]]).join(fc_variant[["LSTM"]].rename(columns={"LSTM": name}), how="inner")
    tab = loss_table(fc, ["LSTM", name, "GARCH", "HAR"])
    tests = [{"variant": name, "comparison": f"{name} vs {b}",
              **dm_hln(qlike(fc.rv5_fwd, fc[name]), qlike(fc.rv5_fwd, fc[b]))} for b in ["GARCH", "HAR", "LSTM"]]
    return tab, pd.DataFrame(tests)


# ---- parts -------------------------------------------------------------------------------------------
def part_lookback66():
    fc = walk_forward(Config(lookback=66, hidden=64, layers=2), INPUTS, "lookback66")
    fc.to_csv(RES / "robust_lookback66_forecasts.csv")
    tab, tests = compare(fc, "LSTM_L66")
    tab.to_csv(RES / "robust_lookback66_losses.csv")
    tests.to_csv(RES / "robust_lookback66_dm.csv", index=False)
    print(tab.filter(like="QLIKE").round(3).to_string(), "\n", tests.round(4).to_string(index=False))


def part_ftse():
    fc = walk_forward(MAIN_CFG, FTSE_INPUTS, "ftse")
    fc.to_csv(RES / "robust_ftse_forecasts.csv")
    tab, tests = compare(fc, "LSTM_FTSE")
    tab.to_csv(RES / "robust_ftse_losses.csv")
    tests.to_csv(RES / "robust_ftse_dm.csv", index=False)
    print(tab.filter(like="QLIKE").round(3).to_string(), "\n", tests.round(4).to_string(index=False))


def part_ftse_shap():
    env = dict(os.environ, DISS_VARIANT="ftse")
    subprocess.run([sys.executable, str(CODE / "07_shap.py")], env=env, check=True, cwd=CODE)


def part_parkinson():
    """Re-score every model against forward 5-day Parkinson variance, scaled to close-to-close RV using
    2003-2006 only (before any test day), because the intraday range excludes overnight moves."""
    raw = pd.read_csv(ROOT / "data" / "raw" / "ISEQ.csv", index_col=0, parse_dates=True)
    park = np.log(raw["High"] / raw["Low"]) ** 2 / (4 * np.log(2))
    park5 = sum(park.shift(-k) for k in range(1, HORIZON + 1))
    d = load_frame()
    rv5 = np.exp(d["log_rv5_fwd"])
    pre = slice("2003-01-01", "2006-12-31")
    c = rv5.loc[pre].mean() / park5.loc[pre].reindex(rv5.loc[pre].index).mean()
    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    fc = bench.join(lstm[["LSTM"]], how="inner")
    fc["rv5_fwd"] = c * park5.reindex(fc.index)
    fc = fc[fc["rv5_fwd"] > 0].dropna(subset=["rv5_fwd"])
    tab = loss_table(fc, ["LSTM", "GARCH", "HAR"])
    tab.to_csv(RES / "robust_parkinson_losses.csv")
    tests = pd.DataFrame([{"variant": "Parkinson proxy", "comparison": f"LSTM vs {b}",
                           **dm_hln(qlike(fc.rv5_fwd, fc.LSTM), qlike(fc.rv5_fwd, fc[b]))} for b in ["GARCH", "HAR"]])
    tests.to_csv(RES / "robust_parkinson_dm.csv", index=False)
    print(f"Parkinson scale factor (2003-2006) = {c:.3f}; {len(fc)} days")
    print(tab.filter(like="QLIKE").round(3).to_string(), "\n", tests.round(4).to_string(index=False))


def _benchmarks(d, year, start, end):
    """HAR and GARCH as in 04, for one refit year, forecasting days start..end."""
    X = np.log(d[["rv_d", "rv_w", "rv_m"]].clip(lower=1e-8))
    y = np.log(d["rv5_fwd"])
    test = d.loc[start:end].dropna(subset=["rv5_fwd"]).index
    tr = d.loc[:f"{year - 1}-12-31"].dropna(subset=["rv5_fwd", "rv_m"]).index[:-HORIZON]
    fit = sm.OLS(y.loc[tr], sm.add_constant(X.loc[tr])).fit()
    har = np.exp(fit.predict(sm.add_constant(X.loc[test], has_constant="add"))) * np.mean(np.exp(fit.resid))
    res = arch_model(d["ret"].dropna() * 100, dist="t").fit(last_obs=pd.Timestamp(f"{year}-01-01"), disp="off")
    garch = res.forecast(horizon=HORIZON, start=test[0], reindex=False).variance.loc[test].sum(axis=1) / 1e4
    return pd.DataFrame({"rv5_fwd": d.loc[test, "rv5_fwd"], "HAR": har, "GARCH": garch})


def part_extension():
    """Sep 2023 - Dec 2025: Sep-Dec 2023 uses the 2023 models; 2024 and 2025 are refitted as usual."""
    raw = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True).loc[:EXT_END]
    d = load_frame(INPUTS, EXT_END)
    parts = [(2023, "2023-09-01", "2023-12-31"), (2024, "2024-01-01", "2024-12-31"), (2025, "2025-01-01", EXT_END)]
    bench = pd.concat([_benchmarks(raw, y, a, b) for y, a, b in parts])
    per_seed = {s: [] for s in SEEDS}
    for s in SEEDS:
        b23 = load_bundle(ROOT / "models" / "lstm" / f"2023_seed{s}.pt")
        per_seed[s].append(forecast(b23, d, d.loc["2023-09-01":"2023-12-31"].dropna(subset=["log_rv5_fwd"]).index))
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as pool:
        fits = list(pool.map(_fit, [(MAIN_CFG, INPUTS, y, s, EXT_END) for y in (2024, 2025) for s in SEEDS]))
    for year, seed, b in fits:
        per_seed[seed].append(forecast(b, d, d.loc[f"{year}-01-01":f"{year}-12-31"].dropna(subset=["log_rv5_fwd"]).index))
    lstm = pd.DataFrame({s: pd.concat(v) for s, v in per_seed.items()}).mean(axis=1).rename("LSTM")
    fc = bench.join(lstm, how="inner")
    fc["window"] = "Sep 2023 - Dec 2025"
    fc.to_csv(RES / "robust_extension_forecasts.csv")
    tab = loss_table(fc, ["LSTM", "GARCH", "HAR"])
    tab.to_csv(RES / "robust_extension_losses.csv")
    tests = pd.DataFrame([{"variant": "Extension Sep 2023 - Dec 2025", "comparison": f"LSTM vs {b}",
                           **dm_hln(qlike(fc.rv5_fwd, fc.LSTM), qlike(fc.rv5_fwd, fc[b]))} for b in ["GARCH", "HAR"]])
    tests.to_csv(RES / "robust_extension_dm.csv", index=False)
    print(f"Extension: {len(fc)} days")
    print(tab.filter(like="QLIKE").round(3).to_string(), "\n", tests.round(4).to_string(index=False))


def part_summary():
    """All-days QLIKE of the LSTM variant vs benchmarks, and the DM verdicts, in one table."""
    rows = []
    main = pd.read_csv(RES / "evaluation_losses.csv", index_col=0)
    main_dm = pd.read_csv(RES / "evaluation_dm_tests.csv")
    main_dm = main_dm[(main_dm.window == "All test days") & (main_dm.loss == "QLIKE")].set_index("comparison")
    rows.append({"check": "Main result", "QLIKE LSTM": main.iloc[0]["QLIKE LSTM"],
                 "QLIKE GARCH": main.iloc[0]["QLIKE GARCH"], "QLIKE HAR": main.iloc[0]["QLIKE HAR"],
                 "p LSTM vs GARCH": main_dm.loc["LSTM vs GARCH", "p_value"],
                 "p LSTM vs HAR": main_dm.loc["LSTM vs HAR", "p_value"]})
    specs = [("Look-back 66 days", "lookback66", "LSTM_L66"), ("FTSE instead of DAX", "ftse", "LSTM_FTSE"),
             ("Parkinson volatility proxy", "parkinson", "LSTM"), ("Extension Sep 2023 - Dec 2025", "extension", "LSTM")]
    for label, tag, col in specs:
        f_loss, f_dm = RES / f"robust_{tag}_losses.csv", RES / f"robust_{tag}_dm.csv"
        if not f_loss.exists():
            continue
        tab, dm = pd.read_csv(f_loss, index_col=0), pd.read_csv(f_dm)
        first = tab.iloc[0]
        p = lambda b: dm[dm.comparison.str.endswith(f"vs {b}")]["p_value"].iloc[0]
        rows.append({"check": label, "QLIKE LSTM": first[f"QLIKE {col}"], "QLIKE GARCH": first["QLIKE GARCH"],
                     "QLIKE HAR": first["QLIKE HAR"], "p LSTM vs GARCH": p("GARCH"), "p LSTM vs HAR": p("HAR")})
    out = pd.DataFrame(rows).set_index("check")
    out["LSTM best?"] = (out["QLIKE LSTM"] < out[["QLIKE GARCH", "QLIKE HAR"]].min(axis=1))
    out.to_csv(RES / "robustness_summary.csv")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string())


PARTS = {"lookback66": part_lookback66, "ftse": part_ftse, "ftse_shap": part_ftse_shap,
         "parkinson": part_parkinson, "extension": part_extension, "summary": part_summary}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    for name in (list(PARTS) if todo == ["all"] else todo):
        print(f"\n===== {name} =====")
        PARTS[name]()

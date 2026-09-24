"""
06_evaluation.py
Pre-registered evaluation.

RQ1 / H1: does the LSTM have lower QLIKE than GARCH(1,1) and HAR out-of-sample (2007 - Aug 2023)?
          Diebold-Mariano with HLN correction, 5% level, all test days. Per-window results are descriptive.
RQ3 / H3: "learning from history" (Minsky test). LSTMs frozen at the start of each crisis (trained only on
          earlier data, never refitted) forecast the whole crisis window. Measure: frozen-LSTM QLIKE / HAR QLIKE
          in the window. Pre-registered ranking of degradation: GFC > COVID > Irish sovereign debt crisis.

Outputs: results/evaluation_losses.csv, evaluation_dm_tests.csv, evaluation_frozen.csv,
figures/evaluation_qlike_by_window.png, figures/evaluation_frozen.png
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import WINDOWS, loss_table, dm_hln, qlike, mse
from lstm_common import Config, load_frame, fit_model, forecast

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
SEEDS = [1, 2, 3, 4, 5]
FROZEN = {"Global Financial Crisis": "2007-08-08", "Irish sovereign debt crisis": "2010-04-22",
          "COVID-19": "2020-02-18"}                      # last training day = day before each window
H3_ORDER = ["Global Financial Crisis", "COVID-19", "Irish sovereign debt crisis"]   # largest -> smallest


def _init():
    import torch
    torch.set_num_threads(2)


def _frozen_job(args):
    cfg, crisis, train_end, seed = args
    return crisis, seed, fit_model(load_frame(), train_end, cfg, seed)


def main():
    bench = pd.read_csv(RES / "benchmark_forecasts.csv", index_col=0, parse_dates=True)
    lstm = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    fc = bench.join(lstm[["LSTM"]], how="inner")
    models = ["LSTM", "GARCH", "HAR", "Naive"]

    # ---- losses ---------------------------------------------------------------------------
    losses = loss_table(fc, models)
    order = ["All test days (2007 - Aug 2023)", "calm", *WINDOWS]
    losses = losses.reindex([w for w in order if w in losses.index])
    losses.to_csv(RES / "evaluation_losses.csv")
    pd.set_option("display.width", 220)
    print("Mean QLIKE (lower is better):")
    print(losses[["days"] + [f"QLIKE {m}" for m in models]].round(3).to_string())

    # ---- H1: Diebold-Mariano-HLN ------------------------------------------------------------
    tests = []
    for loss_name, fn in [("QLIKE", qlike), ("MSE", mse)]:
        for bench_m in ["GARCH", "HAR"]:
            for w_name, g in [("All test days", fc)] + list(fc.groupby("window")):
                t = dm_hln(fn(g["rv5_fwd"], g["LSTM"]), fn(g["rv5_fwd"], g[bench_m]))
                tests.append({"loss": loss_name, "comparison": f"LSTM vs {bench_m}", "window": w_name, **t})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "evaluation_dm_tests.csv", index=False)
    main_tests = tests[(tests.window == "All test days")]
    print("\nH1 - Diebold-Mariano-HLN, all test days (negative mean_diff = LSTM more accurate):")
    print(main_tests[["loss", "comparison", "n", "mean_diff", "stat", "p_value"]].round(4).to_string(index=False))
    q = main_tests[main_tests.loss == "QLIKE"].set_index("comparison")
    h1 = all((q.loc[c, "mean_diff"] < 0) and (q.loc[c, "p_value"] < 0.05) for c in q.index)
    print(f"H1 {'SUPPORTED' if h1 else 'NOT SUPPORTED'}: LSTM QLIKE significantly lower than both benchmarks "
          f"at 5% = {h1}")

    # ---- RQ3: frozen models ---------------------------------------------------------------------
    tune = pd.read_csv(RES / "lstm_tuning.csv")
    L, H, K = tune.groupby(["lookback", "hidden", "layers"])["val_loss"].mean().idxmin()
    cfg = Config(lookback=int(L), hidden=int(H), layers=int(K))
    t0 = time.time()
    jobs = [(cfg, c, end, s) for c, end in FROZEN.items() for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as pool:
        fits = list(pool.map(_frozen_job, jobs))
    d = load_frame()
    rows = []
    for crisis in FROZEN:
        days = fc.index[fc["window"] == crisis]
        frozen = np.mean([forecast(b, d, days).to_numpy() for c, s, b in fits if c == crisis], axis=0)
        real = fc.loc[days, "rv5_fwd"]
        q_frozen = qlike(real, frozen).mean()
        q_refit = qlike(real, fc.loc[days, "LSTM"]).mean()
        q_har = qlike(real, fc.loc[days, "HAR"]).mean()
        rows.append({"crisis": crisis, "trained_until": FROZEN[crisis], "days": len(days),
                     "QLIKE_frozen_LSTM": q_frozen, "QLIKE_refitted_LSTM": q_refit, "QLIKE_HAR": q_har,
                     "ratio_frozen_to_HAR": q_frozen / q_har, "ratio_refitted_to_HAR": q_refit / q_har})
    frozen_tab = pd.DataFrame(rows).set_index("crisis")
    frozen_tab.to_csv(RES / "evaluation_frozen.csv")
    print(f"\nRQ3 - frozen LSTMs ({time.time() - t0:.0f} s):")
    print(frozen_tab.round(3).to_string())
    ranking = list(frozen_tab["ratio_frozen_to_HAR"].sort_values(ascending=False).index)
    print(f"Observed degradation ranking: {' > '.join(ranking)}")
    print(f"H3 {'SUPPORTED' if ranking == H3_ORDER else 'NOT SUPPORTED'} "
          f"(pre-registered: {' > '.join(H3_ORDER)})")

    # ---- figures -----------------------------------------------------------------------------
    show = ["calm", *[w for w in WINDOWS if w in losses.index]]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(show))
    for i, (m, c) in enumerate(zip(["HAR", "GARCH", "LSTM"], ["#1f77b4", "#d62728", "#2ca02c"])):
        ax.bar(x + (i - 1) * 0.27, losses.loc[show, f"QLIKE {m}"], width=0.27, color=c, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" / ", "/\n").replace(" sovereign", "\nsovereign") for s in show], fontsize=8)
    ax.set_ylabel("mean QLIKE (lower is better)")
    ax.legend(frameon=False)
    ax.set_title("Forecast accuracy by window, walk-forward 2007 - Aug 2023", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "evaluation_qlike_by_window.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(frozen_tab))
    ax.bar(x - 0.2, frozen_tab["ratio_frozen_to_HAR"], 0.4, color="#9e9e9e", label="frozen at crisis start")
    ax.bar(x + 0.2, frozen_tab["ratio_refitted_to_HAR"], 0.4, color="#2ca02c", label="refitted every year")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frozen_tab.index, fontsize=8)
    ax.set_ylabel("LSTM QLIKE / HAR QLIKE")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Learning from history: LSTM accuracy relative to HAR in each crisis", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "evaluation_frozen.png", dpi=150)
    print("\nSaved results/evaluation_*.csv and figures/evaluation_*.png")


if __name__ == "__main__":
    main()

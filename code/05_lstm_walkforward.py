"""
05_lstm_walkforward.py
LSTM forecasts of ISEQ volatility, walk-forward with annual refits (pre-registered design).

1. Tuning (pre-registered grid, first windows only): look-back {22, 66} x units {32, 64} x layers {1, 2},
   scored by mean validation loss for the 2007 and 2008 refits (validation = 2006 and 2007),
   2 seeds each. The best configuration is then frozen for every year.
2. Walk-forward: for each test year 2007-2023, train on data to 31 Dec of the previous year with
   5 seeds; the evaluated forecast is the mean of the 5 seed forecasts.

Outputs: results/lstm_tuning.csv, lstm_training_log.csv, lstm_forecasts.csv; models/lstm/*.pt
Run time: roughly 20-60 minutes on a laptop CPU (runs seeds in parallel).
"""

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import TEST_YEARS, SAMPLE_END, window_of
from lstm_common import Config, load_frame, fit_model, forecast, save_bundle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = ROOT / "models" / "lstm"
SEEDS = [1, 2, 3, 4, 5]
TUNE_YEARS, TUNE_SEEDS = [2007, 2008], [101, 102]
GRID = [Config(lookback=L, hidden=H, layers=K) for L, H, K in itertools.product([22, 66], [32, 64], [1, 2])]
WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))


def _init_worker():
    import torch
    torch.set_num_threads(2)


def _job(args):
    cfg, year, seed = args
    d = load_frame()
    t0 = time.time()
    b = fit_model(d, f"{year - 1}-12-31", cfg, seed)
    b["seconds"] = time.time() - t0
    b["year"] = year
    return b


def main():
    RES.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    d = load_frame()
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as pool:
        # ---- 1. tuning on the first windows only ------------------------------------------
        jobs = [(cfg, y, s) for cfg in GRID for y in TUNE_YEARS for s in TUNE_SEEDS]
        tune = [{"lookback": b["cfg"].lookback, "hidden": b["cfg"].hidden, "layers": b["cfg"].layers,
                 "year": b["year"], "seed": b["seed"], "val_loss": b["val_loss"], "epochs": b["epochs"]}
                for b in pool.map(_job, jobs)]
        tune = pd.DataFrame(tune)
        summary = tune.groupby(["lookback", "hidden", "layers"])["val_loss"].mean().sort_values()
        tune.to_csv(RES / "lstm_tuning.csv", index=False)
        L, H, K = (int(v) for v in summary.index[0])             # numpy ints -> Python ints for PyTorch
        best = Config(lookback=L, hidden=H, layers=K)
        print("Tuning (mean validation loss, standardised log RV; lower is better):")
        print(summary.round(4).to_string())
        print(f"-> frozen configuration: look-back {L}, {H} units, {K} layer(s)  "
              f"[{time.time() - t_start:.0f} s]\n")

        # ---- 2. walk-forward, 5 seeds per annual refit ---------------------------------------
        jobs = [(best, y, s) for y in TEST_YEARS for s in SEEDS]
        bundles = list(pool.map(_job, jobs))

    log, per_seed = [], {s: [] for s in SEEDS}
    for b in bundles:
        y = b["year"]
        start = pd.Timestamp(f"{y}-01-01")
        end = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(SAMPLE_END))
        test = d.loc[start:end].dropna(subset=["log_rv5_fwd"]).index
        per_seed[b["seed"]].append(forecast(b, d, test))
        save_bundle(b, MODELS / f"{y}_seed{b['seed']}.pt")
        log.append({"test_year": y, "seed": b["seed"], "epochs": b["epochs"], "val_loss": b["val_loss"],
                    "smear": b["smear"], "n_fit": b["n_fit"], "n_val": b["n_val"], "seconds": round(b["seconds"], 1)})
    log = pd.DataFrame(log).sort_values(["test_year", "seed"])
    log.to_csv(RES / "lstm_training_log.csv", index=False)

    fc = pd.DataFrame({f"LSTM_seed{s}": pd.concat(per_seed[s]) for s in SEEDS}).sort_index()
    fc.insert(0, "LSTM", fc.mean(axis=1))
    fc.insert(0, "rv5_fwd", np.exp(d.loc[fc.index, "log_rv5_fwd"]))
    fc["window"] = window_of(fc.index)
    fc.to_csv(RES / "lstm_forecasts.csv")

    print(log.groupby("test_year")[["epochs", "val_loss", "smear", "seconds"]].mean().round(3).to_string())
    print(f"\nSaved {len(fc)} forecast days, {len(bundles)} models. Total time {time.time() - t_start:.0f} s.")


if __name__ == "__main__":
    main()

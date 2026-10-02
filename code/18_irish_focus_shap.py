"""
18_irish_focus_shap.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A6, E8)
GradientShap of the Set-B LSTM (ISEQ + Irish banks; models from 17_irish_focus.py) on every crisis day, exactly as in
step 07: 1,024 samples, 200 background windows from each model's own fitting period, seeds 1-5, 4-day chunks.
Shares by phase: ISEQ vs banks; direction (returns) vs size (squared returns); the lag profile (last 5 days vs older).
Run from the code folder after 17:  py -3.13 18_irish_focus_shap.py   (about 20 minutes)
"""

import importlib.util
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_common import load_bundle, split_positions, windows

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
N_SAMPLES, CHUNK, N_BACKGROUND = 1024, 4, 200
WORKERS, THREADS = 2, 3
INPUTS = ["ret", "r2", "bank_ret", "bank_r2"]
CHANNELS = {"ISEQ": [0, 1], "banks": [2, 3]}
TYPES = {"direction (returns)": [0, 2], "size (squared returns)": [1, 3]}
SEEDS = [1, 2, 3, 4, 5]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s17 = _load("s17", "17_irish_focus.py")


def _explain(args):
    import torch
    from captum.attr import GradientShap
    torch.set_num_threads(THREADS)
    year, seed, dates = args
    d, _, _, _ = s17.build()
    f = s17.lstm_frame(d, INPUTS)
    b = load_bundle(s17.MODELS_DIR / f"B_{year}_seed{seed}.pt")
    X = ((f[INPUTS] - b["mu"]) / b["sd"]).to_numpy(np.float32)
    lb = b["cfg"].lookback
    fit_pos, _ = split_positions(f, b["train_end"], lb)
    rng = np.random.default_rng(1000 * year + seed)
    bg = torch.tensor(windows(X, rng.choice(fit_pos, N_BACKGROUND, replace=False), lb))
    xe = torch.tensor(windows(X, f.index.get_indexer(pd.DatetimeIndex(dates)), lb))
    torch.manual_seed(seed)
    np.random.seed(seed)                       # Captum draws its samples with numpy
    gs = GradientShap(b["model"])
    attr = np.concatenate([gs.attribute(xe[i:i + CHUNK], baselines=bg, n_samples=N_SAMPLES, stdevs=0.0)
                           .detach().numpy() for i in range(0, len(xe), CHUNK)])
    with torch.no_grad():
        gap = attr.sum(axis=(1, 2)) - (b["model"](xe).numpy() - b["model"](bg).numpy().mean())
    return year, seed, list(dates), attr, gap


def main():
    s07 = _load("s07", "07_shap.py")
    t0 = time.time()
    fc = pd.read_csv(RES / "e8_forecasts.csv", index_col=0, parse_dates=True)
    pick = fc.index[fc.phase != "outside"]
    jobs = [(y, s, list(g)) for (y, g), s in itertools.product(pd.Series(pick, index=pick).groupby(pick.year), SEEDS)]
    print(f"explaining {len(pick)} crisis days x {len(SEEDS)} seeds = {len(jobs)} model-year jobs", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(_explain, jobs), 1):
            out.append(r)
            print(f"  {i}/{len(jobs)} done (year {r[0]}, seed {r[1]}) after {time.time() - t0:.0f} s", flush=True)
    A = {s: {} for s in SEEDS}
    gaps = []
    for year, seed, dates, attr, gap in out:
        for dt, a in zip(dates, attr):
            A[seed][pd.Timestamp(dt)] = a
        gaps.append(np.abs(gap))
    days = pd.DatetimeIndex(sorted(A[SEEDS[0]]))
    ph = fc.loc[days, "phase"]
    abs_all = np.stack([np.stack([np.abs(A[s][dt]) for dt in days]) for s in SEEDS])   # seeds x days x 22 x 4
    abs_mean = abs_all.mean(axis=0)
    print(f"Additivity: mean |sum of attributions - (forecast - baseline)| = {np.concatenate(gaps).mean():.4f}")

    rows, lag_rows = [], []
    for p in s17.PHASES:
        m = (ph == p).to_numpy()
        for measure, groups in [("channel", CHANNELS), ("input type", TYPES)]:
            per_day = np.stack([abs_mean[m][:, :, idx].sum(axis=(1, 2)) for idx in groups.values()], axis=1)
            sh = per_day.sum(axis=0) / per_day.sum()
            lo, hi = s07.block_bootstrap(per_day)
            seed_sh = np.stack([(lambda a: np.array([a[:, :, idx].sum() for idx in groups.values()]) / a.sum())(abs_all[k][m])
                                for k in range(len(SEEDS))])
            for i, g in enumerate(groups):
                rows.append({"phase": p, "measure": measure, "group": g, "days": int(m.sum()), "share": sh[i],
                             "ci_low": lo[i], "ci_high": hi[i], "seed_min": seed_sh[:, i].min(),
                             "seed_max": seed_sh[:, i].max()})
        lag = abs_mean[m].sum(axis=(0, 2))
        lag = lag / lag.sum()
        for k in range(len(lag)):
            lag_rows.append({"phase": p, "days_back": len(lag) - 1 - k, "share": lag[k]})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "e8_lstm_shap.csv", index=False)
    lags = pd.DataFrame(lag_rows)
    lags.to_csv(RES / "e8_lstm_shap_lags.csv", index=False)
    pd.set_option("display.width", 220)
    for measure, g in res.groupby("measure", sort=False):
        t = g.assign(txt=g.apply(lambda r: f"{r.share:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
        print(f"\nLSTM-B GradientShap, {measure} shares (95% block-bootstrap intervals):\n"
              + t.pivot(index="phase", columns="group", values="txt").reindex(list(s17.PHASES)).to_string())
    recent = lags[lags.days_back <= 4].groupby("phase").share.sum().reindex(list(s17.PHASES))
    print("\nShare of attribution on the last 5 days (lags 0-4):\n" + recent.round(3).to_string())

    tree = pd.read_csv(RES / "e8_treeshap_groups.csv")
    lstm_bank = res[(res.measure == "channel") & (res.group == "banks")].set_index("phase").share
    rf_bank = tree[(tree.model == "RF-B") & (tree.group == "banks")].set_index("phase").share
    print("\nExpectation (v), bank share highest in P2:")
    for name, s in [("LSTM-B", lstm_bank), ("RF-B", rf_bank)]:
        print(f"  {name}: highest in {s.idxmax()} ({s.max():.3f}) ->", "met" if s.idxmax() == "P2 Panic and guarantee" else "NOT met")
    plot(res, lags, tree)
    print(f"\nSaved results/e8_lstm_shap*.csv and figures/e8_shap.png  [{time.time() - t0:.0f} s]")


def plot(res, lags, tree):
    phases = list(s17.PHASES)
    short = ["P1 Distress", "P2 Panic", "P3 Relief", "P4 Sovereign"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    x = np.arange(len(phases))
    lb = res[(res.measure == "channel") & (res.group == "banks")].set_index("phase").reindex(phases)
    axs[0].bar(x - 0.2, lb.share, 0.38, yerr=[lb.share - lb.ci_low, lb.ci_high - lb.share], capsize=3,
               color="tab:red", label="LSTM-B (GradientShap)")
    for k, (name, col) in enumerate([("RF-B", "tab:orange")]):
        tb = tree[(tree.model == name) & (tree.group == "banks")].set_index("phase").reindex(phases)
        axs[0].bar(x + 0.2, tb.share, 0.38, yerr=[tb.share - tb.ci_low, tb.ci_high - tb.share], capsize=3,
                   color=col, label=f"{name} (TreeSHAP)")
    axs[0].set_xticks(x, short, fontsize=8)
    axs[0].set_ylabel("share of |SHAP| from the bank inputs")
    axs[0].set_title("Irish banks' share of the explanation", fontsize=10)
    axs[0].legend(fontsize=7, loc="upper left")
    axs[0].set_ylim(0, 0.8)
    th = tree[tree.model == "RF-hybrid-B"]
    bottom = np.zeros(len(phases))
    for g, col in [("ISEQ size", "#1b1b1b"), ("ISEQ leverage", "#777777"), ("banks", "tab:red"), ("GARCH input", "tab:blue")]:
        v = th[th.group == g].set_index("phase").reindex(phases).share.to_numpy()
        axs[1].bar(x, v, 0.6, bottom=bottom, color=col, label=g)
        bottom += v
    axs[1].set_xticks(x, short, fontsize=8)
    axs[1].set_title("What the RF-hybrid adds to GARCH (TreeSHAP)", fontsize=10)
    axs[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for p, col in zip(phases, ["#e6550d", "#a50f15", "#3182bd", "#756bb1"]):
        l = lags[lags.phase == p].sort_values("days_back")
        axs[2].plot(l.days_back, l.share, color=col, lw=1.4, label=p)
    axs[2].set_xlabel("days back (0 = forecast day)")
    axs[2].set_ylabel("share of LSTM |attribution|")
    axs[2].set_title("How far back the LSTM looks", fontsize=10)
    axs[2].legend(fontsize=7)
    fig.suptitle("E8: explaining the Irish-crisis forecasts by phase (Amendment A6)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e8_shap.png", dpi=150)


if __name__ == "__main__":
    main()

"""
07_shap.py
Pre-registered explanation analysis (RQ2 fingerprints, RQ4 stability).

For each forecast day, the model that actually produced it (test year's refit) is explained with
Captum GradientShap (expected gradients; 1,024 samples; background = 200 windows from that model's own
training period), separately for each of the 5 seeds.

Days explained: every day in each evaluation window, plus every 10th calm day (2007 - Aug 2023).
Brexit is explained only in the FTSE robustness run (step 08), so it is skipped here.

Measures (per window):
  * channel share = sum of |attribution| in the channel (all 22 lags, both inputs) / total |attribution|
  * 95% interval from a block bootstrap over explained days (blocks of 10 consecutive explained days)
  * absolute attribution per day, input-type shares (direction vs size), lag profile
  * RQ4: mean pairwise Spearman correlation of channel shares across the 5 seeds

H2 verdicts use the pre-registered rules ("rises" = the window's interval lies above the calm share).

Outputs: results/shap_*.csv, figures/shap_*.png
"""

import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lstm_common import CHANNEL, INPUTS, load_frame, load_bundle, windows, split_positions

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, MODELS = ROOT / "results", ROOT / "figures", ROOT / "models" / "lstm"
SEEDS = [1, 2, 3, 4, 5]
N_SAMPLES, CHUNK, N_BACKGROUND = 1024, 8, 200
CHANNELS = ["domestic", "US", "euro"]
WINDOW_ORDER = ["calm", "Global Financial Crisis", "Irish sovereign debt crisis", "COVID-19", "War / energy shock 2022"]
CH_IDX = {c: [i for i, k in enumerate(INPUTS) if CHANNEL[k] == c] for c in CHANNELS}
SIZE_IDX = [i for i, k in enumerate(INPUTS) if k.endswith("r2")]


def _init():
    import torch
    torch.set_num_threads(2)


def _explain(args):
    """Attributions (days x 22 x 6) for one model on the given dates."""
    import torch
    from captum.attr import GradientShap
    year, seed, dates = args
    d = load_frame()
    b = load_bundle(MODELS / f"{year}_seed{seed}.pt")
    cfg = b["cfg"]
    X = ((d[b["inputs"]] - b["mu"]) / b["sd"]).to_numpy(np.float32)
    fit_pos, _ = split_positions(d, b["train_end"], cfg.lookback)
    rng = np.random.default_rng(1000 * year + seed)
    bg = torch.tensor(windows(X, rng.choice(fit_pos, N_BACKGROUND, replace=False), cfg.lookback))
    xe = torch.tensor(windows(X, d.index.get_indexer(pd.DatetimeIndex(dates)), cfg.lookback))
    torch.manual_seed(seed)
    gs = GradientShap(b["model"])
    attr = np.concatenate([gs.attribute(xe[i:i + CHUNK], baselines=bg, n_samples=N_SAMPLES, stdevs=0.0)
                           .detach().numpy() for i in range(0, len(xe), CHUNK)])
    with torch.no_grad():
        gap = attr.sum(axis=(1, 2)) - (b["model"](xe).numpy() - b["model"](bg).numpy().mean())
    return year, seed, list(dates), attr, gap


def shares(a):
    """Channel shares from an (n, 22, 6) array of |attributions|."""
    tot = a.sum()
    return np.array([a[:, :, CH_IDX[c]].sum() / tot for c in CHANNELS])


def block_bootstrap(per_day, n_boot=1000, block=10, seed=0):
    """95% interval of channel shares; per_day = (n_days, 3) channel |attribution| sums in date order."""
    rng = np.random.default_rng(seed)
    n = len(per_day)
    starts = np.arange(0, n - block + 1)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, int(np.ceil(n / block)))])[:n]
        s = per_day[idx].sum(axis=0)
        draws.append(s / s.sum())
    return np.percentile(draws, [2.5, 97.5], axis=0)


def main():
    t0 = time.time()
    fc = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    calm = fc.index[fc["window"] == "calm"][::10]
    pick = fc.index[fc["window"].isin(WINDOW_ORDER[1:])].union(calm)
    jobs = [(y, s, list(g)) for (y, g), s in itertools.product(pd.Series(pick, index=pick).groupby(pick.year), SEEDS)]
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as pool:
        out = list(pool.map(_explain, jobs))
    print(f"Explained {len(pick)} days x {len(SEEDS)} seeds in {time.time() - t0:.0f} s")

    # assemble: attr[seed] = DataFrame of (day -> 22x6 array)
    A = {s: {} for s in SEEDS}
    gaps = []
    for year, seed, dates, attr, gap in out:
        for dt, a in zip(dates, attr):
            A[seed][pd.Timestamp(dt)] = a
        gaps.append(np.abs(gap))
    gaps = np.concatenate(gaps)
    days = pd.DatetimeIndex(sorted(A[SEEDS[0]]))
    win = fc.loc[days, "window"]
    abs_all = np.stack([np.stack([np.abs(A[s][dt]) for dt in days]) for s in SEEDS])    # seeds x days x 22 x 6
    abs_mean = abs_all.mean(axis=0)                                                       # days x 22 x 6
    print(f"Additivity check: mean |sum of attributions - (forecast - baseline)| = {gaps.mean():.4f}")

    # ---- channel shares with bootstrap intervals, input types, lags -----------------------------
    rows, lag_rows, type_rows, seed_rows, stab_rows = [], [], [], [], []
    for w in WINDOW_ORDER:
        m = (win == w).to_numpy()
        a = abs_mean[m]
        per_day = np.stack([a[:, :, CH_IDX[c]].sum(axis=(1, 2)) for c in CHANNELS], axis=1)
        sh, (lo, hi) = shares(a), block_bootstrap(per_day)
        for i, c in enumerate(CHANNELS):
            rows.append({"window": w, "channel": c, "days": int(m.sum()), "share": sh[i], "ci_low": lo[i],
                         "ci_high": hi[i], "mean_abs_attribution_per_day": per_day[:, i].mean()})
        size = a[:, :, SIZE_IDX].sum() / a.sum()
        type_rows.append({"window": w, "size (squared returns)": size, "direction (returns)": 1 - size})
        lag = a.sum(axis=(0, 2)) / a.sum()
        lag_rows.append({"window": w, **{f"t-{21 - k}": lag[k] for k in range(21, -1, -1)}})
        by_seed = np.stack([shares(abs_all[i][m]) for i in range(len(SEEDS))])
        for i, s in enumerate(SEEDS):
            seed_rows.append({"window": w, "seed": s, **dict(zip(CHANNELS, by_seed[i]))})
        rho = [spearmanr(by_seed[i], by_seed[j])[0] for i, j in itertools.combinations(range(len(SEEDS)), 2)]
        feat = np.stack([abs_all[i][m].sum(axis=(0, 1)) for i in range(len(SEEDS))])
        rho6 = [spearmanr(feat[i], feat[j])[0] for i, j in itertools.combinations(range(len(SEEDS)), 2)]
        stab_rows.append({"window": w, "mean_spearman_channels": np.nanmean(rho),
                          "min_spearman_channels": np.nanmin(rho), "mean_spearman_6_inputs": np.mean(rho6)})

    ch = pd.DataFrame(rows)
    ch.to_csv(RES / "shap_channel_shares.csv", index=False)
    pd.DataFrame(type_rows).to_csv(RES / "shap_input_types.csv", index=False)
    lags = pd.DataFrame(lag_rows).set_index("window")
    lags.to_csv(RES / "shap_lags.csv")
    pd.DataFrame(seed_rows).to_csv(RES / "shap_by_seed.csv", index=False)
    stab = pd.DataFrame(stab_rows).set_index("window")
    stab.to_csv(RES / "shap_stability.csv")

    pd.set_option("display.width", 220)
    tab = ch.pivot(index="window", columns="channel", values="share").reindex(WINDOW_ORDER)[CHANNELS]
    ci = ch.assign(ci=ch.apply(lambda r: f"{r.share:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)) \
        .pivot(index="window", columns="channel", values="ci").reindex(WINDOW_ORDER)[CHANNELS]
    print("\nChannel shares of |attribution| with 95% block-bootstrap intervals:\n" + ci.to_string())
    print("\nInput types:\n" + pd.DataFrame(type_rows).set_index("window").round(3).to_string())
    print("\nRQ4 stability across seeds:\n" + stab.round(3).to_string())

    # ---- H2 verdicts (pre-registered rules) -----------------------------------------------------
    get = lambda w, c, k="share": ch[(ch.window == w) & (ch.channel == c)][k].iloc[0]
    rises = lambda w, c: get(w, c, "ci_low") > get("calm", c)
    change = lambda w: {c: get(w, c) - get("calm", c) for c in CHANNELS}
    gfc = change("Global Financial Crisis")
    spread = lambda w: max(get(w, c) for c in CHANNELS) - min(get(w, c) for c in CHANNELS)
    verdicts = [
        ("H2a GFC: US share rises most; domestic above calm",
         max(gfc, key=gfc.get) == "US" and rises("Global Financial Crisis", "US")
         and rises("Global Financial Crisis", "domestic")),
        ("H2b Irish debt crisis: euro and domestic rise; US below its GFC level",
         rises("Irish sovereign debt crisis", "euro") and rises("Irish sovereign debt crisis", "domestic")
         and get("Irish sovereign debt crisis", "US") < get("Global Financial Crisis", "US")),
        ("H2c COVID: shares move towards equality (smaller max-min spread than calm)",
         spread("COVID-19") < spread("calm")),
        ("H2e 2022 shock: euro share rises", rises("War / energy shock 2022", "euro")),
        ("H4 stability: mean pairwise Spearman >= 0.8 in every window",
         bool((stab["mean_spearman_channels"] >= 0.8).all())),
    ]
    vt = pd.DataFrame(verdicts, columns=["hypothesis", "supported"])
    vt.to_csv(RES / "shap_hypotheses.csv", index=False)
    print("\nPre-registered verdicts:\n" + vt.to_string(index=False))

    # ---- figures -------------------------------------------------------------------------------------
    colours = {"domestic": "#1b1b1b", "US": "#1f77b4", "euro": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 4.3))
    x = np.arange(len(WINDOW_ORDER))
    for i, c in enumerate(CHANNELS):
        sub = ch[ch.channel == c].set_index("window").reindex(WINDOW_ORDER)
        ax.bar(x + (i - 1) * 0.27, sub["share"], 0.27, color=colours[c], label=c,
               yerr=[sub["share"] - sub["ci_low"], sub["ci_high"] - sub["share"]], capsize=3, ecolor="#777")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace(" sovereign", "\nsovereign").replace(" / ", "/\n") for w in WINDOW_ORDER],
                       fontsize=8)
    ax.set_ylabel("share of |SHAP attribution|")
    ax.legend(frameon=False, title="channel")
    ax.set_title("Transmission fingerprints: which market drives the LSTM's Irish volatility forecast "
                 "(95% block-bootstrap intervals)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "shap_channel_shares.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for w, c in zip(WINDOW_ORDER, ["#9e9e9e", "#1f77b4", "#2ca02c", "#9467bd", "#d62728"]):
        ax.plot(range(22), lags.loc[w].values, marker="o", ms=3, color=c, label=w)
    ax.set_xticks(range(0, 22, 3))
    ax.set_xticklabels([f"t-{k}" for k in range(0, 22, 3)])
    ax.set_xlabel("look-back lag (t-0 = day the forecast is made)")
    ax.set_ylabel("share of |attribution|")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("How far back the model looks, by window", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "shap_lag_profiles.png", dpi=150)
    print(f"\nSaved results/shap_*.csv and figures/shap_*.png  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()

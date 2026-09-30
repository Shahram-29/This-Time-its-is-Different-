"""
11_connectedness.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A1, E1)
Diebold-Yilmaz (2012) spillover measures for ISEQ, S&P 500 and DAX volatility, compared with the SHAP channel
shares from 07. Asks whether an econometric measure of cross-market transmission tells the same story as the
explanations of the LSTM.

Specification (fixed in Amendment A1 before this script was run):
  * variables: log 5-day backward realised variance (mean of the last 5 squared daily returns, floor 1e-8)
    of the three markets, from the aligned dataset the LSTM uses;
  * VAR(4) by OLS on 200-day rolling windows ending at each forecast origin t;
  * generalized forecast-error variance decomposition (Pesaran & Shin, 1998), 10-day horizon, rows normalised;
  * ISEQ row (own, from US, from euro) averaged over the SHAP days; full-sample table; total spillover index;
  * sensitivity: daily log squared returns; 100- and 300-day windows.
Run from the code folder:  py -3.13 11_connectedness.py   (about a minute)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import SAMPLE_END, WINDOWS

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
MARKETS = {"domestic": "r2", "US": "us_r2", "euro": "euro_r2"}      # channel names as in the SHAP results
LAGS, H, FLOOR = 4, 10, 1e-8
WINDOW_ORDER = ["calm", "Global Financial Crisis", "Irish sovereign debt crisis", "COVID-19", "War / energy shock 2022"]
SPECS = {"main": ("rv5", 200), "daily r2": ("r2", 200), "100-day window": ("rv5", 100),
         "300-day window": ("rv5", 300)}


def volatility(measure: str) -> pd.DataFrame:
    """Log volatility of the three markets on the Irish calendar (foreign values aligned as-of day t)."""
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv", index_col=0, parse_dates=True).loc[:SAMPLE_END]
    v = d[list(MARKETS.values())]
    if measure == "rv5":
        v = v.rolling(5).mean()
    v = np.log(v.clip(lower=FLOOR)).dropna()
    v.columns = list(MARKETS)
    return v


def var_ols(Y: np.ndarray, p: int):
    """VAR(p) with a constant by OLS: lag matrices B_1..B_p and the residual covariance."""
    T, k = Y.shape
    X = np.hstack([np.ones((T - p, 1))] + [Y[p - j:T - j] for j in range(1, p + 1)])
    coef = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
    resid = Y[p:] - X @ coef
    sigma = resid.T @ resid / (T - p - X.shape[1])
    B = [coef[1 + (j - 1) * k: 1 + j * k].T for j in range(1, p + 1)]
    return B, sigma


def ma_matrices(B, h):
    """Moving-average matrices A_0..A_{h-1}: A_0 = I, A_s = sum_j B_j A_{s-j}."""
    k, p = B[0].shape[0], len(B)
    A = [np.eye(k)]
    for s in range(1, h):
        A.append(sum(B[j] @ A[s - 1 - j] for j in range(min(p, s))))
    return A


def gfevd(B, sigma, h=H):
    """Generalized FEVD (Pesaran & Shin, 1998), each row normalised to sum to 1 (Diebold & Yilmaz, 2012).
    Row i, column j = share of market i's h-step forecast-error variance due to shocks in market j."""
    A = ma_matrices(B, h)
    num = sum((a @ sigma) ** 2 for a in A) / np.diag(sigma)
    den = sum(np.diag(a @ sigma @ a.T) for a in A)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def rolling(v: pd.DataFrame, width: int) -> pd.DataFrame:
    """ISEQ row and total spillover index from the VAR on the `width` days ending at each date."""
    Y, rows = v.to_numpy(), []
    for i in range(width - 1, len(Y)):
        th = gfevd(*var_ols(Y[i - width + 1: i + 1], LAGS))
        rows.append([*th[0], 100 * (th.sum() - np.trace(th)) / len(th)])
    return pd.DataFrame(rows, index=v.index[width - 1:], columns=[*MARKETS, "total_spillover"])


def spillover_table(v: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Diebold-Yilmaz table (in %): rows receive, columns give."""
    th = 100 * gfevd(*var_ols(v.to_numpy(), LAGS))
    t = pd.DataFrame(th, index=list(MARKETS), columns=list(MARKETS))
    t["from others"] = t.sum(axis=1) - np.diag(th)
    to = t[list(MARKETS)].sum(axis=0) - np.diag(th)
    t.loc["to others"] = [*to, to.sum() / len(MARKETS)]              # corner = total spillover index
    t.loc["net"] = [*(to.to_numpy() - t["from others"].iloc[:3].to_numpy()), np.nan]
    return t.round(2)


def main():
    fc = pd.read_csv(RES / "lstm_forecasts.csv", index_col=0, parse_dates=True)
    calm = fc.index[fc["window"] == "calm"][::10]                    # the SHAP days (07_shap.py)
    days = fc.index[fc["window"].isin(WINDOW_ORDER[1:])].union(calm)
    win = fc.loc[days, "window"]

    shap = pd.read_csv(RES / "shap_channel_shares.csv").set_index(["window", "channel"])["share"]
    compare, agree, roll_main = [], [], None
    for spec, (measure, width) in SPECS.items():
        v = volatility(measure)
        roll = rolling(v, width)
        if spec == "main":
            roll_main, table = roll, spillover_table(v)
            table.to_csv(RES / "connectedness_full_sample.csv")
        dy = roll.reindex(days).groupby(win).mean()
        for w in WINDOW_ORDER:
            for c in MARKETS:
                compare.append({"spec": spec, "window": w, "channel": c, "shap_share": shap[(w, c)],
                                "dy_share": dy.loc[w, c], "shap_change": shap[(w, c)] - shap[("calm", c)],
                                "dy_change": dy.loc[w, c] - dy.loc["calm", c],
                                "total_spillover": dy.loc[w, "total_spillover"]})
        cmp = pd.DataFrame([r for r in compare if r["spec"] == spec]).set_index(["window", "channel"])
        signs = [np.sign(cmp.loc[(w, c), "shap_change"]) == np.sign(cmp.loc[(w, c), "dy_change"])
                 for w in WINDOW_ORDER[1:] for c in ("US", "euro")]
        ranks = [(cmp.loc[(w, "US"), "shap_share"] > cmp.loc[(w, "euro"), "shap_share"]) ==
                 (cmp.loc[(w, "US"), "dy_share"] > cmp.loc[(w, "euro"), "dy_share"]) for w in WINDOW_ORDER]
        agree.append({"spec": spec, "sign_agreement_of_8": int(sum(signs)), "rank_agreement_of_5": int(sum(ranks))})

    compare, agree = pd.DataFrame(compare), pd.DataFrame(agree)
    compare.round(4).to_csv(RES / "connectedness_vs_shap.csv", index=False)
    agree.to_csv(RES / "connectedness_agreement.csv", index=False)
    roll_main.round(4).to_csv(RES / "connectedness_rolling.csv")

    print("Full-sample spillover table (%), main spec:\n" + table.to_string())
    m = compare[compare["spec"] == "main"].pivot(index="window", columns="channel",
                                                 values=["shap_share", "dy_share"]).reindex(WINDOW_ORDER)
    print("\nISEQ shares by window, SHAP vs Diebold-Yilmaz (main spec):\n" + m.round(3).to_string())
    print("\nAgreement:\n" + agree.to_string(index=False))

    # ---- Figures --------------------------------------------------------------------------------------
    colours = {"domestic": "#2e7d32", "US": "#1565c0", "euro": "#ef6c00"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for c in ("US", "euro"):
        a1.plot(roll_main.index, roll_main[c], lw=0.8, color=colours[c], label=f"from {c} to ISEQ")
    a2.plot(roll_main.index, roll_main["total_spillover"], lw=0.8, color="black")
    for ax in (a1, a2):
        for name, (a, b) in WINDOWS.items():
            if name != "Brexit referendum":
                ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.15, lw=0)
    a1.set_ylabel("share of ISEQ forecast-\nerror variance")
    a1.legend(loc="upper left", frameon=False)
    a1.set_title("Diebold-Yilmaz spillovers into ISEQ volatility (200-day rolling VAR(4), 10-day horizon)")
    a2.set_ylabel("total spillover index (%)")
    a2.set_title("Total spillover among ISEQ, S&P 500 and DAX (shaded: evaluation windows)")
    fig.tight_layout()
    fig.savefig(FIG / "connectedness_rolling.png", dpi=150)
    plt.close(fig)

    labels = ["calm", "GFC", "Irish debt", "COVID-19", "2022"]
    x = np.arange(len(WINDOW_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, c in zip(axes, MARKETS):
        ax.bar(x - 0.2, m[("shap_share", c)], 0.4, color=colours[c], label="SHAP (LSTM)")
        ax.bar(x + 0.2, m[("dy_share", c)], 0.4, color=colours[c], alpha=0.45, hatch="//", label="Diebold-Yilmaz")
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(f"{c} channel")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("share of ISEQ volatility explained")
    fig.suptitle("Two views of transmission: SHAP attribution shares vs Diebold-Yilmaz variance shares")
    fig.tight_layout()
    fig.savefig(FIG / "connectedness_vs_shap.png", dpi=150)
    plt.close(fig)
    print("Saved results/connectedness_*.csv and figures/connectedness_*.png")


if __name__ == "__main__":
    main()

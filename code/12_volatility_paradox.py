"""
12_volatility_paradox.py  -  EXPLORATORY (PREREGISTRATION.md, Amendment A1, E2)
Danielsson, Valenzuela & Zer (2018, Review of Financial Studies): long spells of volatility below its trend come
before banking crises ("stability is destabilising"). Checks, one market and one crisis at a time, whether the
study's crises were preceded by unusually calm markets.

Specification (fixed in Amendment A1 before this script was run):
  * OECD monthly share prices from FRED: Ireland (the ISEQ, from 1955), United States (1957), Germany (1960);
    monthly averages of daily closes;
  * monthly log returns winsorised at 0.5% / 99.5%; annual volatility = std of the 12 returns July-June x sqrt(12);
  * one-sided (recursive) Hodrick-Prescott trend, lambda = 5,000, from the 10th annual observation;
  * delta_low = min(sigma - trend, 0); measure = mean delta_low over the five complete July-June years before a
    window starts, and its percentile in the same market's history ("unusually calm" = bottom 20%);
  * cross-check: FRED vs Yahoo annual volatility, 2004-2023.
Year t below means July of t-1 to June of t, as in the paper.
Run from the code folder:  py -3.13 12_volatility_paradox.py   (downloads three small FRED files on first run)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from evaluation import WINDOWS

ROOT = Path(__file__).resolve().parent.parent
RAW, RES, FIG = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
FRED = {"Ireland": "SPASTT01IEM661N", "United States": "SPASTT01USM661N", "Germany": "SPASTT01DEM661N"}
YAHOO = {"Ireland": "ISEQ", "United States": "GSPC", "Germany": "GDAXI"}
LAMBDA, BURN, L, CALM_PCT = 5000, 10, 5, 20
LAST_MONTH = "2023-06-30"                  # year 2023 = Jul 2022 - Jun 2023 (main sample ends Aug 2023)
CRISES = ["Global Financial Crisis", "Irish sovereign debt crisis", "COVID-19"]


def vol_year(idx: pd.DatetimeIndex) -> np.ndarray:
    """July-June year label: July 2006 - June 2007 is year 2007."""
    return idx.year + (idx.month >= 7)


def fred_prices(code: str) -> pd.Series:
    path = RAW / f"FRED_{code}.csv"
    if not path.exists():
        pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}").to_csv(path, index=False)
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna().loc[:LAST_MONTH]


def annual_volatility(prices: pd.Series) -> pd.Series:
    r = np.log(prices).diff().dropna()
    r = r.clip(r.quantile(0.005), r.quantile(0.995))
    g = r.groupby(vol_year(r.index))
    return (g.std() * np.sqrt(12))[g.size() == 12]


def one_sided_trend(sigma: pd.Series) -> pd.Series:
    """HP trend for year t estimated only from years up to t (recursive, as in the paper)."""
    x = sigma.to_numpy()
    trend = [hpfilter(x[:t + 1], lamb=LAMBDA)[1][-1] for t in range(BURN - 1, len(x))]
    return pd.Series(trend, index=sigma.index[BURN - 1:])


def crisis_year(start: str) -> int:
    """First July-June year that contains the window start; the five years before it are the pre-crisis years."""
    s = pd.Timestamp(start)
    return s.year + (s.month >= 7)


def main():
    annual, crises, check = [], [], []
    for market, code in FRED.items():
        sigma = annual_volatility(fred_prices(code))
        trend = one_sided_trend(sigma)
        dev = (sigma - trend).dropna()
        low = dev.clip(upper=0)
        low5 = low.rolling(L).mean().shift(1)                         # mean over years t-5 .. t-1
        low5.loc[low5.index.max() + 1] = low.iloc[-L:].mean()          # also defined for the year after the last
        a = pd.DataFrame({"sigma": sigma, "trend": trend, "delta_low": low, "delta_high": dev.clip(lower=0),
                          "low5_before": low5})
        a.insert(0, "market", market)
        annual.append(a.rename_axis("year").reset_index())

        hist = low5.dropna()
        for w in CRISES:
            t = crisis_year(WINDOWS[w][0])
            v = hist.loc[t]
            pct = 100 * (hist <= v).mean()
            crises.append({"market": market, "window": w, "pre_crisis_years": f"{t - L}-{t - 1}",
                           "low5_before": v, "percentile": pct, "unusually_calm": pct <= CALM_PCT,
                           "history": f"{hist.index.min()}-{hist.index.max()}", "n_years": len(hist)})

        d = pd.read_csv(RAW / f"{YAHOO[market]}.csv", index_col=0, parse_dates=True)["Close"].dropna()
        r = np.log(d).diff().dropna().loc[:LAST_MONTH]
        g = r.groupby(vol_year(r.index))
        y = (g.std() * np.sqrt(252)).loc[2004:2023]
        check.append({"market": market, "years": "2004-2023", "corr_fred_vs_yahoo": sigma.loc[y.index].corr(y),
                      "mean_fred": sigma.loc[y.index].mean(), "mean_yahoo": y.mean()})

    annual, crises, check = pd.concat(annual), pd.DataFrame(crises), pd.DataFrame(check)
    annual.round(5).to_csv(RES / "volatility_paradox_annual.csv", index=False)
    crises.round(4).to_csv(RES / "volatility_paradox_crises.csv", index=False)
    check.round(4).to_csv(RES / "volatility_paradox_crosscheck.csv", index=False)
    print("Pre-crisis low-volatility measure (mean delta_low over the 5 years before each window):\n"
          + crises.round(4).to_string(index=False))
    print("\nCross-check, FRED monthly-average vs Yahoo daily annual volatility:\n" + check.round(3).to_string(index=False))

    # ---- Figure: volatility, trend and the 5-year low-volatility measure --------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    for j, market in enumerate(FRED):
        a = annual[annual["market"] == market].set_index("year")
        ax = axes[0, j]
        ax.plot(a.index, a["sigma"], color="black", lw=1, label="annual volatility")
        ax.plot(a.index, a["trend"], color="#c62828", lw=1.2, label="one-sided HP trend")
        ax.fill_between(a.index, a["trend"], a["sigma"], where=a["delta_low"] < 0, color="#1565c0", alpha=0.3,
                        step=None, label="below trend")
        ax.set_title(market)
        ax2 = axes[1, j]
        h = a["low5_before"].dropna()
        ax2.bar(h.index, h, color="#1565c0", width=0.8)
        ax2.axhline(h.quantile(CALM_PCT / 100), color="grey", ls="--", lw=0.8, label="bottom 20%")
        for w in CRISES:
            t = crisis_year(WINDOWS[w][0])
            for x in (ax, ax2):
                x.axvline(t - 0.5, color="#ef6c00", lw=0.8)
            ax2.annotate({"Global Financial Crisis": "GFC", "Irish sovereign debt crisis": "Irish",
                          "COVID-19": "COVID"}[w], (t - 0.5, h.min()), fontsize=7, rotation=90,
                         va="bottom", ha="right")
    axes[0, 0].set_ylabel("annualised volatility")
    axes[1, 0].set_ylabel("mean below-trend volatility,\nprevious 5 years")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Volatility paradox check (Danielsson, Valenzuela & Zer, 2018): were crises preceded by calm?")
    fig.tight_layout()
    fig.savefig(FIG / "volatility_paradox.png", dpi=150)
    plt.close(fig)
    print("Saved results/volatility_paradox_*.csv and figures/volatility_paradox.png")


if __name__ == "__main__":
    main()

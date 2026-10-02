"""
19_irish_data_profile.py  -  DESCRIPTIVE (no forecasting, no hypothesis test of the study)
In-depth profile of the Irish dataset used in the Irish focus (E8): coverage and quality checks, descriptive
statistics by Kindleberger-Minsky phase, stylised facts (fat tails, volatility clustering, leverage), feature
ranges in the boom vs the crisis, correlations, ISEQ-bank lead-lag, stationarity, and the walk-forward splits.
Feeds docs/Irish_Dataset_Handbook.md.
Run from the code folder:  py -3.13 19_irish_data_profile.py   (seconds)
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests

ROOT = Path(__file__).resolve().parent.parent
RAW, RES, FIG = ROOT / "data" / "raw", ROOT / "results", ROOT / "figures"
spec = importlib.util.spec_from_file_location("s17", Path(__file__).parent / "17_irish_focus.py")
s17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s17)

PERIODS = {"Boom (2003 - 19 Feb 2007)": ("2003-01-02", "2007-02-19"), **s17.PHASES,
           "Whole crisis (P1-P4)": s17.WHOLE[1]}
START, END = "2003-01-02", "2012-12-31"


def phase_label(dates):
    lab = s17.phase_of(dates)
    lab[(dates >= "2003-01-02") & (dates <= "2007-02-19")] = "Boom"
    return lab


def return_stats(r, price):
    r = r.dropna()
    jb = stats.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_r2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(r, nlags=5)[1]
    dd = (price / price.cummax() - 1).min()
    return {"days": len(r), "mean (% a year)": 100 * r.mean() * 252, "volatility (% a year)": 100 * r.std() * np.sqrt(252),
            "skewness": stats.skew(r), "excess kurtosis": stats.kurtosis(r),
            "worst day (%)": 100 * r.min(), "worst day": r.idxmin().date(), "best day (%)": 100 * r.max(),
            "best day": r.idxmax().date(), "max drawdown (%)": 100 * dd, "Jarque-Bera p": jb.pvalue,
            "Ljung-Box Q(10) returns p": lb_r, "Ljung-Box Q(10) squared p": lb_r2, "ARCH-LM(5) p": arch_p}


def main():
    d, x, y, ok = s17.build()
    d = d.loc[START:END]
    iseq = pd.read_csv(RAW / "ISEQ.csv", index_col=0, parse_dates=True).loc["2002-10-01":END]
    banks = {t: pd.read_csv(RAW / f"{t.replace('.', '_')}.csv", index_col=0, parse_dates=True).loc["2002-10-01":END]
             for t in s17.BANKS}

    # ---- 1. coverage and quality ---------------------------------------------------------------------------------
    rows = []
    for name, df, col in [("ISEQ Overall (^ISEQ)", iseq, "Close")] + [(f"{s17.BANKS[t]} ({t})", banks[t], "Adj Close") for t in banks]:
        p = df[col].loc[START:END]
        r = np.log(p).diff().dropna()
        rows.append({"series": name, "first": df.index.min().date(), "last": df.index.max().date(),
                     "days 2003-2012": len(p), "missing prices": int(p.isna().sum()),
                     "duplicated dates": int(df.index.duplicated().sum()),
                     "zero-return days (%)": 100 * (r == 0).mean(),
                     "zero-volume days (%)": 100 * (df["Volume"].loc[START:END] == 0).mean(),
                     "dates not in ISEQ": int((~p.index.isin(iseq.index)).sum()),
                     "ISEQ dates missing here": int((~iseq.loc[START:END].index.isin(p.index)).sum())})
    cov = pd.DataFrame(rows)
    cov.to_csv(RES / "irish_data_coverage.csv", index=False)

    q = []
    io = iseq.loc[START:END]
    prev_close = iseq["Close"].shift(1).loc[START:END]
    for yr in range(2003, 2013):
        m = io.index.year == yr
        rr = np.log(io["Close"]).diff()[m]
        row = {"year": yr, "ISEQ days": int(m.sum()),
               "ISEQ zero-return (%)": 100 * (rr == 0).mean(),
               "ISEQ open = previous close (%)": 100 * (np.abs(io["Open"][m] / prev_close[m] - 1) < 1e-6).mean(),
               "ISEQ high = low (%)": 100 * (io["High"][m] == io["Low"][m]).mean()}
        for t in banks:
            br = np.log(banks[t]["Adj Close"]).diff()
            br = br[br.index.year == yr]
            row[f"{s17.BANKS[t]} zero-return (%)"] = 100 * (br == 0).mean()
        q.append(row)
    qual = pd.DataFrame(q)
    qual.to_csv(RES / "irish_data_quality_by_year.csv", index=False)

    # ---- 2. descriptive statistics by phase ---------------------------------------------------------------------------
    bank_level = np.exp(d["bank_ret"].fillna(0).cumsum())
    rows = []
    for per, (a, b) in PERIODS.items():
        for name, r, price in [("ISEQ", d["ret"], iseq["Close"]), ("Irish banks (BoI + AIB)", d["bank_ret"], bank_level)]:
            st = return_stats(r.loc[a:b], price.loc[a:b])
            st["corr ISEQ-banks"] = d.loc[a:b, ["ret", "bank_ret"]].corr().iloc[0, 1]
            rows.append({"period": per, "series": name, **st})
    ph = pd.DataFrame(rows)
    ph.to_csv(RES / "irish_data_phase_stats.csv", index=False)

    tgt = []
    for per, (a, b) in PERIODS.items():
        t = d.loc[a:b, "rv5_fwd"].dropna()
        tgt.append({"period": per, "days": len(t), "mean 5-day variance (x1e4)": 1e4 * t.mean(),
                    "median (x1e4)": 1e4 * t.median(), "90th percentile (x1e4)": 1e4 * t.quantile(0.9),
                    "max (x1e4)": 1e4 * t.max(), "max date": t.idxmax().date(),
                    "implied annual volatility (%)": 100 * np.sqrt(t.mean() * 252 / 5)})
    tgt = pd.DataFrame(tgt)
    tgt.to_csv(RES / "irish_data_target_stats.csv", index=False)

    lab = phase_label(d.index)
    big = d["ret"].abs().sort_values(ascending=False).head(20).index
    moves = pd.DataFrame({"ISEQ return (%)": 100 * d.loc[big, "ret"], "bank return (%)": 100 * d.loc[big, "bank_ret"],
                          "phase": lab[big]}).sort_index()
    moves.index = moves.index.date
    moves.to_csv(RES / "irish_data_largest_moves.csv", index_label="date")

    # ---- 3. features: boom range vs crisis --------------------------------------------------------------------------
    feats = s17.SETS["B"]
    boom, crisis = x.loc["2003-01-02":"2006-12-31", feats], x.loc[s17.WHOLE[1][0]:s17.WHOLE[1][1], feats]
    crisis_ph = s17.phase_of(crisis.index)
    fr = []
    for c in feats:
        lev = c.startswith("lev")
        out = crisis[c] < boom[c].min() if lev else crisis[c] > boom[c].max()
        row = {"feature": c, "boom mean": boom[c].mean(), "boom sd": boom[c].std(), "boom min": boom[c].min(),
               "boom max": boom[c].max(), "crisis mean": crisis[c].mean(), "crisis min": crisis[c].min(),
               "crisis max": crisis[c].max(), "crisis days outside the boom range (%)": 100 * out.mean()}
        for p in s17.PHASES:
            row[f"outside, {p} (%)"] = 100 * out[crisis_ph == p].mean()
        fr.append(row)
    fr = pd.DataFrame(fr)
    fr.to_csv(RES / "irish_data_features.csv", index=False)
    corr = x.loc[START:"2012-12-31", feats].join(y.rename("target (log RV5)")).corr()
    corr.to_csv(RES / "irish_data_feature_corr.csv")

    # ---- 4. ISEQ-bank lead-lag (descriptive) ----------------------------------------------------------------------------
    lr2 = pd.DataFrame({"iseq": np.log(d["r2"].clip(lower=1e-8)), "bank": np.log(d["bank_r2"].clip(lower=1e-8))})
    ll = []
    for per, (a, b) in [("Boom", ("2003-01-02", "2007-02-19")), ("Crisis P1-P4", s17.WHOLE[1])]:
        z = lr2.loc[a:b].dropna()
        for k in range(-5, 6):
            ll.append({"period": per, "lag k (bank leads ISEQ by k days)": k, "corr": z["iseq"].corr(z["bank"].shift(k))})
        for cause, effect in [("bank", "iseq"), ("iseq", "bank")]:
            g = grangercausalitytests(z[[effect, cause]], maxlag=5)
            ll.append({"period": per, "granger": f"{cause} -> {effect}", "F-test p (5 lags)": g[5][0]["ssr_ftest"][1]})
    ll = pd.DataFrame(ll)
    ll.to_csv(RES / "irish_data_leadlag.csv", index=False)

    # ---- 5. stationarity -------------------------------------------------------------------------------------------
    st_rows = []
    series = {"ISEQ return": d["ret"], "ISEQ log squared return": lr2["iseq"], "target: log RV5 (forward)": y.loc[START:END],
              "bank return": d["bank_ret"], "bank log squared return": lr2["bank"]}
    for name, s in series.items():
        s = s.dropna()
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
        st_rows.append({"series": name, "ADF statistic": adf[0], "ADF p": adf[1], "KPSS statistic": kp[0], "KPSS p": kp[1]})
    stn = pd.DataFrame(st_rows)
    stn.to_csv(RES / "irish_data_stationarity.csv", index=False)

    # ---- 6. walk-forward splits -------------------------------------------------------------------------------------------
    sp = []
    for yr in s17.YEARS:
        fit, val, test = s17.days_for(d, ok, yr)
        sp.append({"test year": yr, "fitting days": len(fit), "fitting from": fit[0].date(), "fitting to": fit[-1].date(),
                   "validation days": len(val), "test days": len(test)})
    sp = pd.DataFrame(sp)
    sp.to_csv(RES / "irish_data_splits.csv", index=False)

    pd.set_option("display.width", 250)
    for title, t in [("Coverage", cov), ("Quality by year", qual.round(1)), ("Phase statistics", ph.round(3)),
                     ("Target", tgt.round(2)), ("Largest ISEQ moves", moves.round(1)), ("Features", fr.round(2)),
                     ("Lead-lag", ll.round(3)), ("Stationarity", stn.round(3)), ("Splits", sp)]:
        print(f"\n== {title} ==\n{t.to_string()}")
    plots(d, iseq, lr2, fr, corr, ll)


def plots(d, iseq, lr2, fr, corr, ll):
    colours = {"P1 Distress": "#fde0c5", "P2 Panic and guarantee": "#f4a582", "P3 Relief": "#d1e5f0",
               "P4 Sovereign crisis": "#e7d4e8"}

    def shade(ax):
        ax.axvspan(pd.Timestamp("2003-01-02"), pd.Timestamp("2007-02-19"), color="#e5f5e0", lw=0)
        for name, (s, e) in s17.PHASES.items():
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=colours[name], lw=0)

    # overview
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    p = iseq["Close"].loc["2003":"2012"]
    bank = np.exp(d["bank_ret"].fillna(0).cumsum())
    shade(axs[0])
    axs[0].plot(p.index, 100 * p / p.iloc[0], color="black", lw=1, label="ISEQ Overall")
    axs[0].plot(bank.index, 100 * bank / bank.iloc[0], color="tab:red", lw=1, label="Irish banks (BoI + AIB, equal weight)")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("index, 2 Jan 2003 = 100 (log)")
    axs[0].legend(fontsize=8, loc="lower left")
    axs[0].set_title("The Irish dataset, 2003-2012: boom (green) and crisis phases P1-P4", fontsize=10)
    for ax, col, name, c in [(axs[1], "ret", "ISEQ daily return (%)", "black"), (axs[2], "bank_ret", "bank daily return (%)", "tab:red")]:
        shade(ax)
        ax.plot(d.index, 100 * d[col], color=c, lw=0.5)
        ax.set_ylabel(name)
    axs[2].set_ylim(-60, 45)
    fig.tight_layout()
    fig.savefig(FIG / "irish_data_overview.png", dpi=150)

    # distribution and clustering
    r = d["ret"].dropna()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    z = (r - r.mean()) / r.std()
    axs[0].hist(z, bins=120, density=True, color="#888", alpha=0.8, label="ISEQ daily returns (standardised)")
    g = np.linspace(-8, 8, 400)
    axs[0].plot(g, stats.norm.pdf(g), color="tab:blue", label="normal")
    axs[0].set_xlim(-8, 8)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-4, 1)
    axs[0].legend(fontsize=7, loc="upper left")
    axs[0].set_title("Fat tails (log density)", fontsize=10)
    lags = np.arange(1, 61)
    axs[1].bar(lags - 0.2, acf(r, nlags=60)[1:], 0.4, color="#999", label="returns")
    axs[1].bar(lags + 0.2, acf(r ** 2, nlags=60)[1:], 0.4, color="tab:blue", label="squared returns")
    axs[1].axhline(1.96 / np.sqrt(len(r)), color="k", lw=0.6, ls="--")
    axs[1].axhline(-1.96 / np.sqrt(len(r)), color="k", lw=0.6, ls="--")
    axs[1].legend(fontsize=7)
    axs[1].set_xlabel("lag (days)")
    axs[1].set_title("Volatility clustering (autocorrelation)", fontsize=10)
    nxt = d["rv5_fwd"].dropna()
    falls = d.loc[nxt.index, "ret"]
    bins = pd.qcut(falls, 10, labels=False)
    axs[2].bar(np.arange(10), [1e4 * nxt[bins == k].mean() for k in range(10)], color="tab:purple")
    axs[2].set_xticks(np.arange(10), [f"{k + 1}" for k in range(10)], fontsize=8)
    axs[2].set_xlabel("today's ISEQ return decile (1 = biggest falls)")
    axs[2].set_ylabel("next 5 days' variance (x1e4)")
    axs[2].set_title("Leverage: falls predict more volatility", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "irish_data_stylised_facts.png", dpi=150)

    # features: outside the boom range, and correlations
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.3, 0.9]})
    phs = list(s17.PHASES)
    xx = np.arange(len(fr))
    for i, p in enumerate(phs):
        axs[0].bar(xx + (i - 1.5) * 0.2, fr[f"outside, {p} (%)"], 0.2, color=list(colours.values())[i], edgecolor="#555",
                   label=p)
    axs[0].set_xticks(xx, fr["feature"], rotation=45, fontsize=8)
    axs[0].set_ylabel("% of days outside the 2003-06 range")
    axs[0].set_title("Crisis values never seen in the boom", fontsize=10)
    axs[0].legend(fontsize=7)
    im = axs[1].imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_xticks(range(len(corr)), corr.columns, rotation=60, fontsize=7)
    axs[1].set_yticks(range(len(corr)), corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            axs[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    axs[1].set_title("Feature correlations, 2003-2012", fontsize=10)
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    cc = ll.dropna(subset=["corr"])
    for per, c in [("Boom", "#2ca25f"), ("Crisis P1-P4", "#de2d26")]:
        t = cc[cc.period == per]
        axs[2].plot(t["lag k (bank leads ISEQ by k days)"], t["corr"], marker="o", color=c, label=per)
    axs[2].set_xlabel("k: bank log r2 shifted by k days\n(k > 0: banks lead the ISEQ)")
    axs[2].set_ylabel("correlation with ISEQ log r2")
    axs[2].set_title("ISEQ-bank lead-lag", fontsize=10)
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "irish_data_features.png", dpi=150)


if __name__ == "__main__":
    main()

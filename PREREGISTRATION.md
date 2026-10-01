# Pre-registration — "This Time Is Different?"

> **DRAFT — prepared with AI assistance on 24 Sep 2026 from the agreed research plan (v1.3).**
> Before committing: read every line, rewrite the wording in your own words, change anything you do not agree with,
> then commit this file to Git **before running any model**. The commit date is your proof that the hypotheses
> came before the results. After that commit, any change must be recorded under "Amendments" with a date and reason.

## Study
- **Title:** This Time Is Different? Explainable Deep Learning of Crisis Transmission into Irish Stock Market
  Volatility: The Global Financial Crisis, the Irish Sovereign Debt Crisis and COVID-19
- **Design:** quantitative, deductive, longitudinal time-series study of secondary public market data.

## Data (fixed)
- Source: Yahoo Finance via `yfinance`; raw downloads cached in `data/raw/`; download end date 31 Dec 2025.
- Markets: ISEQ Overall (`^ISEQ`, target and domestic channel), S&P 500 (`^GSPC`, US channel),
  DAX (`^GDAXI`, euro channel). FTSE 100 (`^FTSE`) is used **only** in the robustness run that replaces DAX.
- Main sample: 2 Jan 2003 – 31 Aug 2023. Sep 2023 – Dec 2025 is used only for a robustness check
  (ISEQ composition break: CRH, Flutter, Smurfit).
- Target: forward 5-day realised variance of the ISEQ, `rv5_fwd(t) = r²(t+1) + … + r²(t+5)`, modelled in logs.
- Model inputs per day (6): `ret, r2, us_ret, us_r2, euro_ret, euro_r2`. Foreign values are aligned as-of day t
  (never t+1). Crisis labels are never model inputs.

## Evaluation windows (fixed before results)
| Window | Dates | Role |
|---|---|---|
| Global Financial Crisis | 9 Aug 2007 – 9 Mar 2009 | core |
| Irish sovereign debt crisis | 23 Apr 2010 – 26 Jul 2012 | core |
| COVID-19 | 19 Feb 2020 – 30 Jun 2020 | core |
| War / energy shock | 10 Feb 2022 – 31 Oct 2022 | validation (main model) |
| Brexit referendum | 1 Jun 2016 – 31 Jul 2016 | validation (FTSE robustness run only) |
| Calm | all other days | comparison |

## Models (fixed)
- GARCH(1,1), Student-t errors, on daily ISEQ returns; 5-day forecast = sum of 1–5-step variance forecasts.
- HAR on log daily, weekly (5-day) and monthly (22-day) average squared returns (OLS).
- LSTM (PyTorch): look-back 22 days (66 as a tuning alternative), 1–2 layers of 32 units (64 alternative),
  dropout 0.2, Adam (lr 1e-3), MSE on log target, early stopping on the last 12 months of each training window.
- Walk-forward: expanding window, first training Jan 2003 – Dec 2006, refit once per calendar year, test years
  2007 – Aug 2023. LSTM: 5 random seeds per refit; the mean forecast is evaluated.
- Scalers are fitted on each training window only.
- Tuning: only the small grid above, chosen on validation loss in the first windows, then frozen.

## Research questions and hypotheses
**RQ1 — Accuracy.** Does the LSTM forecast Irish volatility better than GARCH(1,1) and HAR?
- H1: the LSTM has lower average QLIKE than each benchmark out-of-sample (2007 – Aug 2023).
- Test: Diebold–Mariano with Newey–West variance (4 lags) and the Harvey–Leybourne–Newbold correction, 5% level.
  Per-window comparisons are reported descriptively (small samples).

**RQ2 — Fingerprints.** Do grouped SHAP attributions change across crises in line with each crisis's origin?
Measure: share of total absolute attribution per channel (domestic, US, euro), per window, with block-bootstrap
95% confidence intervals. "Rises" means the window share's interval lies above the calm-period share.
- H2a GFC: the US share rises most; the domestic share is also above calm.
- H2b Irish sovereign debt crisis: euro and domestic shares rise; the US share is below its GFC level.
- H2c COVID-19: all three shares move towards equality (no single channel dominates).
- H2e 2022 shock: the euro share rises.
- H2d Brexit (FTSE robustness run): the UK share rises. (Competing evidence: Li, 2020.)

**RQ3 — Learning from history (Minsky test).** Does a model trained only on earlier history cope with a new crisis?
Frozen LSTMs trained to 8 Aug 2007, 22 Apr 2010 and 18 Feb 2020 are compared with the annually refitted LSTM.
Measure: QLIKE of the frozen model divided by QLIKE of HAR, per crisis.
- H3: the degradation ranking is GFC (largest) > COVID > Irish sovereign debt crisis (smallest).

**RQ4 — Robustness of explanations.** Are channel rankings stable across the 5 seeds?
- H4: mean pairwise Spearman correlation of channel shares across seeds ≥ 0.8 in every window.

## Robustness checks (fixed)
FTSE in place of DAX · Parkinson range volatility (2007–2023) · Sep 2023 – Dec 2025 extension ·
look-back 66 days.

## Interpretation rule
SHAP attributions describe what drives the model's forecasts, not causes. Results will be described as
"transmission fingerprints consistent with …", never as proof of contagion. Results that contradict a
hypothesis will be reported in full.

## Amendments
_(date — change — reason)_

### A1 — 30 Sep 2026 — two exploratory analyses, added after the main results were known
> **DRAFT prepared with AI assistance — rewrite in your own words.**

These analyses are **exploratory**. They were chosen after the RQ1–RQ4 results had been seen, so they cannot
confirm or rescue any hypothesis, and they will be reported in a separate "Exploratory extensions" section.
Their specification is fixed here, before either analysis is run.

**E1 — Does an econometric spillover measure tell the same story as SHAP?** (Diebold & Yilmaz, 2012)
- Variables: log 5-day backward realised variance (mean of the last 5 squared daily returns, floor 1e-8) of the
  ISEQ, S&P 500 and DAX, from the same aligned dataset the LSTM uses.
- VAR(4) by OLS on 200-day rolling windows ending at each forecast origin t; generalized forecast-error variance
  decomposition (Pesaran & Shin, 1998) at a 10-day horizon, rows normalised to sum to 1.
- Measure: the ISEQ row (own share, share from the US, share from the DAX), averaged over the same days that were
  explained with SHAP (every day in each window, every 10th calm day). Also reported: the full-sample spillover
  table and the total spillover index.
- Comparison with the SHAP channel shares (descriptive, no test): (i) for the 4 crisis windows × 2 foreign
  channels, does the change from calm have the same sign in both measures (count out of 8)? (ii) in each of the
  5 windows (calm included), do both measures rank the US and euro channels the same way (count out of 5)?
- Sensitivity: daily log squared returns instead of 5-day realised variance; rolling windows of 100 and 300 days.
- Caveat: a variance decomposition of a linear model and the attributions of a non-linear network measure
  different things. Disagreement is a finding, not an error.

**E2 — Were the crises preceded by unusually calm markets? (volatility paradox; Danielsson, Valenzuela & Zer, 2018)**
- Data: OECD monthly share-price indices from FRED for Ireland (from 1955), the United States (1957) and Germany
  (1960). They are monthly averages of daily closes; the Irish series is the ISEQ (monthly-return correlation
  0.9996 with the Yahoo ISEQ, 2002–2025). A long history is needed to estimate a trend; the Yahoo data is too short.
- Method as in the paper: monthly log returns winsorised at 0.5% / 99.5%; annual volatility = standard deviation
  of the 12 monthly returns from July to June × √12; one-sided (recursive) Hodrick–Prescott trend with
  λ = 5,000, starting at the 10th annual observation; δlow = min(σ − trend, 0).
- Deviations from the paper: nominal instead of real returns; monthly averages instead of month-end prices;
  single markets, so no panel logit.
- Measure: the mean of δlow over the five complete July–June years before each window starts (GFC: 2003–07;
  Irish sovereign debt crisis: 2005–09; COVID-19: 2015–19), and its percentile within the same market's history.
  "Unusually calm" = bottom 20%.
- Expectation (Minsky; Danielsson et al.): the GFC, a credit-driven crisis, was preceded by unusually calm
  markets, at least in Ireland and the US; COVID-19, an external shock, was not. One observation per market
  and crisis, so descriptive only.
- Cross-check: annual volatility from FRED vs from the Yahoo daily closes, 2004–2023 (correlation).

### A2 — 1 Oct 2026 — classical machine-learning benchmarks (Random Forest, SVR), added after the main results were known
> **DRAFT prepared with AI assistance — rewrite in your own words.**

This analysis is **exploratory**. It was chosen after the RQ1–RQ4 results, and after diagnostic checks of the LSTM's
losses, had been seen: most of the LSTM's disadvantage came from 2007–2010, and it reacted weakly to shocks larger
than any in its training data. It cannot confirm or rescue H1 and will be reported with the other exploratory
extensions. Its specification is fixed here, before any Random Forest or SVR forecast for a test year (2007–2023)
has been produced; the code was checked on 2003–2006 data only.

**E3 — Do the classical ML methods of the MSc module beat the LSTM, and do they share its weakness in unseen crises?**
- Models (scikit-learn): `RandomForestRegressor(max_features='sqrt', random_state=100)` and `SVR`.
- Inputs (9): logs of the daily, weekly (mean of 5 days) and monthly (mean of 22 days) squared returns of the ISEQ,
  S&P 500 and DAX, from the same aligned dataset (floor 1e-8). Target: log forward 5-day realised variance.
- Walk-forward exactly as for the LSTM. For test year Y (2007–2023): fitting days up to 31 Dec of Y−2, validation
  year Y−1 (the last 5 days of each part dropped), used only for the Duan smearing factor; forecasts for every day
  of Y. SVR inputs are standardised with the mean and standard deviation of that year's fitting days.
- Tuning once, on the first window (fitting days 2003–2005), with `GridSearchCV`, `cv = TimeSeriesSplit(5 folds,
  gap 5)`, scoring = negative MSE of the log target; the chosen settings are then frozen for all years.
  - Random Forest grid: n_estimators {200, 500} × max_depth {3, 5, 10, None} × min_samples_leaf {1, 5, 20, 50}.
  - SVR grid: kernel {linear, rbf} × C {0.1, 1, 10} × epsilon {0.05, 0.1, 0.3} × gamma {scale, 0.01}.
- Evaluation as for RQ1: QLIKE (primary) and MSE on all test days and per window; DM-HLN tests (QLIKE) of the Random
  Forest and the SVR against GARCH, HAR and the LSTM on all test days. Per-window comparisons are descriptive.
- Expectations stated in advance: (i) the Random Forest has a higher QLIKE than the LSTM in the GFC window, because
  a forest cannot forecast above the largest value in its training data; (ii) neither the Random Forest nor the SVR
  has a lower all-days QLIKE than GARCH.
- Also reported (descriptive): Random Forest impurity importance summed by channel (domestic / US / euro). It is
  not a SHAP value, so it is set beside the SHAP channel shares, not tested against them.

Disclosure: a post-hoc check with an ISEQ-only LSTM (inputs ret and r2, same settings) was run on 1 Oct 2026,
before this amendment. It is not covered by A2 and, if used, will be reported as post-hoc.

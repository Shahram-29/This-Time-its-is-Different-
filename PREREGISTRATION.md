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

### A3 — 1 Oct 2026 — XGBoost, GARCH-hybrid models, extra inputs (VIX, leverage) and a forecast combination, added after the E3 results were known
> **DRAFT prepared with AI assistance — rewrite in your own words.**

These analyses are **exploratory**. They were chosen after the RQ1–RQ4 results and the E3 results (Random Forest and
SVR) had been seen, and after a literature check (Christensen et al., 2023; Branco, Rubesam & Zevallos, 2024;
Audrino & Chassot, 2024; Kim & Won, 2018; Brini, 2026). They cannot confirm or rescue H1. Their specification is
fixed here, before any of the models below has produced a forecast for a test year. The only data checked before
this amendment was availability: the VIX is on Yahoo from 1999; the VSTOXX is not, so it is not used.

**E4 — Can hybrid, boosted or better-informed models do better, and does the "unseen crisis" weakness remain?**

Common rules (as in A2): walk-forward over test years 2007–2023; for test year Y, fitting days up to 31 Dec of Y−2,
validation year Y−1 (the last 5 days of each part dropped) used only for the Duan smearing factor; SVR inputs
standardised on the fitting days; each model tuned once on the first window (fitting days 2003–2005) with
`GridSearchCV`, `TimeSeriesSplit(5 folds, gap 5)`, negative MSE, then frozen. Random Forest and SVR grids as in A2.
XGBoost (`xgboost`, `objective='reg:squarederror'`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=100`):
grid n_estimators {200, 500} × max_depth {2, 3, 5} × learning_rate {0.03, 0.1} × min_child_weight {1, 10}.

1. **XGBoost** with the 9 A2 inputs.
2. **GARCH hybrids (Random Forest, SVR, XGBoost).** Target = log(RV5) − log(GARCH 5-day forecast); inputs = the 9 A2
   inputs + log GARCH forecast; forecast = GARCH × exp(prediction) × smearing factor. The GARCH forecasts come from
   the main study's GARCH(1,1)-t refit for year Y (data to 31 Dec of Y−1). For test days they equal the main study's
   GARCH forecasts; for training days the variance recursion uses only past returns, but the parameters are
   estimated in-sample (stated as a limitation).
3. **Extra inputs.** Log VIX close (Yahoo `^VIX`, joined backward on calendar date like the US and DAX series) and
   the leverage terms of Corsi & Renò (2012): the negative part of the ISEQ return, min(r, 0), on the day and
   averaged over 5 and 22 days (not logged).
   - **HAR-X:** log HAR (daily, weekly, monthly) + log VIX + 3 leverage terms; OLS on all days before Y (last 5
     dropped), smearing from in-sample residuals, exactly as the main HAR.
   - **Random Forest-X, SVR-X, XGBoost-X:** the 9 A2 inputs + log VIX + 3 leverage terms (13 inputs).
4. **Forecast combination:** the equal-weight average of the saved GARCH and LSTM forecasts (variance units).

Evaluation: QLIKE (primary) and MSE, all test days and per window.
- **Primary comparison:** each GARCH hybrid vs GARCH, all test days, QLIKE, DM-HLN two-sided, with a Holm
  adjustment across the three tests. No direction is predicted.
- Secondary (unadjusted, descriptive): every new model vs GARCH, HAR and the LSTM, all test days.
- Expectations stated in advance: (i) XGBoost does not beat GARCH on all test days, and its GFC QLIKE is above
  HAR's (0.676); (ii) each hybrid has a lower GFC QLIKE than the same learner without the hybrid target; (iii) HAR-X
  has a lower all-days QLIKE than HAR (0.414); (iv) Random Forest-X, SVR-X and XGBoost-X still have a GFC QLIKE
  above HAR's (0.676); (v) the GARCH–LSTM average has a lower all-days QLIKE than the LSTM (0.463).

### A4 — 1 Oct 2026 — TreeSHAP of the GARCH correction and a Model Confidence Set, added after the E4 results were known
> **DRAFT prepared with AI assistance — rewrite in your own words.**

These analyses are **exploratory**. They were chosen after the RQ1–RQ4 and E1–E4 results had been seen. They cannot
confirm or rescue any hypothesis. Their specification is fixed here, before either analysis is run.

**E5 — What does GARCH miss? TreeSHAP of the hybrid's correction** (Lundberg et al., 2020)
- Model: the Random Forest GARCH hybrid of E4 (primary); the XGBoost GARCH hybrid as a check. For each test year
  the model is refitted exactly as in E4 (settings from `e4_tuning.csv`); the refitted forecasts must equal the
  saved E4 forecasts.
- Explained quantity: the model's predicted correction log(RV5) − log(GARCH forecast), in logs, before smearing.
- Method: `shap.TreeExplainer`, interventional, background = 200 fitting days of that year's model drawn at random
  (`numpy.random.default_rng(1000 × year)`). Additivity is checked on every explained day.
- Days: the same days as the LSTM SHAP analysis (every day in the GFC, Irish, COVID-19 and 2022 windows; every
  10th calm day).
- Channels: domestic (ISEQ daily, weekly, monthly), US (S&P 500, same three), euro (DAX, same three), and the GARCH
  input (log GARCH forecast). Shares as in step 07: the channel's sum of |SHAP| over the window's days divided by the
  total; 95% intervals from the same block bootstrap (blocks of 10 explained days, 1,000 draws, seed 0). Reported
  with all four channels, and with the three information channels only (GARCH input left out) for comparison with
  the LSTM.
- Expectations stated in advance:
  (i) GARCH uses only the ISEQ's own returns, so in each of the four crisis windows the foreign share (US + euro,
  three-channel version) of the correction is larger than the LSTM's foreign share in that window;
  (ii) applying the H2 rules to the correction ("rises" = interval above the calm share): the US share rises in the
  GFC and in COVID-19, the euro share rises in the Irish sovereign debt crisis and in 2022 (count out of 4).
- Also reported (descriptive, as in E1): agreement with the LSTM SHAP shares — same sign of change from calm (out
  of 8) and same US-vs-euro ranking (out of 5).

**E6 — Which models are statistically best? Model Confidence Set** (Hansen, Lunde & Nason, 2011)
- Models: the 14 compared in E4 (GARCH, HAR, LSTM, Random Forest, SVR, XGBoost, the three GARCH hybrids, HAR-X, the
  three -X learners, the GARCH–LSTM average), on all 4,230 test days.
- Losses: daily QLIKE (primary) and MSE (secondary).
- Method: `arch.bootstrap.MCS`, range statistic (`method='R'`), stationary bootstrap, average block length 10,
  10,000 draws, seed 2026, size 0.10 (the 90% Model Confidence Set). MCS p-values are reported for every model.
- Secondary (descriptive): the same QLIKE MCS in each window (calm, GFC, Irish sovereign debt crisis, COVID-19,
  2022). Sensitivity for all days, QLIKE: block lengths 5 and 22; the max statistic.
- Expectations stated in advance: (i) GARCH is in the 90% MCS on all days under QLIKE; (ii) the LSTM, Random Forest,
  SVR and XGBoost (plain versions) are not; (iii) under MSE, HAR-X is not.

### A5 — 1 Oct 2026 — the Irish sovereign debt crisis with ISEQ-only models, added after the E1–E6 results were known
> **DRAFT prepared with AI assistance — rewrite in your own words.**

This analysis is **exploratory and post-hoc**. It focuses on the domestic crisis and asks whether the foreign inputs
matter there. Some results are already known: GARCH and HAR use only ISEQ data (Irish-window QLIKE 0.374 and 0.380),
and a post-hoc ISEQ-only LSTM check was run on 1 Oct 2026 before A2 (Irish-window QLIKE 0.418). The specification
below is fixed before any other ISEQ-only model is run.

**E7 — Which model is best in the Irish sovereign debt crisis when only ISEQ data is used?**
- Window: the pre-registered Irish sovereign debt crisis window (23 Apr 2010 – 26 Jul 2012, 574 test days).
  Forecasts come from the usual walk-forward refits for the test years 2010, 2011 and 2012; each refit uses only data
  before its test year. Data after the crisis (the files run to Dec 2025) cannot enter these forecasts without
  look-ahead, so the sample end does not change them.
- Inputs: ISEQ only. The 14 models of E4, each in an ISEQ-only version:
  - GARCH and HAR (unchanged; saved forecasts);
  - LSTM with inputs ret and r2 (settings 22/64/2 and seeds 1–5 as in the main study; the configuration was tuned with
    six inputs and is not re-tuned);
  - Random Forest, SVR and XGBoost with the 3 ISEQ HAR inputs;
  - their GARCH hybrids (3 ISEQ HAR inputs + log GARCH forecast; target log(RV5) − log(GARCH));
  - HAR-L and Random Forest-L, SVR-L, XGBoost-L: the ISEQ HAR inputs + the 3 Corsi–Renò leverage terms (the VIX is
    left out, because it is a US series);
  - the equal-weight average of GARCH and the ISEQ-only LSTM.
- Rules as in A2–A3: fitting days to 31 Dec of Y−2, validation year Y−1 for smearing, SVR inputs standardised;
  learners tuned once on the first window (fitting days 2003–2005) with `GridSearchCV`, `TimeSeriesSplit(5, gap 5)`,
  negative MSE and the A2–A3 grids; HAR-L as HAR. LSTM as in the main study.
- Evaluation, Irish window only: QLIKE (primary) and MSE. **Primary:** the 90% Model Confidence Set over the 14
  ISEQ-only models, QLIKE, with the E6 settings. Secondary (descriptive): DM-HLN of each model against GARCH; each
  ISEQ-only model against its three-market version from E3–E4.
- Expectations stated in advance: (i) GARCH is in the 90% MCS; (ii) no ISEQ-only model has a significantly lower
  QLIKE than GARCH (DM-HLN, 5%); (iii) each ISEQ-only plain learner (LSTM, Random Forest, SVR, XGBoost) has an
  Irish-window QLIKE no higher than its three-market version.

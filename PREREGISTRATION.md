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

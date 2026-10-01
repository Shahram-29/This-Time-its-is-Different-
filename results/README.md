# Results

| File | Produced by | Contents |
|---|---|---|
| `benchmark_forecasts.csv` | `04_benchmarks.py` | One row per test day (2007 – Aug 2023): realised 5-day variance, HAR, GARCH and naive forecasts, window |
| `benchmark_losses.csv` | `04_benchmarks.py` | Mean QLIKE and MSE by model, all test days and each window |
| `benchmark_parameters.csv` | `04_benchmarks.py` | HAR coefficients and R²; GARCH ω, α, β, persistence, Student-t ν — per annual refit |
| `lstm_tuning.csv` | `05_lstm_walkforward.py` | Validation loss for every configuration, year and seed in the pre-registered grid |
| `lstm_training_log.csv` | `05_lstm_walkforward.py` | Epochs, validation loss, smearing factor and run time per refit year and seed |
| `lstm_forecasts.csv` | `05_lstm_walkforward.py` | LSTM forecast (mean of 5 seeds) and each seed's forecast, per test day |
| `evaluation_losses.csv` | `06_evaluation.py` | Mean QLIKE and MSE for LSTM, GARCH, HAR, naive, by window |
| `evaluation_dm_tests.csv` | `06_evaluation.py` | Diebold–Mariano–HLN tests (QLIKE and MSE), LSTM vs each benchmark, all days and per window |
| `evaluation_frozen.csv` | `06_evaluation.py` | Frozen-at-crisis-start LSTMs vs refitted LSTM and HAR (RQ3) |
| `shap_*.csv` | `07_shap.py` | Channel shares with intervals, lag profiles, input types, per-seed shares, stability, H2/H4 verdicts |
| `robust_*.csv`, `robustness_summary.csv` | `08_robustness.py` | Each robustness check and a summary against the main result |
| `connectedness_full_sample.csv` | `11_connectedness.py` | Exploratory E1: full-sample Diebold–Yilmaz table (%), rows receive, columns give; to/from/net and total index |
| `connectedness_rolling.csv` | `11_connectedness.py` | ISEQ row (own, from US, from euro) and total spillover index, 200-day rolling, per day |
| `connectedness_vs_shap.csv`, `connectedness_agreement.csv` | `11_connectedness.py` | SHAP vs Diebold–Yilmaz shares by window and channel, changes from calm, agreement counts, all four specifications |
| `volatility_paradox_annual.csv` | `12_volatility_paradox.py` | Exploratory E2: annual volatility, one-sided HP trend, below/above-trend parts, 5-year below-trend mean (Ireland, US, Germany) |
| `volatility_paradox_crises.csv`, `volatility_paradox_crosscheck.csv` | `12_volatility_paradox.py` | Pre-crisis measure and percentile per market and window; FRED vs Yahoo volatility check |
| `ml_tuning.csv` | `13_ml_benchmarks.py` | Exploratory E3: mean cross-validated MSE (log target) for every Random Forest and SVR setting in the grid (tuned on 2003–05) |
| `ml_forecasts.csv`, `ml_training_log.csv` | `13_ml_benchmarks.py` | Random Forest and SVR forecasts per test day; smearing factor and largest forecast vs actual per year |
| `ml_losses.csv`, `ml_dm_tests.csv` | `13_ml_benchmarks.py` | QLIKE and MSE of all five models by window; DM-HLN tests of Random Forest and SVR vs GARCH, HAR, LSTM |
| `ml_importance_by_channel.csv` | `13_ml_benchmarks.py` | Random Forest impurity importance summed by channel (domestic / US / euro), per refit year |
| `e4_tuning.csv` | `14_hybrid_boosting.py` | Exploratory E4: mean cross-validated MSE for every setting of XGBoost, the three GARCH hybrids and the three -X learners (tuned on 2003–05) |
| `e4_forecasts.csv`, `e4_training_log.csv` | `14_hybrid_boosting.py` | Forecasts of all 14 compared models per test day; smearing factors, largest forecasts, HAR-X VIX and leverage coefficients per year |
| `e4_losses.csv`, `e4_dm_tests.csv` | `14_hybrid_boosting.py` | QLIKE and MSE by window; DM-HLN tests (primary family Holm-adjusted: hybrids vs GARCH; secondary unadjusted) |

Variances are in decimal units (e.g. 0.0004 over 5 days). Annualised volatility = √(variance × 252 / 5).
QLIKE = RV/F − ln(RV/F) − 1 (Patton, 2011); lower is better.

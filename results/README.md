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
| `e5_correction_shares.csv` | `15_treeshap_mcs.py` | Exploratory E5: channel shares of \|TreeSHAP\| of the GARCH hybrids' correction by window (4- and 3-channel versions, 95% block-bootstrap intervals; Random Forest and XGBoost hybrids) |
| `e5_checks.csv` | `15_treeshap_mcs.py` | E5 expectations and agreement counts with the LSTM SHAP shares |
| `e6_mcs.csv` | `15_treeshap_mcs.py` | Exploratory E6: Model Confidence Set p-values and 90% membership for 14 models — QLIKE and MSE on all days, sensitivity runs, windows |
| `e7_tuning.csv` | `16_irish_iseq_only.py` | Exploratory E7: cross-validated MSE for every setting of the nine ISEQ-only learners (tuned on 2003–05) |
| `e7_forecasts.csv` | `16_irish_iseq_only.py` | Forecasts of the 14 ISEQ-only models for the test years 2010–2012 |
| `e7_irish_results.csv` | `16_irish_iseq_only.py` | Irish-window QLIKE and MSE, DM-HLN vs GARCH, three-market QLIKE, MCS p-value and 90% membership per model |
| `e8_tuning.csv` | `17_irish_focus.py` | Irish focus E8: cross-validated MSE for every setting of the 8 learners (Sets A and B; tuned on 2003–05) |
| `e8_forecasts.csv` | `17_irish_focus.py` | Forecasts of the 12 models, refitted and frozen (estimated to 2006 only), every test day 2007–2012, with the phase |
| `e8_results.csv` | `17_irish_focus.py` | Per phase and model: QLIKE, MSE, DM-HLN vs GARCH, MCS p-value and 90% membership, frozen QLIKE and frozen/refit ratio |
| `e8_bank_tests.csv` | `17_irish_focus.py` | DM-HLN of each learner with bank inputs (Set B) vs ISEQ only (Set A), per phase |
| `e8_treeshap_groups.csv`, `e8_treeshap_features.csv` | `17_irish_focus.py` | TreeSHAP of Random Forest-B and RF-hybrid-B on every crisis day: group shares with 95% intervals; mean \|SHAP\| per feature |
| `e8_lstm_shap.csv`, `e8_lstm_shap_lags.csv` | `18_irish_focus_shap.py` | GradientShap of LSTM-B: ISEQ vs bank and direction vs size shares per phase (95% intervals, seed range); lag profile |

Variances are in decimal units (e.g. 0.0004 over 5 days). Annualised volatility = √(variance × 252 / 5).
QLIKE = RV/F − ln(RV/F) − 1 (Patton, 2011); lower is better.

# The simple notebook series — start here

The whole project is split into **27 small notebooks**, one step each, written at the level of the MSc
machine-learning class notebooks: `read_csv`, `head()`, `info()`, `describe()`, simple column calculations,
`print(..., round(...))`, plain `for` loops, and an explanation before every cell. Open them in order. Each
notebook ends with outputs already shown, so you can read it before running it.

## Set-up (once)
1. Upload the folder **`ThisTimeIsDifferent`** to Google Drive, inside **`MyDrive/DataSets/`**. On the laptop it
   is `colab/DataSets_for_Drive/ThisTimeIsDifferent`. It holds:
   * the price files;
   * `results/`, the saved results of the study;
   * `work/`, the files the notebooks make.

   `work/` is already filled, so every notebook can also run on its own.
2. In Colab: **File → Upload notebook**, choose a notebook, then **Runtime → Run all**.

## The notebooks

| # | Notebook | What you learn / do | Class notebook it follows |
|---|---|---|---|
| **Part 1** | **The data** | | |
| 01 | Data Reading | read the four price files; `head`, `tail`, `shape`, `info`, `describe` | Data_Reading |
| 02 | Date and Time | text → dates, `.dt.year`, dates as the index, choosing periods | DateTime (Bike Rental) |
| 03 | Data Cleaning | missing / impossible / duplicated values; the 'stale open' problem | Data Preparation |
| 04 | Returns and Volatility | returns, squared returns, the target, HAR ingredients | Data Encoding |
| 05 | Joining the Markets | `merge_asof` without using the future; correlations | — |
| 06 | Crisis Windows | labelling the windows, how wild each was, the main chart | — |
| 07 | x, y, scaling, splitting | why `train_test_split` must not shuffle here; walk-forward; scaling on training years | Data Preparation |
| **Part 2** | **Benchmark models** | | |
| 08 | Measuring Forecast Errors | QLIKE and MSE with small examples | Evaluation |
| 09 | HAR, one year | `OLS`, `summary`, `params`, testing, evaluation, prediction | LR (Simple Salary), Method #1 |
| 10 | HAR, all years | the walk-forward loop | Method #2 |
| 11 | GARCH, one year | `arch_model`: building, training, testing | Method #1 |
| 12 | GARCH, all years | the walk-forward loop | Method #2 |
| 13 | Comparing HAR and GARCH | losses by window; the Diebold–Mariano test step by step | Evaluation |
| **Part 3** | **The LSTM** | | |
| 14 | LSTM input windows | the 22-day windows; fitting / validation / test days; scaling | — |
| 15 | LSTM, one model | building, training (early stopping), testing, evaluation, saving | Method #1 |
| 16 | LSTM, choosing the settings | the grid search, and why not `GridSearchCV` | Method #2 |
| 17 | LSTM, all years | 85 models in a loop (saved results by default) | Method #2 |
| 18 | Comparing all models | research question 1 (H1) | Evaluation |
| 19 | Learning from history | frozen models, research question 3 (H3) | — |
| **Part 4** | **Explaining the forecasts (SHAP)** | | |
| 20 | The idea of SHAP | SHAP by hand on the HAR model | — |
| 21 | SHAP for one LSTM forecast | GradientShap; channel shares; direction vs size; look-back | — |
| 22 | SHAP crisis fingerprints | research question 2 (H2a–H2e) | — |
| 23 | SHAP stability | research question 4 (H4), Spearman by hand | — |
| **Part 5** | **Checks, extensions, prediction** | | |
| 24 | Robustness: Parkinson | the same forecasts against another measure of the truth | — |
| 25 | Exploratory: spillovers | Diebold–Yilmaz with a `statsmodels` VAR, compared with SHAP | — |
| 26 | Exploratory: volatility paradox | calm before the storm? (FRED data from 1955) | — |
| 27 | Prediction and summary | the last forecast; a new situation; `joblib.dump`; all results | Prediction |

Most notebooks run in seconds; 15 and 17 take about a minute; 25 about two minutes. The heavy jobs are switched
off by default and load the saved results instead:
* in notebook 17, set `run_all = True` to train all 85 LSTMs (about 30 minutes on a laptop, 1–1.5 hours on Colab);
* in notebook 19, set `run_frozen = True` to train the 15 frozen models.

Every recomputed number has been checked against the saved results of the study.

## For the maintainer
`build_series_part1.py` … `part3.py` write the notebooks. `run_series_locally.py` runs them in order on the laptop
and stores their outputs.

These notebooks were prepared with AI assistance, as declared in `notes/ai_use.md`. They are study notebooks, not
dissertation text.

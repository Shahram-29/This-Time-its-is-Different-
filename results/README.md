# Results

| File | Produced by | Contents |
|---|---|---|
| `benchmark_forecasts.csv` | `code/04_benchmarks.py` | One row per test day (2007 – Aug 2023): realised 5-day variance, HAR, GARCH and naive forecasts, evaluation window |
| `benchmark_losses.csv` | `code/04_benchmarks.py` | Mean QLIKE and MSE by model, for all test days and each window |
| `benchmark_parameters.csv` | `code/04_benchmarks.py` | HAR coefficients and R², GARCH ω, α, β, persistence and Student-t degrees of freedom, per annual refit |

Variances are in decimal units (e.g. 0.0004 = 2% over 5 days). Annualised volatility = √(variance × 252 / 5).

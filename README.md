# This Time Is Different?

**Explainable deep learning of crisis transmission into Irish stock market volatility:
the Global Financial Crisis, the Irish sovereign debt crisis and COVID-19**

MSc Business Analytics dissertation · Dublin Business School · 2026 · *work in progress*

![Irish equity market 2003–2025 with crisis windows](figures/crisis_chart.png)

## The study in brief
| | |
|---|---|
| **Question** | Can an explainable LSTM forecast Irish stock market volatility better than GARCH and HAR — and do its SHAP explanations show how crises of different origin reached Ireland? |
| **Data** | Daily closes (Yahoo Finance) for the **ISEQ Overall** (target; domestic channel), **S&P 500** (US channel) and **DAX** (euro-area channel); FTSE 100 in one robustness run. Main sample Jan 2003 – Aug 2023 (5,243 trading days). |
| **Target** | Forward 5-day realised variance of the ISEQ. |
| **Models** | GARCH(1,1) with Student-t errors · HAR · LSTM (PyTorch). Walk-forward, annual refits, test years 2007 – 2023. |
| **Explanations** | Captum GradientShap, grouped by channel (domestic / US / euro), look-back lag and input type. |
| **Crises compared** | Global Financial Crisis (US origin) · Irish sovereign debt crisis (euro / domestic) · COVID-19 (global, exogenous) · validation: 2022 war/energy shock, Brexit. |
| **Hypotheses** | Fixed before any model was run: [`PREREGISTRATION.md`](PREREGISTRATION.md) (draft committed first in the history). |

## Results so far

### Benchmarks (step 04)
Walk-forward forecasts, 2007 – Aug 2023, 4,230 forecast days. Mean **QLIKE** loss (lower is better):

| Window | Days | HAR | GARCH(1,1) | Naive (reference) |
|---|---:|---:|---:|---:|
| **All test days** | 4,230 | 0.414 | **0.380** | 0.802 |
| Calm periods | 2,935 | 0.360 | **0.351** | 0.802 |
| Global Financial Crisis | 402 | 0.676 | **0.401** | 0.834 |
| Irish sovereign debt crisis | 574 | 0.380 | **0.374** | 0.802 |
| COVID-19 | 92 | 0.763 | **0.700** | 1.053 |
| War / energy shock 2022 | 185 | **0.282** | 0.290 | 0.506 |
| Brexit referendum | 42 | 1.973 | 1.941 | **1.264** |

Diebold–Mariano test with Harvey–Leybourne–Newbold correction, QLIKE, HAR vs GARCH over all test days:
statistic 3.39, p = 0.0007 (GARCH more accurate). Full tables: [`results/`](results/).

![Benchmark forecasts vs realised volatility](figures/benchmark_forecasts.png)

### LSTM and SHAP (steps 05–08)
*Not yet run.* The SHAP tooling has been checked on calm 2017 days only (no crisis window explained before the
pre-registration), with attributions adding up to the forecast (Integrated Gradients error ≈ 0):

![SHAP smoke test — tooling check only, not a result](figures/smoke_test_shap.png)

## Descriptive figures (from the dataset)

**How six shocks reached the Irish market** — ISEQ vs S&P 500, FTSE 100 and DAX around each event (index = 100 the day before).
![Event windows](figures/hb_event_windows.png)

**Stability before the storm** — the calm 2003–07 boom and the 2008 crash (Minsky).
![Calm boom then crash](figures/hb_minsky_boom.png)

**Links between Dublin and foreign markets over time** — 250-day correlations; crisis windows shaded.
![Rolling correlations](figures/hb_rolling_correlation.png)

**Returns are hard to predict; volatility is not** — autocorrelation of returns vs squared returns.
![Volatility clustering](figures/hb_volatility_clustering.png)

**The three HAR ingredients during the GFC** — daily, weekly and monthly volatility.
![HAR components](figures/hb_har_components.png)

**Why the New York close of day t can be used for Dublin's day t+1** — trading hours in Irish time.
![Trading hours](figures/hb_trading_hours.png)

## Reproduce
```
py -3.13 -m pip install --user -r requirements.txt
cd code
py -3.13 01_data_pipeline.py      # download/cache data, build dataset, quality report, crisis chart
py -3.13 02_handbook_figures.py   # descriptive figures
py -3.13 03_shap_smoke_test.py    # tooling check: LSTM + GradientShap on calm 2017 days only
py -3.13 04_benchmarks.py         # HAR and GARCH walk-forward, losses by window
```
Raw and processed market data are not committed (Yahoo Finance terms of use); `01_data_pipeline.py` rebuilds them.

## Repository layout
| Path | Contents |
|---|---|
| [`code/`](code/) | numbered scripts, run in order; `evaluation.py` holds the shared windows, losses and tests |
| [`results/`](results/) | forecast and loss tables |
| [`figures/`](figures/) | all charts shown above |
| [`data/data_quality_report.txt`](data/data_quality_report.txt) | data checks behind each design decision |
| [`docs/`](docs/) | research plan, Chapter 2 blueprint, literature handbook (planning documents, not dissertation text) |
| [`notes/decisions.md`](notes/decisions.md) | dated design decisions with evidence |
| [`notes/ai_use.md`](notes/ai_use.md) | AI-assistance log |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | hypotheses and design |

## Roadmap
- [x] Data pipeline and quality checks (`01`)
- [x] Descriptive figures (`02`)
- [x] SHAP tooling smoke test (`03`)
- [x] Benchmarks: HAR and GARCH walk-forward (`04`)
- [ ] Pre-registration finalised in my own words
- [ ] LSTM walk-forward, 5 seeds (`05`)
- [ ] Evaluation: Diebold–Mariano–HLN tests; frozen-model "learning from history" test (`06`)
- [ ] Explanations: grouped SHAP by window with bootstrap intervals; seed stability (`07`)
- [ ] Robustness: FTSE for DAX (incl. Brexit), Parkinson, post-2023, look-back 66 (`08`)
- [ ] Dissertation chapters and submission

## Notes
Academic work in progress. Data: Yahoo Finance via `yfinance`, for research use. AI assistance is declared in
[`notes/ai_use.md`](notes/ai_use.md); the dissertation text is written by the author.

# This Time Is Different?

**Explainable deep learning of crisis transmission into Irish stock market volatility:
the Global Financial Crisis, the Irish sovereign debt crisis and COVID-19**

MSc Business Analytics dissertation · Dublin Business School · 2026

## The study in brief
- **Question:** can an explainable LSTM forecast Irish stock market volatility better than GARCH and HAR, and do
  its SHAP explanations show how crises of different origin reached Ireland?
- **Data:** daily closes from Yahoo Finance for the ISEQ Overall (target, domestic channel), S&P 500 (US channel)
  and DAX (euro channel); FTSE 100 for one robustness run. Main sample Jan 2003 – Aug 2023.
- **Target:** forward 5-day realised variance of the ISEQ.
- **Models:** GARCH(1,1), HAR, LSTM (PyTorch); walk-forward with annual refits, 2007–2023.
- **Explanations:** Captum GradientShap, grouped by channel, look-back lag and input type.
- **Hypotheses:** fixed in advance in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Reproduce
```
py -3.13 -m pip install --user -r requirements.txt
cd code
py -3.13 01_data_pipeline.py      # download/cache data, build dataset, quality report, crisis chart
py -3.13 02_handbook_figures.py   # descriptive figures
py -3.13 03_shap_smoke_test.py    # tooling check: LSTM + GradientShap on calm 2017 days only
```
Raw and processed data are not committed (Yahoo Finance terms); `01_data_pipeline.py` rebuilds them.

## Repository layout
| Path | Contents |
|---|---|
| `code/` | numbered scripts, run in order |
| `data/` | `data_quality_report.txt` (committed); `raw/` and `processed/` rebuilt locally |
| `figures/` | generated charts |
| `docs/` | research plan, Chapter 2 blueprint, literature handbook (planning documents, not dissertation text) |
| `notes/decisions.md` | dated design decisions with evidence |
| `notes/ai_use.md` | AI-assistance log for the DBS declaration |
| `PREREGISTRATION.md` | hypotheses and design, committed before any model is run |

## Roadmap
- [x] Data pipeline and quality checks (`01`)
- [x] Descriptive figures (`02`)
- [x] SHAP tooling smoke test (`03`)
- [ ] Final pre-registration committed (rewrite the draft in my own words)
- [x] Benchmarks: HAR and GARCH walk-forward, QLIKE/MSE by window (`04`)
- [ ] LSTM walk-forward, 5 seeds (`05`)
- [ ] Evaluation: Diebold–Mariano–HLN tests; RQ3 frozen models (`06`)
- [ ] Explanations: grouped SHAP by window with bootstrap intervals; seed stability (`07`)
- [ ] Robustness: FTSE for DAX (incl. Brexit), Parkinson, post-2023, look-back 66 (`08`)
- [ ] Chapters 2–3 drafted · Chapters 4–6 drafted · Abstract and Introduction · Submission

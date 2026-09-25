# Decision log

Each design decision, when it was made and why. Cite the evidence in Chapter 3.

| Date | Decision | Reason / evidence |
|---|---|---|
| 24 Sep 2026 | Target = forward 5-day realised variance from closes; Garman–Klass rejected | Yahoo ^ISEQ open equals previous close on 38–75% of days in some periods; GK vs r² monthly corr only 0.54 in 2003–06 (`data/data_quality_report.txt`) |
| 24 Sep 2026 | Main sample ends 31 Aug 2023 | CRH (Sep 2023), Flutter (Jan 2024), Smurfit (Jul 2024) left the ISEQ; ≈€74bn of ≈€152bn ISEQ 20 value (Irish Times) |
| 24 Sep 2026 | Dot-com crash removed; sample starts Jan 2003 | Irish median vol only 14.9% vs 13.1% calm; only 668 days before it. 2005 start rejected: only 659 days before the GFC; 2003 gives 1,166 |
| 24 Sep 2026 | Three markets: ISEQ, S&P 500, DAX | Nasdaq–S&P 0.95; VIX–S&P −0.73; US 10y–ISEQ 0.05; FTSE–DAX 0.84 (2003–2023) |
| 24 Sep 2026 | FTSE only as a robustness swap for DAX (+ Brexit check) | Strongest ISEQ link (0.71) but overlaps DAX; keeps UK question without blurring SHAP |
| 24 Sep 2026 | Foreign data aligned as-of day t | New York closes ≈21:00 Irish time; forecast issued 08:00 on t+1 (Hamao et al., 1990) |
| 24 Sep 2026 | PyTorch + Captum GradientShap | Known shap/TensorFlow-2 LSTM issues (shap GitHub #1193, #1677) |
| 24 Sep 2026 | QLIKE primary, MSE secondary; DM with HLN | Patton (2011); overlapping 5-day forecasts |
| 24 Sep 2026 | Annual expanding-window refits, 5 seeds | Daily refits infeasible; Christensen et al. (2023) also avoid rolling NN refits for cost |
| 24 Sep 2026 | Diebold–Yilmaz and Monte Carlo intervals deferred to PhD | Scope for a 3-month MSc |
| 24 Sep 2026 | GradientShap with 1,024 samples, explained 8 days at a time; Integrated Gradients as exact cross-check | Smoke test (calm 2017 only): additivity gap fell 0.082 → 0.043 → 0.021 for 64 → 256 → 1,024 samples (corr 0.81 → 0.94 → 0.985); IG additivity error ≈ 0; 1,024 samples on all days at once exceeded memory. Channel shares barely changed with n |
| 24 Sep 2026 | Squared-return inputs modelled as log(r², floor 1e-8) | Heavy tails; stabilises LSTM training |
| 25 Sep 2026 | HAR back-transformed with Duan smearing; last 5 training days dropped from HAR estimation; GARCH parameters fixed within each test year | Unbiased level forecasts from a log model; no target overlap into the test year; matches annual refit design. `arch` alignment verified on simulated data (last_obs exclusive; h.1 at origin t uses returns up to t) |
| 25 Sep 2026 | LSTM tuning result frozen: look-back 22, 64 units, 2 layers | Lowest mean validation loss (1.307) over the 8-config grid on the 2007/2008 refits; identical on re-run (reproducible). All configs > 1.0 because validation years (2006, 2007) differ from calm training data |
| 25 Sep 2026 | LSTM back-transformation by Duan smearing on the validation year | Pre-registered "bias adjustment on validation"; factors 1.1–3.1 (largest for 2008: validation 2007 contained the GFC onset) |
| 25 Sep 2026 | SHAP days: every day in each crisis/validation window + every 10th calm day; 1,024 samples, 4-day chunks, 2 workers | Full windows allow a block bootstrap over consecutive days; 6 workers × 8-day chunks exceeded 7.7 GB RAM; measured 0.14 s per day |
| 25 Sep 2026 | H2c operationalised as "max − min channel share smaller than in calm" | Pre-registered wording "shares move towards equality"; stated before SHAP was run (commit 69fb194) |
| 25 Sep 2026 | Naive forecast (last week's RV × 5) reported as reference only | Sanity check; not a pre-registered benchmark |
| 24 Sep 2026 | COVID hypothesis stated in shares as "towards equality" | Channel shares sum to 100%, so all shares cannot rise together; absolute attributions reported alongside |

"""Notebooks 20-27 of the simple series: SHAP, robustness, the two exploratory extensions, prediction."""
from series_common import NB, SETUP
from build_series_part2 import PREP_LSTM, DAYS_2007, LSTM_CLASS, DM_FUNCTION, READ_DATASET

# ------------------------------------------------------------------------------------------------
nb = NB("20_SHAP_Idea_with_HAR", "20. The idea of SHAP — explained with the HAR model", r"""
**Part 4 — Explaining the forecasts (SHAP).** Before using SHAP on the LSTM, we see the idea on a model we already know: the HAR regression of notebook 09. For a linear model, SHAP can be computed by hand.

* **Needs:** `work/dataset.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code(READ_DATASET)
nb.md("## The HAR model of notebook 09 (trained on 2003–2006)")
nb.code("""from numpy import log, exp
from statsmodels.api import OLS, add_constant
x = log(dataset[['rv_d', 'rv_w', 'rv_m']].clip(lower=0.00000001))
y = log(dataset['rv5_fwd'])
train_days = dataset.loc[:'2006-12-31'].dropna(subset=['rv5_fwd', 'rv_m']).index[:-5]
model = OLS(y.loc[train_days], add_constant(x.loc[train_days]))     # building
best_model = model.fit()                                             # training
best_model.params""")
nb.md("""## The idea of SHAP
Think of a team that wins a bonus. **Shapley values** (from game theory) share the bonus fairly among the players, according to how much each one added. **SHAP** (Lundberg & Lee, 2017) does the same for a forecast:

* the 'bonus' = **this day's forecast − the average forecast**;
* the 'players' = the **inputs**;
* each input gets a **contribution**, and the contributions **add up** exactly to the bonus.

For a straight-line model the fair share has a simple formula:

**contribution of an input = its coefficient × (its value today − its average value)**""")
nb.md("## One day: 16 August 2007 (inside the Global Financial Crisis)")
nb.code("""x_day = x.loc['2007-08-16']
x_average = x.loc[train_days].mean()
coefficients = best_model.params[['rv_d', 'rv_w', 'rv_m']]
contributions = coefficients * (x_day - x_average)
contributions""")
nb.md("Check that they add up: the forecast for this day minus the average forecast must equal the sum of the contributions.")
nb.code("""forecast_day = best_model.params['const'] + (coefficients * x_day).sum()
average_forecast = best_model.params['const'] + (coefficients * x_average).sum()
print('forecast - average forecast =', round(forecast_day - average_forecast, 4))
print('sum of the contributions    =', round(contributions.sum(), 4))""")
nb.md("""All three contributions are positive: today, last week and last month were all more volatile than usual, so each one **pushed the forecast up**. The **monthly** input pushed most.

## Shares
To compare inputs we use the size of each contribution (positive or negative does not matter) as a share of the total:""")
nb.code("""shares = contributions.abs() / contributions.abs().sum()
shares.round(3)""")
nb.md("## A calm day for comparison: 15 May 2007")
nb.code("""x_day = x.loc['2007-05-15']
contributions = coefficients * (x_day - x_average)
contributions""")
nb.md("""On this calmer day the contributions are much **smaller** (together about 0.15 instead of 1.03): the inputs were close to their usual level, so they moved the forecast only a little. On a day with lower-than-usual volatility they would be negative and push the forecast **down**.

## From HAR to the LSTM
* HAR uses only the **Irish** market — all its contributions belong to one 'channel' (domestic).
* The LSTM uses **three markets** and reads **22 days × 6 inputs = 132 inputs** for every forecast. It is not a straight line, so there is no simple formula; we use **GradientShap** (notebook 21), a SHAP method for neural networks.
* Adding the contributions by market gives the **channel shares**: which market the model was 'listening to'. This is the main measure of the explanation part of the study.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("21_SHAP_One_LSTM_Forecast", "21. SHAP for one LSTM forecast (Method #1)", r"""
We explain **one** forecast of the LSTM trained in notebook 15 — the forecast for **16 August 2007**, the second week of the Global Financial Crisis — with **GradientShap** from the library `captum`.

* **Needs:** `work/model_data.csv` and the model saved by notebook 15 (`work/lstm_2007_seed1.pt` + its two scaling files). **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("!pip install -q captum")
nb.code(PREP_LSTM)
nb.code(DAYS_2007)
nb.md("## Load the model of notebook 15\nFirst the blueprint (the same class), then the saved weights and scaling numbers.")
nb.code("import torch\n" + LSTM_CLASS)
nb.code("""model = LSTMVol()
model.load_state_dict(torch.load(folder + 'work/lstm_2007_seed1.pt'))
model.eval()
scaling = read_csv(folder + 'work/lstm_2007_seed1_scaling.csv', index_col=0)
x_scaled = (x - scaling['mean']) / scaling['std']
scaling""")
nb.md("""## The background (the 'average' situation)
SHAP compares the forecast of our day with forecasts for **typical** situations. As typical situations we take **200 random windows** from the model's own fitting days (2003–2005). `default_rng(...)` with a fixed number makes the random choice repeatable.""")
nb.code("""from numpy import random, array
fit_positions = []
for day in fit_days:
    fit_positions.append(x.index.get_loc(day))
rng = random.default_rng(1000 * 2007 + 1)
chosen = rng.choice(fit_positions, 200, replace=False)

background = []
for position in chosen:
    background.append(x_scaled.values[position - 21:position + 1])
background = torch.tensor(array(background, dtype='float32'))
print(background.shape)""")
nb.md("## The window of the day we explain")
nb.code("""from pandas import Timestamp
position = x.index.get_loc(Timestamp('2007-08-16'))
day_window = torch.tensor(array([x_scaled.values[position - 21:position + 1]], dtype='float32'))
print(day_window.shape)""")
nb.md("""## GradientShap
* **building:** `GradientShap(model)` prepares the explainer for our model;
* **explaining:** `.attribute(...)` returns one contribution for each of the 132 inputs (22 days × 6). It uses `n_samples=1024` random points between the background windows and our day (more samples = more precise).""")
nb.code("""from captum.attr import GradientShap
torch.manual_seed(1)
random.seed(1)
explainer = GradientShap(model)                                                                   # building
attributions = explainer.attribute(day_window, baselines=background, n_samples=1024, stdevs=0.0)   # explaining
print(attributions.shape)""")
nb.md("As a table: one row per day of the window (t-21 = 21 days before, t-0 = the forecast day), one column per input.")
nb.code("""from pandas import DataFrame
names = []
for k in range(21, -1, -1):
    names.append('t-' + str(k))
table = DataFrame(attributions[0].detach().numpy(), index=names, columns=x.columns)
table.round(4)""")
nb.md("## Do the contributions add up?")
nb.code("""with torch.no_grad():
    forecast_day = model(day_window).item()
    average_forecast = model(background).mean().item()
print('forecast - average forecast   =', round(forecast_day - average_forecast, 4))
print('sum of the 132 contributions  =', round(table.values.sum(), 4))""")
nb.md("Very close (GradientShap is an approximation; the gap shrinks with more samples).")
nb.md("""## Channel shares — which market was the model listening to?
We take the size of every contribution (`abs`) and add them up for the two inputs of each market.""")
nb.code("""abs_table = table.abs()
total = abs_table.values.sum()
domestic = (abs_table['ret'].sum() + abs_table['log_r2'].sum()) / total
us = (abs_table['us_ret'].sum() + abs_table['log_us_r2'].sum()) / total
euro = (abs_table['euro_ret'].sum() + abs_table['log_euro_r2'].sum()) / total
print('domestic (ISEQ):', round(domestic, 3))
print('US (S&P 500)   :', round(us, 3))
print('euro (DAX)     :', round(euro, 3))""")
nb.code("""import matplotlib.pyplot as plt
plt.bar(['domestic', 'US', 'euro'], [domestic, us, euro], color=['black', 'blue', 'green'])
plt.ylabel('share of |contribution|')
plt.title('16 August 2007: which market drove the forecast?')
plt.show()""")
nb.md("## Direction or size?\nReturns show the **direction** of the moves; squared returns their **size**.")
nb.code("""direction = (abs_table['ret'].sum() + abs_table['us_ret'].sum() + abs_table['euro_ret'].sum()) / total
size = 1 - direction
print('direction (returns)        :', round(direction, 3))
print('size (squared returns)     :', round(size, 3))""")
nb.md("## How far back did the model look?")
nb.code("""by_day = abs_table.sum(axis=1) / total
by_day.plot(figsize=(10, 3), marker='o', title='share of |contribution| by day of the window')
plt.show()""")
nb.md("""The Irish market supplies about half of the explanation, the US about 28%, the euro area about 22%. Notebook 22 repeats this for **1,547 days and 5 models per day** and compares the crisis windows with calm times.""")
nb.save()

# ------------------------------------------------------------------------------------------------
WINDOWS4 = "['Global Financial Crisis', 'Irish sovereign debt crisis', 'COVID-19', 'War / energy shock 2022']"

nb = NB("22_SHAP_Crisis_Fingerprints", "22. SHAP fingerprints of the crises (research question 2)", r"""
Notebook 21 explained one day with one model. The study explained **every day in each crisis window plus every 10th calm day — 1,547 days — with all 5 models** of the year (about 20 minutes of computing). Here we load those results and test the hypotheses H2a–H2e.

* **Needs:** `results/shap_channel_shares.csv`, `results/shap_input_types.csv`, `results/shap_lags.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
shap = read_csv(folder + 'results/shap_channel_shares.csv')
shap""")
nb.md("""Each row: a window, a channel, its **share** of the total |contribution|, and a **95% interval** (`ci_low` to `ci_high`).

**How the interval was made (block bootstrap):** draw the explained days again at random — in blocks of 10 neighbouring days, because neighbouring days are alike — recompute the share, repeat 1,000 times, and keep the middle 95% of the results. It shows how much the share could change by chance.""")
nb.code("""order = ['calm', 'Global Financial Crisis', 'Irish sovereign debt crisis', 'COVID-19', 'War / energy shock 2022']
share = shap.pivot(index='window', columns='channel', values='share').loc[order, ['domestic', 'US', 'euro']]
low = shap.pivot(index='window', columns='channel', values='ci_low').loc[order, ['domestic', 'US', 'euro']]
share.round(3)""")
nb.code("""share.plot(kind='bar', figsize=(11, 4), color=['black', 'blue', 'green'], title='Channel shares by window (the fingerprints)')""")
nb.md("""## The rule (written before the results)
A channel **rises** in a crisis if the **lower end of its interval** is **above** its share in calm times.""")
nb.code("""for window in """ + WINDOWS4 + """:
    for channel in ['domestic', 'US', 'euro']:
        if low.loc[window, channel] > share.loc['calm', channel]:
            print(window, '-', channel, ': RISES')
        else:
            print(window, '-', channel, ': does not rise')""")
nb.md("## H2a — Global Financial Crisis: the US share rises the most, and the domestic share is above calm")
nb.code("""change = share.loc['Global Financial Crisis'] - share.loc['calm']
print(change.round(3))
us_rises_most = change.idxmax() == 'US'
us_rises = low.loc['Global Financial Crisis', 'US'] > share.loc['calm', 'US']
domestic_rises = low.loc['Global Financial Crisis', 'domestic'] > share.loc['calm', 'domestic']
print('US rises most:', us_rises_most, '| US rises:', us_rises, '| domestic rises:', domestic_rises)
print('H2a supported:', us_rises_most and us_rises and domestic_rises)""")
nb.md("The US share rises most — but the domestic share **falls**, so H2a is only partly met.")
nb.md("## H2b — Irish debt crisis: euro and domestic rise; the US share is below its GFC level")
nb.code("""euro_rises = low.loc['Irish sovereign debt crisis', 'euro'] > share.loc['calm', 'euro']
domestic_rises = low.loc['Irish sovereign debt crisis', 'domestic'] > share.loc['calm', 'domestic']
us_below_gfc = share.loc['Irish sovereign debt crisis', 'US'] < share.loc['Global Financial Crisis', 'US']
print('euro rises:', euro_rises, '| domestic rises:', domestic_rises, '| US below GFC:', us_below_gfc)
print('H2b supported:', euro_rises and domestic_rises and us_below_gfc)""")
nb.md("## H2c — COVID-19: the shares move towards equality\nMeasured by the gap between the largest and the smallest share (smaller gap = more equal).")
nb.code("""gap_covid = share.loc['COVID-19'].max() - share.loc['COVID-19'].min()
gap_calm = share.loc['calm'].max() - share.loc['calm'].min()
print('gap in COVID =', round(gap_covid, 3), '| gap in calm =', round(gap_calm, 3))
print('H2c supported:', gap_covid < gap_calm)""")
nb.md("Supported — but only just (0.308 vs 0.312).")
nb.md("## H2e — 2022 war / energy shock: the euro share rises")
nb.code("""print('H2e supported:', low.loc['War / energy shock 2022', 'euro'] > share.loc['calm', 'euro'])""")
nb.md("The euro share actually **falls** in 2022 (0.211 → 0.143) while the US share rises.")
nb.code("""saved = read_csv(folder + 'results/shap_hypotheses.csv')
saved""")
nb.md("## Direction or size? And how far back?")
nb.code("""types = read_csv(folder + 'results/shap_input_types.csv', index_col=0)
types.round(3)""")
nb.md("In the GFC, **62%** of the explanation is on the returns (the **direction** of the moves) against 41% in calm times: falls mattered more than the size of moves — the 'leverage effect'.")
nb.code("""lags = read_csv(folder + 'results/shap_lags.csv', index_col=0)
lags.T.plot(figsize=(11, 4), marker='o', title='share of |contribution| by day of the window')""")
nb.md("""In the fast crises (GFC, COVID, 2022) the model looks mostly at the **last 1–5 days**; in the slow Irish debt crisis it looks back like in calm times.

**Remember:** SHAP describes **the model**, not the economy. The study therefore speaks of *fingerprints consistent with transmission*, never of proof of contagion.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("23_SHAP_Stability", "23. Are the explanations stable? (research question 4)", r"""
Each year has 5 LSTMs (seeds). Do they tell the **same story** — do they rank the three markets in the same order? **H4:** the average Spearman correlation between the seeds' rankings is at least **0.8** in every window.

* **Needs:** `results/shap_by_seed.csv`, `results/shap_stability.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
by_seed = read_csv(folder + 'results/shap_by_seed.csv')
by_seed.head(10)""")
nb.md("The channel shares of the 5 seeds in the Global Financial Crisis:")
nb.code("""gfc = by_seed[by_seed['window'] == 'Global Financial Crisis'].set_index('seed')[['domestic', 'US', 'euro']]
gfc.round(3)""")
nb.md("""## Spearman correlation = agreement between two rankings
`rank` gives each market its place (1 = smallest share, 3 = largest):""")
nb.code("gfc.rank(axis=1)")
nb.md("Spearman's correlation is +1 when two seeds rank the markets identically and lower when they disagree. Seeds 1 and 2:")
nb.code("""from scipy.stats import spearmanr
rho = spearmanr(gfc.loc[1], gfc.loc[2])[0]
print('Spearman, seed 1 vs seed 2:', rho)""")
nb.md("All 10 pairs of seeds, with two loops (`a < b` so that each pair is counted once):")
nb.code("""values = []
for a in [1, 2, 3, 4, 5]:
    for b in [1, 2, 3, 4, 5]:
        if a < b:
            rho = spearmanr(gfc.loc[a], gfc.loc[b])[0]
            values.append(rho)
            print('seed', a, 'vs seed', b, ':', round(rho, 2))
print('average:', round(sum(values) / len(values), 2))""")
nb.md("## Every window")
nb.code("""for window in ['calm', 'Global Financial Crisis', 'Irish sovereign debt crisis', 'COVID-19', 'War / energy shock 2022']:
    table = by_seed[by_seed['window'] == window].set_index('seed')[['domestic', 'US', 'euro']]
    values = []
    for a in [1, 2, 3, 4, 5]:
        for b in [1, 2, 3, 4, 5]:
            if a < b:
                values.append(spearmanr(table.loc[a], table.loc[b])[0])
    print(window, ': average Spearman =', round(sum(values) / len(values), 2))""")
nb.code("""saved = read_csv(folder + 'results/shap_stability.csv', index_col=0)
saved.round(2)""")
nb.code("""if (saved['mean_spearman_channels'] >= 0.8).all():
    print('H4 supported')
else:
    print('H4 NOT supported: in some windows the seeds disagree')""")
nb.md("""Fully stable in calm times and 2022 (1.0), less so in the GFC and the Irish crisis (0.7). With only three markets one swapped pair already lowers Spearman to 0.5, so the measure is coarse; the column with 6 inputs gives a finer picture. The explanations were least stable in the GFC — the window where the model was also least accurate.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("24_Robustness_Parkinson", "24. Robustness check — another measure of the truth", r"""
**Part 5 — Checks, extensions and prediction.** All scores so far compare the forecasts with the realised variance from **closing prices**. Would the ranking of the models change with a different measure of the 'truth'? The **Parkinson** measure uses each day's **high and low** prices.

* **Needs:** `work/dataset.csv` and the three forecast files in `work/`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv, DataFrame
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
har = read_csv(folder + 'work/har_forecasts.csv', index_col=0, parse_dates=True)
garch = read_csv(folder + 'work/garch_forecasts.csv', index_col=0, parse_dates=True)
lstm = read_csv(folder + 'work/lstm_forecasts.csv', index_col=0, parse_dates=True)""")
nb.md(r"""## The Parkinson measure
$\sigma^2_{day} = \dfrac{(\ln \text{High} - \ln \text{Low})^2}{4 \ln 2}$ — a day with a wide range between high and low was a volatile day. Summing the next 5 days gives a second 'truth' for next week.""")
nb.code("""from numpy import log
dataset['park'] = log(dataset['High'] / dataset['Low']) ** 2 / (4 * log(2))
dataset['park5'] = dataset['park'].shift(-1) + dataset['park'].shift(-2) + dataset['park'].shift(-3) + dataset['park'].shift(-4) + dataset['park'].shift(-5)
dataset[['rv5_fwd', 'park5']].loc['2008-10-01':'2008-10-08']""")
nb.md("The high–low range misses the move overnight (between one close and the next open), so the Parkinson numbers are put on the same level as the close-to-close numbers, using **2003–2006 only** (before any test day):")
nb.code("""scale = dataset.loc['2003':'2006', 'rv5_fwd'].mean() / dataset.loc['2003':'2006', 'park5'].mean()
print('scale factor =', round(scale, 3))""")
nb.md("## Score the same forecasts against the new truth")
nb.code("""results = DataFrame({'truth': scale * dataset.loc[har.index, 'park5'],
                     'LSTM': lstm['LSTM'], 'GARCH': garch['GARCH'], 'HAR': har['HAR'],
                     'window': dataset.loc[har.index, 'window']})
results = results[results['truth'] > 0]
for m in ['LSTM', 'GARCH', 'HAR']:
    ratio = results['truth'] / results[m]
    results['QLIKE_' + m] = ratio - log(ratio) - 1
results[['QLIKE_LSTM', 'QLIKE_GARCH', 'QLIKE_HAR']].mean().round(3)""")
nb.code("""results.groupby('window')[['QLIKE_LSTM', 'QLIKE_GARCH', 'QLIKE_HAR']].mean().round(3)""")
nb.code(DM_FUNCTION)
nb.code("""print('LSTM vs GARCH:')
dm_test(results['QLIKE_LSTM'], results['QLIKE_GARCH'])
print('LSTM vs HAR:')
dm_test(results['QLIKE_LSTM'], results['QLIKE_HAR'])""")
nb.code("""saved = read_csv(folder + 'results/robust_parkinson_losses.csv', index_col=0)
saved.filter(like='QLIKE').round(3)""")
nb.md("""**Result:** GARCH is still the most accurate overall, the LSTM is still significantly worse than GARCH (but no longer significantly worse than HAR), and the LSTM is still the best model in COVID-19. The main conclusions do not depend on how the 'truth' is measured.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("25_Exploratory_Spillovers", "25. Exploratory E1 — a second lens: spillovers (Diebold–Yilmaz)", r"""
*Exploratory: added after the main results (PREREGISTRATION.md, Amendment A1).*

SHAP asked which markets the **neural network** listens to. The **Diebold–Yilmaz spillover index** asks a simple **linear model** of the three markets — a **VAR** (vector autoregression) — how much of the uncertainty in forecasting the ISEQ's volatility comes from shocks in each market. Do the two tools tell the same story?

* **Needs:** `work/dataset.csv`, `work/lstm_forecasts.csv`, `results/shap_channel_shares.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code(READ_DATASET)
nb.md("## The data: log 5-day volatility of the three markets")
nb.code("""from numpy import log
vol = dataset[['r2', 'us_r2', 'euro_r2']].rolling(5).mean()
vol = log(vol.clip(lower=0.00000001)).dropna()
vol.columns = ['domestic', 'US', 'euro']
vol.head()""")
nb.md("""## A VAR — one regression per market
Each market's volatility today is regressed on the last **4 days** of **all three** markets. `statsmodels` builds and fits it in the usual two steps:""")
nb.code("""from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')     # statsmodels switches some warnings back on: hide them again
model = VAR(vol)              # building
best_model = model.fit(4)     # training: 4 days of lags
best_model.params.round(3)""")
nb.md(r"""## How a shock travels, and who causes the uncertainty
* `ma_rep(9)` gives 10 tables $A_0 … A_9$: how a shock today shows up 0, 1, … 9 days later.
* `sigma_u` tells how the three markets' daily surprises move together.
* The **generalized variance decomposition** (Pesaran & Shin, 1998) adds up, over 10 days, how much of each market's forecast uncertainty comes from shocks in each market. `@` is matrix multiplication.""")
nb.code("""from numpy import asarray, zeros
A = best_model.ma_rep(9)
sigma = asarray(best_model.sigma_u)

share = zeros((3, 3))
for i in range(3):                      # the market whose forecast we look at
    denominator = 0
    for h in range(10):
        denominator = denominator + (A[h] @ sigma @ A[h].T)[i, i]
    for j in range(3):                  # the market the shock comes from
        numerator = 0
        for h in range(10):
            numerator = numerator + (A[h] @ sigma)[i, j] ** 2
        share[i, j] = numerator / sigma[j, j] / denominator

for i in range(3):                      # rescale each row to add up to 1
    share[i] = share[i] / share[i].sum()

from pandas import DataFrame
table = DataFrame(share * 100, index=['domestic', 'US', 'euro'], columns=['from domestic', 'from US', 'from euro'])
table.round(2)""")
nb.md("""Read the first row: of the uncertainty in forecasting **Irish** volatility, about 64% comes from Irish shocks, 18% from the US and 18% from the euro area. The **total spillover index** is the average share that comes from *other* markets:""")
nb.code("""total = (share.sum() - share[0, 0] - share[1, 1] - share[2, 2]) / 3 * 100
print('total spillover index =', round(total, 1), '%')""")
nb.md("""## Over time: a 200-day rolling window
The same calculation on the 200 days up to each day, moving one day at a time (about 5,000 VARs — a minute or two). The steps above go into a function:""")
nb.code("""def iseq_row(data200):
    best_model = VAR(data200).fit(4)
    A = best_model.ma_rep(9)
    sigma = asarray(best_model.sigma_u)
    share = zeros((3, 3))
    for i in range(3):
        denominator = 0
        for h in range(10):
            denominator = denominator + (A[h] @ sigma @ A[h].T)[i, i]
        for j in range(3):
            numerator = 0
            for h in range(10):
                numerator = numerator + (A[h] @ sigma)[i, j] ** 2
            share[i, j] = numerator / sigma[j, j] / denominator
    for i in range(3):
        share[i] = share[i] / share[i].sum()
    total = (share.sum() - share[0, 0] - share[1, 1] - share[2, 2]) / 3 * 100
    return [share[0, 0], share[0, 1], share[0, 2], total]


rows = []
for end in range(199, len(vol)):
    rows.append(iseq_row(vol.iloc[end - 199:end + 1]))
    if end % 1000 == 0:
        print(end, 'of', len(vol))
rolling = DataFrame(rows, index=vol.index[199:], columns=['domestic', 'US', 'euro', 'total'])
rolling.tail()""")
nb.code("""rolling[['US', 'euro']].plot(figsize=(12, 4), title='Share of ISEQ forecast uncertainty from the US and the euro area (200-day windows)')""")
nb.md("## Compare with SHAP on the same days\nThe SHAP days were every day in the crisis windows plus every 10th calm day.")
nb.code("""from pandas import read_csv
lstm = read_csv(folder + 'work/lstm_forecasts.csv', index_col=0, parse_dates=True)
windows = dataset.loc[lstm.index, 'window']
calm_days = lstm.index[windows == 'calm'][::10]
crisis_days = lstm.index[windows.isin(""" + WINDOWS4 + """)]
shap_days = crisis_days.union(calm_days)
dy = rolling.loc[shap_days].groupby(windows.loc[shap_days]).mean()
order = ['calm', 'Global Financial Crisis', 'Irish sovereign debt crisis', 'COVID-19', 'War / energy shock 2022']
dy = dy.loc[order]
dy.round(3)""")
nb.code("""shap = read_csv(folder + 'results/shap_channel_shares.csv')
shap_share = shap.pivot(index='window', columns='channel', values='share').loc[order, ['domestic', 'US', 'euro']]
shap_share.round(3)""")
nb.md("""## Do they agree?
1. For the 4 crises × 2 foreign markets: does the share **move the same way** (up or down) from calm in both tools?
2. In each of the 5 windows: do both tools put the **same foreign market first**?""")
nb.code("""same_direction = 0
for window in """ + WINDOWS4 + """:
    for channel in ['US', 'euro']:
        shap_up = shap_share.loc[window, channel] > shap_share.loc['calm', channel]
        dy_up = dy.loc[window, channel] > dy.loc['calm', channel]
        if shap_up == dy_up:
            same_direction = same_direction + 1
print('same direction of change:', same_direction, 'of 8')

same_order = 0
for window in order:
    shap_us_first = shap_share.loc[window, 'US'] > shap_share.loc[window, 'euro']
    dy_us_first = dy.loc[window, 'US'] > dy.loc[window, 'euro']
    if shap_us_first == dy_us_first:
        same_order = same_order + 1
print('same foreign market first:', same_order, 'of 5')""")
nb.md("""**Weak agreement.** Both tools see the US share highest in COVID-19, but they differ most in 2022 (the spillover euro share rises, SHAP's falls). The two tools measure different things — a linear model of volatility only, against a non-linear network that also uses the direction of moves — so the 'fingerprints' depend on the lens. That is a finding in itself, to discuss in your own words.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("26_Exploratory_Volatility_Paradox", "26. Exploratory E2 — was there calm before the storm?", r"""
*Exploratory: added after the main results (PREREGISTRATION.md, Amendment A1).*

Danielsson, Valenzuela & Zer (2018) looked at 60 countries over up to 211 years: **long spells of volatility below its usual level** came before banking crises — calm encourages borrowing and risk ('stability is destabilising', Minsky). Was that true before our crises? A long history is needed, so we use **monthly** share prices from the OECD (via FRED): Ireland from 1955, the US from 1957, Germany from 1960.

* **Needs:** the three `FRED_*.csv` files. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.md("## Ireland, step by step")
nb.code("""from pandas import read_csv
prices = read_csv(folder + 'FRED_SPASTT01IEM661N.csv', index_col=0, parse_dates=True)
prices.head()""")
nb.code("""prices = prices['SPASTT01IEM661N'].dropna().loc[:'2023-06-30']
from numpy import log, sqrt
r = log(prices).diff().dropna()                         # monthly returns
r = r.clip(r.quantile(0.005), r.quantile(0.995))        # cap the most extreme 0.5% at each end
r.tail()""")
nb.md("""## Annual volatility, from July to June
As in the paper, a 'year' runs from **July to June** (July 2006 – June 2007 = year 2007). `(r.index.month >= 7)` is `True` (counted as 1) from July on, so those months move to the next year's label.""")
nb.code("""year_label = r.index.year + (r.index.month >= 7)
groups = r.groupby(year_label)
sigma = groups.std() * sqrt(12)
sigma = sigma[groups.size() == 12]       # complete years only
sigma.tail()""")
nb.md("""## The usual level: a one-sided trend
The **Hodrick–Prescott filter** splits a series into a smooth **trend** and the deviations from it (`lamb=5000` sets how smooth). 'One-sided' means: the trend for each year is estimated from the **past years only** — no peeking at the future. So we run the filter again for every year, each time on the data up to that year, and keep its last point.""")
nb.code("""from statsmodels.tsa.filters.hp_filter import hpfilter
from pandas import Series
trend_values = []
for t in range(9, len(sigma)):                       # start after 10 years of data
    cycle, trend = hpfilter(sigma.values[:t + 1], lamb=5000)
    trend_values.append(trend[-1])
trend = Series(trend_values, index=sigma.index[9:])
trend.tail()""")
nb.md("## Below-trend volatility, and its average over the previous 5 years")
nb.code("""below = (sigma - trend).dropna().clip(upper=0)      # 0 when volatility is above its trend
below5 = below.rolling(5).mean().shift(1)           # average of the 5 years BEFORE each year
below5.loc[below5.index.max() + 1] = below.iloc[-5:].mean()
below5 = below5.dropna()
below5.loc[2004:2009].round(3)""")
nb.md("""For the GFC we look at the five years **before** it started (2003–2007), i.e. the value for 2008. Is it unusually low (more negative) compared with Ireland's whole history? The **percentile** = the share of years with a value as low or lower; **bottom 20% = unusually calm**.""")
nb.code("""value = below5.loc[2008]
percentile = (below5 <= value).mean() * 100
print('Ireland before the GFC:', round(value, 3), '| percentile:', round(percentile, 1))""")
nb.md("40th percentile: calm, but **not unusually** calm by Ireland's own history.")
nb.md("## All three markets and all three crises\nThe same steps in a loop.")
nb.code("""codes = {'Ireland': 'SPASTT01IEM661N', 'United States': 'SPASTT01USM661N', 'Germany': 'SPASTT01DEM661N'}
crisis_year = {'Global Financial Crisis': 2008, 'Irish sovereign debt crisis': 2010, 'COVID-19': 2020}

for market in codes:
    prices = read_csv(folder + 'FRED_' + codes[market] + '.csv', index_col=0, parse_dates=True)
    prices = prices[codes[market]].dropna().loc[:'2023-06-30']
    r = log(prices).diff().dropna()
    r = r.clip(r.quantile(0.005), r.quantile(0.995))
    groups = r.groupby(r.index.year + (r.index.month >= 7))
    sigma = (groups.std() * sqrt(12))[groups.size() == 12]
    trend_values = []
    for t in range(9, len(sigma)):
        cycle, trend = hpfilter(sigma.values[:t + 1], lamb=5000)
        trend_values.append(trend[-1])
    trend = Series(trend_values, index=sigma.index[9:])
    below = (sigma - trend).dropna().clip(upper=0)
    below5 = below.rolling(5).mean().shift(1)
    below5.loc[below5.index.max() + 1] = below.iloc[-5:].mean()
    below5 = below5.dropna()
    for crisis in crisis_year:
        value = below5.loc[crisis_year[crisis]]
        percentile = (below5 <= value).mean() * 100
        if percentile <= 20:
            verdict = 'UNUSUALLY CALM'
        else:
            verdict = 'not unusual'
        print(market, '|', crisis, '|', round(value, 3), '| percentile', round(percentile, 1), '|', verdict)""")
nb.code("""saved = read_csv(folder + 'results/volatility_paradox_crises.csv')
saved[['market', 'window', 'low5_before', 'percentile', 'unusually_calm']].round(3)""")
nb.md("""**Result:** before the GFC only **Germany** was unusually calm (4th percentile — its deepest calm in the record). Ireland and the US were calm but not unusually so. No market was unusually calm before COVID-19 — as expected for a shock from outside the financial system.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("27_Prediction", "27. Prediction and summary", r"""
The class notebooks ended with a **Prediction** for a new case (`DT_classifier1.predict(new1_scaled)`) and saved the model (`joblib.dump`). We do the same, and finish with a summary of all results.

* **Needs:** `work/dataset.csv` and the forecast files in `work/`. **Makes:** `work/HAR_model_2023.pkl`.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
har = read_csv(folder + 'work/har_forecasts.csv', index_col=0, parse_dates=True)
garch = read_csv(folder + 'work/garch_forecasts.csv', index_col=0, parse_dates=True)
lstm = read_csv(folder + 'work/lstm_forecasts.csv', index_col=0, parse_dates=True)""")
nb.md("## The last forecast of the study\nOn **31 August 2023**, the last day of the main sample, each model forecast the next week. What happened?")
nb.code("""from numpy import sqrt
day = '2023-08-31'
print('realised :', round(sqrt(dataset.loc[day, 'rv5_fwd'] * 252 / 5) * 100, 1), '%')
print('LSTM     :', round(sqrt(lstm.loc[day, 'LSTM'] * 252 / 5) * 100, 1), '%')
print('GARCH    :', round(sqrt(garch.loc[day, 'GARCH'] * 252 / 5) * 100, 1), '%')
print('HAR      :', round(sqrt(har.loc[day, 'HAR'] * 252 / 5) * 100, 1), '%')""")
nb.md("## A new situation, and saving the model\nThe HAR model of the last year (trained up to 31 December 2022):")
nb.code("""from numpy import log, exp
from statsmodels.api import OLS, add_constant
data = dataset.loc[:'2023-08-31']
x = log(data[['rv_d', 'rv_w', 'rv_m']].clip(lower=0.00000001))
y = log(data['rv5_fwd'])
train_days = data.loc[:'2022-12-31'].dropna(subset=['rv5_fwd', 'rv_m']).index[:-5]
model = OLS(y.loc[train_days], add_constant(x.loc[train_days]))     # building
best_model = model.fit()                                             # training
smear = exp(best_model.resid).mean()
best_model.params""")
nb.md("**Prediction:** suppose that today the ISEQ fell **6%**, after a week of 2% daily moves and a month of 1.2% moves. Next week's expected volatility?")
nb.code("""from pandas import DataFrame
new1 = DataFrame([{'const': 1, 'rv_d': log(0.06 ** 2), 'rv_w': log(0.02 ** 2), 'rv_m': log(0.012 ** 2)}])
new_pred = exp(best_model.predict(new1)) * smear
print('forecast for next week (annualised volatility):', round(sqrt(new_pred[0] * 252 / 5) * 100, 1), '%')""")
nb.code("""import joblib
joblib.dump(best_model, folder + 'work/HAR_model_2023.pkl')
HAR_loaded = joblib.load(folder + 'work/HAR_model_2023.pkl')
HAR_loaded.params.round(3)""")
nb.md("""# Summary of the whole series

| Question | Notebook | Result |
|---|---|---|
| RQ1 / H1 — is the LSTM more accurate than GARCH and HAR? | 18 | **Not supported**: over 2007–2023 the LSTM is significantly *less* accurate (QLIKE 0.463 vs 0.380 and 0.414). Best model in COVID-19 and 2022, worst in the GFC. |
| RQ2 / H2 — do the SHAP fingerprints follow each crisis's origin? | 22 | **Mostly not**: H2a and H2b partly, H2c narrowly supported, H2e not supported. The Irish market supplies about half of every forecast. |
| RQ3 / H3 — does a model trained on past crises cope with a new one? | 19 | **Not as ranked**, but the GFC model (trained only on calm years) was 4.3 × worse than HAR; the COVID model beat HAR. |
| RQ4 / H4 — are the explanations stable across seeds? | 23 | **Not supported**: Spearman 0.7 in the GFC and Irish windows. |
| Robustness — Parkinson measure | 24 | Same conclusions. |
| E1 (exploratory) — SHAP vs spillovers | 25 | Weak agreement (4 of 8, 2 of 5). |
| E2 (exploratory) — calm before the storm? | 26 | Only Germany unusually calm before the GFC. |

Most predictions were not supported — and because they were written down **before** the results (pre-registration), these are genuine findings. Interpreting them, and writing the dissertation, is your work.""")
nb.save()

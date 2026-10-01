"""Notebooks 09-19 of the simple series: HAR, GARCH, comparing benchmarks, and the LSTM."""
from series_common import NB, SETUP

READ_DATASET = """from pandas import read_csv
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
dataset = dataset.loc[:'2023-08-31']        # main sample
print(dataset.shape)"""

# ------------------------------------------------------------------------------------------------
nb = NB("09_HAR_One_Year", "09. HAR model — one year (Method #1)", r"""
The first benchmark, **HAR**, is a **Linear Regression** fitted with `statsmodels` `OLS` — the same tool as the class notebook *LR (Simple Salary)*. In this notebook we build it for **one** test year: train on 2003–2006, test on 2007.

* **Needs:** `work/dataset.csv`. **Makes:** nothing new (notebook 10 does all years).
""")
nb.code(SETUP)
nb.code(READ_DATASET)
nb.md(r"""## The idea
Next week's volatility depends on **today's**, **last week's** and **last month's** volatility (Corsi, 2009) — like traders who look at different horizons:

$$\ln RV5 = B_0 + B_d \cdot \ln(\text{today}) + B_w \cdot \ln(\text{last week}) + B_m \cdot \ln(\text{last month})$$

Compare with the Salary notebook: *Salary = B0 + B1 × YearsExperience*. Here there are three x's instead of one, and everything is in logs (notebook 04 explained why).""")
nb.md("## Divide data to x and y")
nb.code("""from numpy import log, exp
x = log(dataset[['rv_d', 'rv_w', 'rv_m']].clip(lower=0.00000001))
y = log(dataset['rv5_fwd'])
x.tail()""")
nb.md("""## Data Splitting (by time)
* **Training:** 2003 – 2006, **without the last 5 days of 2006**: their target (the next 5 days) reaches into 2007, the test year — that would be peeking at the answers.
* **Testing:** every day of 2007.""")
nb.code("""train_days = dataset.loc[:'2006-12-31'].dropna(subset=['rv5_fwd', 'rv_m']).index
train_days = train_days[:-5]
test_days = dataset.loc['2007-01-01':'2007-12-31'].dropna(subset=['rv5_fwd']).index
print(len(train_days), 'training days')
print(len(test_days), 'test days')""")
nb.code("""x_train = x.loc[train_days]
y_train = y.loc[train_days]
x_test = x.loc[test_days]
y_test = dataset.loc[test_days, 'rv5_fwd']      # the realised variance we want to forecast
print(x_train.shape)
print(x_test.shape)""")
nb.md("# Linear Regression (HAR)\n## Modelling")
nb.code("""from statsmodels.api import OLS, add_constant
x_train = add_constant(x_train)
model = OLS(y_train, x_train)      # building
best_model = model.fit()           # training
print(best_model.summary())""")
nb.code("best_model.params")
nb.md("""Read it like the Salary equation, for example:

**log RV5 = −3.83 + 0.007 × log(today) + 0.035 × log(last week) + 0.425 × log(last month)**

The **monthly** term has the biggest weight: the slow level of volatility matters most.""")
nb.code("""print('R2 =', round(best_model.rsquared * 100, 2), '%')""")
nb.md("""The R² is only about 11%: the calm 2003–2006 years are hard to learn from (there was little to learn). Notebook 10 shows how R² grows once crisis years are in the training data.""")
nb.md(r"""## Testing
The model predicts the **log** of the variance. To go back we use `exp(...)`. But $\exp$ of an average log is a little too small on average, so we multiply by a **smearing factor** (Duan): the average of $\exp$(residual) in the training data.""")
nb.code("""x_test = add_constant(x_test, has_constant='add')
log_pred = best_model.predict(x_test)          # testing (in logs)
smear = exp(best_model.resid).mean()
print('smearing factor =', round(smear, 3))
y_pred1 = exp(log_pred) * smear
y_pred1.head()""")
nb.md("## Evaluation")
nb.code("""ratio = y_test / y_pred1
QLIKE1 = (ratio - log(ratio) - 1).mean()
print('QLIKE (HAR, 2007) =', round(QLIKE1, 3))""")
nb.md("Check: are these the same forecasts as in the project's saved results?")
nb.code("""saved = read_csv(folder + 'results/benchmark_forecasts.csv', index_col=0, parse_dates=True)
print('largest difference from the saved HAR forecasts:', (y_pred1 - saved.loc[test_days, 'HAR']).abs().max())""")
nb.md("""# Prediction
Like `best_model.predict([[1, 20]])` in the Salary notebook: a **new** situation. Suppose the ISEQ moved **5%** today, about **2%** a day over the last week and **1.5%** a day over the last month. What does HAR expect for next week?""")
nb.code("""from pandas import DataFrame
from numpy import sqrt
new1 = DataFrame([{'const': 1, 'rv_d': log(0.05 ** 2), 'rv_w': log(0.02 ** 2), 'rv_m': log(0.015 ** 2)}])
new_pred = exp(best_model.predict(new1)) * smear
print('forecast for next week (annualised volatility):', round(sqrt(new_pred[0] * 252 / 5) * 100, 1), '%')""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("10_HAR_All_Years", "10. HAR model — all years (Method #2)", r"""
Notebook 09 did one year. Now we repeat the same three steps — **building, training, testing** — for every test year from 2007 to 2023 (the *walk-forward* of notebook 07). A `for` loop does the repeating.

* **Needs:** `work/dataset.csv`. **Makes:** `work/har_forecasts.csv`.
""")
nb.code(SETUP)
nb.code(READ_DATASET)
nb.code("""from numpy import log, exp
x = log(dataset[['rv_d', 'rv_w', 'rv_m']].clip(lower=0.00000001))
y = log(dataset['rv5_fwd'])""")
nb.md("""## The loop
For each `year`: the training days run until 31 December of the year before (minus the last 5 days), and the test days are the days of `year`. Each forecast is added to a list.""")
nb.code("""from statsmodels.api import OLS, add_constant
forecasts = []
years = []
r2_list = []
monthly_weight = []

for year in range(2007, 2024):
    train_days = dataset.loc[:str(year - 1) + '-12-31'].dropna(subset=['rv5_fwd', 'rv_m']).index[:-5]
    test_days = dataset.loc[str(year) + '-01-01':str(year) + '-12-31'].dropna(subset=['rv5_fwd']).index

    model = OLS(y.loc[train_days], add_constant(x.loc[train_days]))        # building
    best_model = model.fit()                                                 # training
    smear = exp(best_model.resid).mean()
    y_pred = exp(best_model.predict(add_constant(x.loc[test_days], has_constant='add'))) * smear   # testing

    forecasts.append(y_pred)
    years.append(year)
    r2_list.append(best_model.rsquared)
    monthly_weight.append(best_model.params['rv_m'])
    print(year, ':', len(test_days), 'test days | R2 =', round(best_model.rsquared, 2))""")
nb.md("`concat` puts the 17 yearly pieces one after the other:")
nb.code("""from pandas import concat, DataFrame
har = concat(forecasts)
print(len(har), 'forecast days')
har.head()""")
nb.md("## How the model changed over the years")
nb.code("""DataFrame({'test year': years, 'R2': r2_list, 'monthly weight': monthly_weight}).round(2)""")
nb.md("R² jumps from 0.11 to about 0.5 once the 2008 crisis is in the training data: the model learns what a crisis looks like.")
nb.md("## Check and save")
nb.code("""saved = read_csv(folder + 'results/benchmark_forecasts.csv', index_col=0, parse_dates=True)
print('largest difference from the saved HAR forecasts:', (har - saved['HAR']).abs().max())
DataFrame({'HAR': har}).to_csv(folder + 'work/har_forecasts.csv')""")
nb.md("## Picture: forecasts against what happened")
nb.code("""from numpy import sqrt
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 4))
plt.plot(sqrt(dataset.loc[har.index, 'rv5_fwd'] * 252 / 5), color='lightgrey', label='realised')
plt.plot(sqrt(har * 252 / 5), color='blue', label='HAR forecast')
plt.ylim(0, 1.3)
plt.legend()
plt.title('HAR forecasts of next-week volatility (annualised)')
plt.show()""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("11_GARCH_One_Year", "11. GARCH model — one year (Method #1)", r"""
The second benchmark, **GARCH(1,1)**, is the industry-standard volatility model and is famously hard to beat. It needs the library `arch`. As in notebook 09, we build it for one year: estimate on data before 2007, test on 2007.

* **Needs:** `work/dataset.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.md("The library `arch` is not in Colab by default, so we install it (like `pip install category_encoders` in the Life Expectancy notebook):")
nb.code("!pip install -q arch")
nb.code(READ_DATASET)
nb.md(r"""## The idea
$$\sigma^2_{tomorrow} = \omega + \alpha \cdot \text{shock}_{today}^2 + \beta \cdot \sigma^2_{today}$$
Tomorrow's variance = a small constant + a reaction to today's surprise + most of today's variance carried forward.

A calculation by hand (returns in %): $\omega$ = 0.02, $\alpha$ = 0.09, $\beta$ = 0.90; today's variance is 1, and today the market fell 3%.""")
nb.code("""omega = 0.02
alpha = 0.09
beta = 0.90
variance_today = 1.0
shock_today = -3.0
variance_tomorrow = omega + alpha * shock_today ** 2 + beta * variance_today
print('variance tomorrow =', variance_tomorrow)""")
nb.md("A 3% fall pushes tomorrow's variance from 1 to 1.73. $\\alpha + \\beta$ close to 1 means the shock fades only slowly.")
nb.md("## The data: daily returns in per cent\nGARCH works on the daily returns (×100, in per cent, which helps the estimation).")
nb.code("""returns = dataset['ret'].dropna() * 100
returns.head()""")
nb.md("# GARCH(1,1)\n## Modelling\nStudent-t errors (`dist='t'`) allow for crashes (fat tails). `last_obs='2007-01-01'` means: estimate using data **before** 2007 only.")
nb.code("""from arch import arch_model
GARCH_model1 = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='t')   # building
GARCH_fit1 = GARCH_model1.fit(last_obs='2007-01-01', disp='off')                        # training
print(GARCH_fit1.summary())""")
nb.code("""print('alpha (reaction) =', round(GARCH_fit1.params['alpha[1]'], 3))
print('beta (memory)    =', round(GARCH_fit1.params['beta[1]'], 3))
print('alpha + beta     =', round(GARCH_fit1.params['alpha[1]'] + GARCH_fit1.params['beta[1]'], 3))
print('nu (tails)       =', round(GARCH_fit1.params['nu'], 1))""")
nb.md("## Testing\nFrom every day of 2007 the model forecasts the variance 1, 2, 3, 4 and 5 days ahead (`horizon=5`). The parameters stay fixed, but each day's forecast uses that day's newest return.")
nb.code("""test_days = dataset.loc['2007-01-01':'2007-12-31'].dropna(subset=['rv5_fwd']).index
forecast1 = GARCH_fit1.forecast(horizon=5, start=test_days[0], reindex=False)     # testing
forecast1.variance.head()""")
nb.md("Columns `h.1` … `h.5` are the variances 1 to 5 days ahead. Next week's variance is their **sum**. Because the returns were in per cent, we divide by 100² = 10,000 to come back to decimals.")
nb.code("""y_pred1 = forecast1.variance.loc[test_days].sum(axis=1) / 10000
y_pred1.head()""")
nb.md("## Evaluation")
nb.code("""from numpy import log
y_test = dataset.loc[test_days, 'rv5_fwd']
ratio = y_test / y_pred1
QLIKE1 = (ratio - log(ratio) - 1).mean()
print('QLIKE (GARCH, 2007) =', round(QLIKE1, 3))""")
nb.code("""saved = read_csv(folder + 'results/benchmark_forecasts.csv', index_col=0, parse_dates=True)
print('largest difference from the saved GARCH forecasts:', (y_pred1 - saved.loc[test_days, 'GARCH']).abs().max())""")
nb.md("In 2007 GARCH (0.52) is more accurate than HAR (0.66 in notebook 09): it reacts faster when the crisis starts in August.")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("12_GARCH_All_Years", "12. GARCH model — all years (Method #2)", r"""
The walk-forward for GARCH: one estimation per year, 2007–2023, with a `for` loop — the same idea as notebook 10.

* **Needs:** `work/dataset.csv`. **Makes:** `work/garch_forecasts.csv`.
""")
nb.code(SETUP)
nb.code("!pip install -q arch")
nb.code(READ_DATASET)
nb.code("""returns = dataset['ret'].dropna() * 100""")
nb.code("""from arch import arch_model
forecasts = []
years = []
persistence = []
tails = []

for year in range(2007, 2024):
    test_days = dataset.loc[str(year) + '-01-01':str(year) + '-12-31'].dropna(subset=['rv5_fwd']).index

    GARCH_model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='t')   # building
    GARCH_fit = GARCH_model.fit(last_obs=str(year) + '-01-01', disp='off')                # training
    forecast = GARCH_fit.forecast(horizon=5, start=test_days[0], reindex=False)          # testing
    y_pred = forecast.variance.loc[test_days].sum(axis=1) / 10000

    forecasts.append(y_pred)
    years.append(year)
    persistence.append(GARCH_fit.params['alpha[1]'] + GARCH_fit.params['beta[1]'])
    tails.append(GARCH_fit.params['nu'])
    print(year, ': alpha + beta =', round(persistence[-1], 3), '| nu =', round(tails[-1], 1))""")
nb.code("""from pandas import concat, DataFrame
garch = concat(forecasts)
print(len(garch), 'forecast days')
DataFrame({'test year': years, 'alpha + beta': persistence, 'nu (tails)': tails}).round(3)""")
nb.md("Persistence stays between about 0.96 and 0.999: shocks to Irish volatility fade slowly. `nu` between 6 and 8 means fatter tails than a bell curve.")
nb.code("""saved = read_csv(folder + 'results/benchmark_forecasts.csv', index_col=0, parse_dates=True)
print('largest difference from the saved GARCH forecasts:', (garch - saved['GARCH']).abs().max())
DataFrame({'GARCH': garch}).to_csv(folder + 'work/garch_forecasts.csv')""")
nb.code("""from numpy import sqrt
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 4))
plt.plot(sqrt(dataset.loc[garch.index, 'rv5_fwd'] * 252 / 5), color='lightgrey', label='realised')
plt.plot(sqrt(garch * 252 / 5), color='red', label='GARCH forecast')
plt.ylim(0, 1.3)
plt.legend()
plt.title('GARCH forecasts of next-week volatility (annualised)')
plt.show()""")
nb.save()

# ------------------------------------------------------------------------------------------------
DM_STEPS = """d = results['QLIKE_HAR'] - results['QLIKE_GARCH']      # step 1: daily difference in loss
n = len(d)
d_mean = d.mean()                                      # step 2: average difference
print('n =', n, '| average difference =', round(d_mean, 4))"""

nb = NB("13_Comparing_HAR_and_GARCH", "13. Comparing HAR and GARCH (and the Diebold–Mariano test)", r"""
Both benchmarks are ready. Which one is more accurate — overall and in each crisis? And is the difference **real**, or could it be luck? That second question is answered by the **Diebold–Mariano test**, built here step by step.

* **Needs:** `work/har_forecasts.csv`, `work/garch_forecasts.csv`, `work/dataset.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv, DataFrame
har = read_csv(folder + 'work/har_forecasts.csv', index_col=0, parse_dates=True)
garch = read_csv(folder + 'work/garch_forecasts.csv', index_col=0, parse_dates=True)
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)

results = DataFrame({'rv5_fwd': dataset.loc[har.index, 'rv5_fwd'],
                     'HAR': har['HAR'],
                     'GARCH': garch['GARCH'],
                     'Naive': dataset.loc[har.index, 'rv_w'] * 5,
                     'window': dataset.loc[har.index, 'window']})
results.head()""")
nb.md("## QLIKE for every day and every model")
nb.code("""from numpy import log
ratio = results['rv5_fwd'] / results['HAR']
results['QLIKE_HAR'] = ratio - log(ratio) - 1
ratio = results['rv5_fwd'] / results['GARCH']
results['QLIKE_GARCH'] = ratio - log(ratio) - 1
ratio = results['rv5_fwd'] / results['Naive']
results['QLIKE_Naive'] = ratio - log(ratio) - 1
results[['QLIKE_HAR', 'QLIKE_GARCH', 'QLIKE_Naive']].mean().round(3)""")
nb.md("Over all 4,230 test days GARCH has the lowest QLIKE (0.380), then HAR (0.414). In each window:")
nb.code("""results.groupby('window')[['QLIKE_HAR', 'QLIKE_GARCH', 'QLIKE_Naive']].mean().round(3)""")
nb.md("""HAR is clearly worse in the GFC (0.676 vs 0.401): it was trained only on the calm 2003–06 years and reacted too slowly. In the Irish crisis the two are almost equal.

## The Diebold–Mariano test, step by step
**Question:** is the average difference in QLIKE between HAR and GARCH really different from zero?""")
nb.code(DM_STEPS)
nb.md("""**Step 3 — how much does `d` vary?** Neighbouring days are related (the 5-day targets overlap), so besides the normal variance we also add how each day relates to the 1, 2, 3 and 4 days before it (*autocovariances*):""")
nb.code("""dc = d - d_mean
gamma0 = (dc * dc).sum() / n
gamma1 = (dc * dc.shift(1)).sum() / n
gamma2 = (dc * dc.shift(2)).sum() / n
gamma3 = (dc * dc.shift(3)).sum() / n
gamma4 = (dc * dc.shift(4)).sum() / n
long_run_variance = gamma0 + 2 * (gamma1 + gamma2 + gamma3 + gamma4)
print('long-run variance =', long_run_variance)""")
nb.md("**Step 4 — the test statistic**, like a t-statistic: the average divided by its standard error, with a small-sample correction (Harvey, Leybourne & Newbold, *HLN*):")
nb.code("""from numpy import sqrt
dm = d_mean / sqrt(long_run_variance / n)
h = 5
dm_hln = dm * sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
print('DM statistic =', round(dm_hln, 2))""")
nb.md("**Step 5 — the p-value** from the t distribution: the probability of a difference this large if the two models were really equally good.")
nb.code("""from scipy.stats import t
p_value = 2 * t.sf(abs(dm_hln), df=n - 1)
print('p-value =', round(p_value, 4))""")
nb.md("""p = 0.0007 < 0.05: GARCH is **significantly** more accurate than HAR (the average difference is positive = HAR's loss is higher).

## The same steps in a function
We will need this test again (notebooks 18 and 24), so the steps go into one function — nothing new, just the five steps above:""")
nb.code("""def dm_test(loss_a, loss_b):
    d = loss_a - loss_b
    n = len(d)
    d_mean = d.mean()
    dc = d - d_mean
    long_run_variance = (dc * dc).sum() / n
    for k in [1, 2, 3, 4]:
        long_run_variance = long_run_variance + 2 * (dc * dc.shift(k)).sum() / n
    dm = d_mean / sqrt(long_run_variance / n)
    dm_hln = dm * sqrt((n + 1 - 10 + 20 / n) / n)
    p_value = 2 * t.sf(abs(dm_hln), df=n - 1)
    print('average difference =', round(d_mean, 4), '| DM =', round(dm_hln, 2), '| p-value =', round(p_value, 4))


dm_test(results['QLIKE_HAR'], results['QLIKE_GARCH'])""")
nb.md("With MSE instead of QLIKE (×1,000,000 to make the numbers readable):")
nb.code("""results['MSE_HAR'] = (results['rv5_fwd'] - results['HAR']) ** 2 * 1000000
results['MSE_GARCH'] = (results['rv5_fwd'] - results['GARCH']) ** 2 * 1000000
dm_test(results['MSE_HAR'], results['MSE_GARCH'])""")
nb.save()

# ------------------------------------------------------------------------------------------------
PREP_LSTM = """from pandas import read_csv
dataset = read_csv(folder + 'work/model_data.csv', index_col=0, parse_dates=True)
x = dataset[['ret', 'log_r2', 'us_ret', 'log_us_r2', 'euro_ret', 'log_euro_r2']]
y = dataset['log_rv5_fwd']
print(x.shape)
print(y.shape)"""

DAYS_2007 = """fit_days = y.iloc[21:].loc[:'2005-12-31'].dropna().index
fit_days = fit_days[:-5]
val_days = y.loc['2006-01-01':'2006-12-31'].dropna().index
val_days = val_days[:-5]
test_days = y.loc['2007-01-01':'2007-12-31'].dropna().index
print('fitting   :', len(fit_days), 'days,', fit_days[0].date(), 'to', fit_days[-1].date())
print('validation:', len(val_days), 'days,', val_days[0].date(), 'to', val_days[-1].date())
print('test      :', len(test_days), 'days,', test_days[0].date(), 'to', test_days[-1].date())"""

SCALING = """mean = x.loc[:fit_days[-1]].mean()
std = x.loc[:fit_days[-1]].std()
x_scaled = (x - mean) / std

y_mean = y.loc[fit_days].mean()
y_std = y.loc[fit_days].std()
y_scaled = (y - y_mean) / y_std"""

nb = NB("14_LSTM_Input_Windows", "14. Preparing the data for the LSTM (22-day windows)", r"""
**Part 3 — The LSTM.** The class models (decision tree, random forest, SVR) look at **one row** at a time. An LSTM looks at a **sequence**: for each forecast it reads the last **22 trading days** (about a month) of the 6 inputs — a small table of 22 rows × 6 columns. This notebook builds those tables for the 2007 model.

* **Needs:** `work/model_data.csv`. **Makes:** nothing new (notebook 15 repeats these steps and trains the model).
""")
nb.code(SETUP)
nb.code(PREP_LSTM)
nb.md("""## What is a 'window'? A small example
Ten days of one input, and windows of 3 days. The window for day 5 is days 3, 4 and 5 — always **up to and including** the forecast day, never after it.""")
nb.code("""from numpy import arange
small = arange(1, 11)
print('all days         :', small)
print('window for day 5 :', small[2:5])
print('window for day 10:', small[7:10])""")
nb.md("""## Three groups of days for the 2007 model
* **fitting days** — the model learns from them (2003 – 2005);
* **validation days** — the last 12 months before the test year (2006). They are **not** learned from; they tell us **when to stop training** and give the smearing factor;
* **test days** — 2007, the forecasts we score.

Rules: the first 21 days cannot be used (their 22-day window would start before the data), and the last 5 days of each group are dropped (their target reaches into the next group).""")
nb.code(DAYS_2007)
nb.md("""## Scaling (fitted on the fitting days only)
The same idea as `StandardScaler` in notebook 07: subtract the mean and divide by the standard deviation, both learned **only** from the data up to the last fitting day. The target y is scaled too (and the forecast is turned back at the end).""")
nb.code(SCALING)
nb.code("x_scaled.loc['2003-01-01':'2005-12-31'].describe().round(2)")
nb.md("## One window\nThe window of one fitting day: `get_loc` gives the row number of the day, and we take the 22 rows ending at it.")
nb.code("""day = fit_days[100]
position = x.index.get_loc(day)
window = x_scaled.iloc[position - 21:position + 1]
print('forecast day:', day.date())
window""")
nb.md("## All windows\nA `for` loop does this for every fitting day and puts the windows in a list; `array` turns the list into one 3-dimensional block: (number of days, 22 days, 6 inputs).")
nb.code("""from numpy import array
x_fit = []
for day in fit_days:
    position = x.index.get_loc(day)
    x_fit.append(x_scaled.values[position - 21:position + 1])
x_fit = array(x_fit, dtype='float32')
y_fit = array(y_scaled.loc[fit_days], dtype='float32')
print(x_fit.shape)
print(y_fit.shape)""")
nb.md("""So the 2007 model learns from 734 examples; each example is a 22 × 6 table, and its answer is one number (the scaled log volatility of the following week). Notebook 15 builds and trains the LSTM on exactly these.""")
nb.save()

# ------------------------------------------------------------------------------------------------
WINDOW_LOOPS = """from numpy import array
x_fit = []
for day in fit_days:
    position = x.index.get_loc(day)
    x_fit.append(x_scaled.values[position - 21:position + 1])
x_val = []
for day in val_days:
    position = x.index.get_loc(day)
    x_val.append(x_scaled.values[position - 21:position + 1])
x_test = []
for day in test_days:
    position = x.index.get_loc(day)
    x_test.append(x_scaled.values[position - 21:position + 1])

import torch
x_fit = torch.tensor(array(x_fit, dtype='float32'))
y_fit = torch.tensor(array(y_scaled.loc[fit_days], dtype='float32'))
x_val = torch.tensor(array(x_val, dtype='float32'))
y_val = torch.tensor(array(y_scaled.loc[val_days], dtype='float32'))
x_test = torch.tensor(array(x_test, dtype='float32'))
print(x_fit.shape, x_val.shape, x_test.shape)"""

LSTM_CLASS = """from torch import nn
torch.set_num_threads(2)


class LSTMVol(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, num_layers=2, batch_first=True, dropout=0.2)   # 6 inputs, 64 units, 2 layers
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(64, 1)                                                # one number out

    def forward(self, x):
        out, _ = self.lstm(x)                  # read the 22 days, one after the other
        last_day = out[:, -1]                  # the memory after the last day
        return self.head(self.drop(last_day)).squeeze(-1)"""

nb = NB("15_LSTM_One_Model", "15. LSTM — one model (Method #1)", r"""
The LSTM version of the class steps `DT_Regressor1 = DecisionTreeRegressor(...)` (building), `.fit(x_train, y_train)` (training) and `.predict(x_test)` (testing). We train **one** LSTM: trained on 2003–2006, tested on 2007, random seed 1. It takes about a minute.

* **Needs:** `work/model_data.csv`. **Makes:** `work/lstm_2007_seed1.pt` (the trained model) and two small files with its scaling numbers (used for SHAP in notebook 21).
""")
nb.code(SETUP)
nb.md("## Data Preparation (the steps of notebook 14)")
nb.code(PREP_LSTM)
nb.code(DAYS_2007)
nb.code(SCALING)
nb.code(WINDOW_LOOPS)
nb.md("""## How an LSTM works, in short
A **Long Short-Term Memory** network reads the 22 days **one after the other** and keeps a **memory**. At every day, three learned 'gates' decide what to **forget**, what new information to **store**, and what to **pass on**. After the 22nd day, a final layer turns the memory into one number: the forecast.

In class, `DecisionTreeRegressor()` came ready-made from scikit-learn. A neural network is described with a small **class** — a blueprint with two parts: `__init__` lists the layers, `forward` says how the data flows through them.

| Layer | What it does |
|---|---|
| `nn.LSTM(6, 64, num_layers=2, …)` | two stacked LSTM layers; 6 inputs per day, a memory of 64 numbers |
| `nn.Dropout(0.2)` | during training, switches off 20% of the memory at random, so the model cannot just memorise (prevents over-fitting) |
| `nn.Linear(64, 1)` | turns the 64 memory numbers into one forecast |""")
nb.code(LSTM_CLASS)
nb.md("# LSTM\n## Modelling\nThe random **seed** fixes the random starting values, so the result can be repeated exactly (like `random_state=100` in class).")
nb.code("""torch.manual_seed(1)
from numpy import random
random.seed(1)
LSTM_Regressor1 = LSTMVol()         # building
print(LSTM_Regressor1)""")
nb.md("""### Training
* **Loss:** mean squared error on the scaled log volatility. **Optimiser:** `Adam` with learning rate 0.001 — the method that adjusts the weights a little after each batch.
* **Epoch** = one pass through all fitting windows, in **batches of 64** (shuffled each epoch).
* After every epoch we compute the loss on the **validation** days. If it has not improved for **15 epochs**, we stop (**early stopping**) and keep the best weights — this prevents over-fitting.""")
nb.code("""optimizer = torch.optim.Adam(LSTM_Regressor1.parameters(), lr=0.001)
loss_function = nn.MSELoss()""")
nb.code("""best_val_loss = 1000000.0
wait = 0
for epoch in range(1, 201):
    LSTM_Regressor1.train()                              # training mode (dropout on)
    order = torch.randperm(len(x_fit))                   # shuffle the fitting windows
    for start in range(0, len(x_fit), 64):               # batches of 64
        batch = order[start:start + 64]
        optimizer.zero_grad()
        loss = loss_function(LSTM_Regressor1(x_fit[batch]), y_fit[batch])
        loss.backward()                                  # how should each weight change?
        optimizer.step()                                 # change the weights a little

    LSTM_Regressor1.eval()                               # evaluation mode (dropout off)
    with torch.no_grad():
        val_loss = loss_function(LSTM_Regressor1(x_val), y_val).item()
    print('epoch', epoch, '| validation loss', round(val_loss, 4))

    if val_loss < best_val_loss - 0.0001:                # better than before: remember these weights
        best_val_loss = val_loss
        best_weights = {}
        for name, value in LSTM_Regressor1.state_dict().items():
            best_weights[name] = value.clone()
        wait = 0
    else:
        wait = wait + 1
        if wait >= 15:
            print('no improvement for 15 epochs: stop')
            break

LSTM_Regressor1.load_state_dict(best_weights)           # keep the best weights
LSTM_Regressor1.eval()
print('best validation loss =', round(best_val_loss, 4))""")
nb.md(r"""### The smearing factor
As for HAR (notebook 09), the forecast comes out in logs and $\exp$ of a log forecast is a little too small on average. The correction is measured on the **validation** days.""")
nb.code("""from numpy import exp
with torch.no_grad():
    val_pred_scaled = LSTM_Regressor1(x_val).numpy()
val_pred_log = val_pred_scaled * y_std + y_mean
smear = exp(y.loc[val_days].values - val_pred_log).mean()
print('smearing factor =', round(smear, 3))""")
nb.md("### Testing\nThe test windows go through the trained model; then we undo the scaling (× std + mean), undo the log (`exp`) and apply the smearing factor.")
nb.code("""from pandas import Series
with torch.no_grad():
    test_pred_scaled = LSTM_Regressor1(x_test).numpy()
y_pred1 = Series(exp(test_pred_scaled * y_std + y_mean) * smear, index=test_days)    # testing
y_pred1.head()""")
nb.md("## Evaluation")
nb.code("""from numpy import log
y_test = dataset.loc[test_days, 'rv5_fwd']
ratio = y_test / y_pred1
QLIKE1 = (ratio - log(ratio) - 1).mean()
print('QLIKE (LSTM, one model, 2007) =', round(QLIKE1, 3))""")
nb.md("One LSTM is worse than GARCH (0.52) and HAR (0.66) in 2007. The study averages **5** models (seeds) per year — notebook 17. Check against the saved forecasts of seed 1:")
nb.code("""saved = read_csv(folder + 'results/lstm_forecasts.csv', index_col=0, parse_dates=True)
print('largest relative difference:', ((y_pred1 / saved.loc[test_days, 'LSTM_seed1']) - 1).abs().max())""")
nb.code("""from numpy import sqrt
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(sqrt(y_test * 252 / 5), color='lightgrey', label='realised')
plt.plot(sqrt(y_pred1 * 252 / 5), color='green', label='LSTM forecast (one model)')
plt.legend()
plt.title('2007: the LSTM trained only on calm years meets the start of the crisis')
plt.show()""")
nb.md("## Save the model\nLike `joblib.dump(best_svc, 'MysvcModel.pkl')` in class. For PyTorch we save the weights with `torch.save`, and the scaling numbers in two small CSV files.")
nb.code("""from pandas import DataFrame
torch.save(LSTM_Regressor1.state_dict(), folder + 'work/lstm_2007_seed1.pt')
DataFrame({'mean': mean, 'std': std}).to_csv(folder + 'work/lstm_2007_seed1_scaling.csv')
DataFrame({'value': [y_mean, y_std, smear]}, index=['y_mean', 'y_std', 'smear']).to_csv(folder + 'work/lstm_2007_seed1_target.csv')
print('saved')""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("16_LSTM_Choosing_Settings", "16. LSTM — choosing the settings (Method #2: the grid search)", r"""
In class, `GridSearchCV(estimator, param_grid, scoring, cv=5)` tried every combination of settings and kept the best (`best_params_`). This notebook shows how the LSTM's settings were chosen in the same spirit — and why `GridSearchCV` itself could not be used.

* **Needs:** `results/lstm_tuning.csv` (the saved search). **Makes:** nothing new.
""")
nb.code(SETUP)
nb.md("""## Why not `GridSearchCV`?
`cv=5` cuts the data into 5 **random** parts and tests on each in turn — for time series that again means training on the future (notebook 07). So the search was done by hand, with **time-ordered** validation:

* **grid:** look-back {22, 66} days × units {32, 64} × layers {1, 2} = **8 settings**;
* each setting trained for the **first two test years only** (2007 and 2008; validation years 2006 and 2007), with 2 seeds → 32 models;
* **score:** the average validation loss (lower is better);
* the best setting is then **frozen** for all years — the later years never influence the choice.

Training 32 models takes about 15 minutes, so we load the saved search.""")
nb.code("""from pandas import read_csv
tuning = read_csv(folder + 'results/lstm_tuning.csv')
print(tuning.shape)
tuning.head(8)""")
nb.md("Each row is one trained model. The average validation loss of each setting (`groupby` the three settings):")
nb.code("""grid_scores = tuning.groupby(['lookback', 'hidden', 'layers'])['val_loss'].mean().sort_values()
grid_scores.round(3)""")
nb.code("""best_params = grid_scores.index[0]
print('best_params: lookback =', best_params[0], '| units =', best_params[1], '| layers =', best_params[2])
print('best_score (validation loss) =', round(grid_scores.iloc[0], 3))""")
nb.md("""**22 days, 64 units, 2 layers** — the setting used in notebook 15 and for every year.

All scores are above 1.0, i.e. worse than simply predicting the average: the validation years 2006 and 2007 (the start of the crisis) look very different from the calm 2003–05 training years. That is the same problem the 'learning from history' question (notebook 19) is about.""")
nb.save()

# ------------------------------------------------------------------------------------------------
FUNCTIONS = '''from pandas import Timestamp, DateOffset, Timedelta, Series
from numpy import array, exp, random


def train_lstm(train_end, seed):
    """Notebook 15 in one function: train one LSTM on the data up to train_end."""
    # the three groups of days (notebook 14)
    train_end = Timestamp(train_end)
    val_start = train_end - DateOffset(years=1) + Timedelta(days=1)
    fit_days = y.iloc[21:].loc[:val_start - Timedelta(days=1)].dropna().index[:-5]
    val_days = y.loc[val_start:train_end].dropna().index[:-5]
    # scaling
    mean = x.loc[:fit_days[-1]].mean()
    std = x.loc[:fit_days[-1]].std()
    x_scaled = (x - mean) / std
    y_mean = y.loc[fit_days].mean()
    y_std = y.loc[fit_days].std()
    y_scaled = (y - y_mean) / y_std
    # windows
    x_fit = []
    for day in fit_days:
        position = x.index.get_loc(day)
        x_fit.append(x_scaled.values[position - 21:position + 1])
    x_val = []
    for day in val_days:
        position = x.index.get_loc(day)
        x_val.append(x_scaled.values[position - 21:position + 1])
    x_fit = torch.tensor(array(x_fit, dtype='float32'))
    y_fit = torch.tensor(array(y_scaled.loc[fit_days], dtype='float32'))
    x_val = torch.tensor(array(x_val, dtype='float32'))
    y_val = torch.tensor(array(y_scaled.loc[val_days], dtype='float32'))
    # building
    torch.manual_seed(seed)
    random.seed(seed)
    model = LSTMVol()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    # training with early stopping
    best_val_loss = 1000000.0
    wait = 0
    for epoch in range(1, 201):
        model.train()
        order = torch.randperm(len(x_fit))
        for start in range(0, len(x_fit), 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            loss_function(model(x_fit[batch]), y_fit[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(x_val), y_val).item()
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            best_weights = {}
            for name, value in model.state_dict().items():
                best_weights[name] = value.clone()
            wait = 0
        else:
            wait = wait + 1
            if wait >= 15:
                break
    model.load_state_dict(best_weights)
    model.eval()
    # smearing factor on the validation days
    with torch.no_grad():
        val_pred_log = model(x_val).numpy() * y_std + y_mean
    smear = exp(y.loc[val_days].values - val_pred_log).mean()
    return {'model': model, 'mean': mean, 'std': std, 'y_mean': y_mean, 'y_std': y_std, 'smear': smear}


def forecast_lstm(m, days):
    """Testing: forecasts (variance) of a trained model for the given days."""
    x_scaled = (x - m['mean']) / m['std']
    x_days = []
    for day in days:
        position = x.index.get_loc(day)
        x_days.append(x_scaled.values[position - 21:position + 1])
    with torch.no_grad():
        pred_scaled = m['model'](torch.tensor(array(x_days, dtype='float32'))).numpy()
    return Series(exp(pred_scaled * m['y_std'] + m['y_mean']) * m['smear'], index=days)'''

nb = NB("17_LSTM_All_Years", "17. LSTM — all years and 5 seeds (85 models)", r"""
The walk-forward for the LSTM: for each test year 2007–2023, train **5** LSTMs (seeds 1–5) on the data up to 31 December of the year before, forecast the year, and **average** the 5 forecasts. 17 years × 5 seeds = **85 models**.

To repeat notebook 15 85 times, its steps are put into two **functions**: `train_lstm(...)` (building + training) and `forecast_lstm(...)` (testing). Nothing new happens inside them.

* **Needs:** `work/model_data.csv`. **Makes:** `work/lstm_forecasts.csv`.
""")
nb.code(SETUP)
nb.code(PREP_LSTM)
nb.code("import torch\n" + LSTM_CLASS)
nb.md("## The two functions (notebook 15 in one block)")
nb.code(FUNCTIONS)
nb.md("## Test the functions: seed 1, test year 2007\nThis must give exactly the forecasts of notebook 15.")
nb.code("""test_days = y.loc['2007-01-01':'2007-12-31'].dropna().index
m = train_lstm('2006-12-31', 1)
y_pred1 = forecast_lstm(m, test_days)
saved = read_csv(folder + 'results/lstm_forecasts.csv', index_col=0, parse_dates=True)
print('largest relative difference from the saved seed-1 forecasts:', ((y_pred1 / saved.loc[test_days, 'LSTM_seed1']) - 1).abs().max())""")
nb.md("""## All 85 models
This takes about 30 minutes on a laptop and roughly 1–1.5 hours on Colab's CPU. Set `run_all = True` to train everything here; with `run_all = False` the saved forecasts of the 85 models are loaded.""")
nb.code("""run_all = False

if run_all:
    from pandas import concat, DataFrame
    seed_forecasts = {1: [], 2: [], 3: [], 4: [], 5: []}
    for year in range(2007, 2024):
        test_days = y.loc[str(year) + '-01-01':str(year) + '-12-31'].dropna().index
        for seed in [1, 2, 3, 4, 5]:
            m = train_lstm(str(year - 1) + '-12-31', seed)               # building + training
            seed_forecasts[seed].append(forecast_lstm(m, test_days))     # testing
        print(year, 'done')
    lstm = DataFrame()
    for seed in [1, 2, 3, 4, 5]:
        lstm['LSTM_seed' + str(seed)] = concat(seed_forecasts[seed])
    lstm['LSTM'] = lstm.mean(axis=1)                                   # the average of the 5 seeds
else:
    lstm = read_csv(folder + 'results/lstm_forecasts.csv', index_col=0, parse_dates=True)

lstm = lstm[['LSTM', 'LSTM_seed1', 'LSTM_seed2', 'LSTM_seed3', 'LSTM_seed4', 'LSTM_seed5']]
print(len(lstm), 'forecast days')
lstm.head()""")
nb.md("""## Why average 5 seeds?
Two LSTMs trained on the same data but with different random starts give different forecasts. How different? The spread between the seeds, as a share of their average:""")
nb.code("""spread = lstm[['LSTM_seed1', 'LSTM_seed2', 'LSTM_seed3', 'LSTM_seed4', 'LSTM_seed5']].std(axis=1) / lstm['LSTM']
print('average spread between seeds:', round(spread.mean() * 100, 1), '%')""")
nb.md("About 13%: one seed alone would be noticeably random. The average of 5 is more stable — and in notebook 23 we ask whether the **explanations** are stable across seeds too.")
nb.code("""lstm.to_csv(folder + 'work/lstm_forecasts.csv')
print('saved')""")
nb.save()

# ------------------------------------------------------------------------------------------------
DM_FUNCTION = """from numpy import sqrt
from scipy.stats import t


def dm_test(loss_a, loss_b):
    '''The Diebold-Mariano test of notebook 13 (negative difference = model A more accurate).'''
    d = loss_a - loss_b
    n = len(d)
    d_mean = d.mean()
    dc = d - d_mean
    long_run_variance = (dc * dc).sum() / n
    for k in [1, 2, 3, 4]:
        long_run_variance = long_run_variance + 2 * (dc * dc.shift(k)).sum() / n
    dm = d_mean / sqrt(long_run_variance / n)
    dm_hln = dm * sqrt((n + 1 - 10 + 20 / n) / n)
    p_value = 2 * t.sf(abs(dm_hln), df=n - 1)
    print('average difference =', round(d_mean, 4), '| DM =', round(dm_hln, 2), '| p-value =', round(p_value, 4))
    return d_mean, p_value"""

nb = NB("18_Comparing_All_Models", "18. Comparing all models (research question 1)", r"""
**RQ1 / hypothesis H1:** *the LSTM forecasts Irish volatility more accurately than GARCH and HAR.* We put the three models side by side, overall and in each window, and test the differences.

* **Needs:** `work/har_forecasts.csv`, `work/garch_forecasts.csv`, `work/lstm_forecasts.csv`, `work/dataset.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv, DataFrame
har = read_csv(folder + 'work/har_forecasts.csv', index_col=0, parse_dates=True)
garch = read_csv(folder + 'work/garch_forecasts.csv', index_col=0, parse_dates=True)
lstm = read_csv(folder + 'work/lstm_forecasts.csv', index_col=0, parse_dates=True)
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)

results = DataFrame({'rv5_fwd': dataset.loc[har.index, 'rv5_fwd'],
                     'LSTM': lstm['LSTM'],
                     'GARCH': garch['GARCH'],
                     'HAR': har['HAR'],
                     'window': dataset.loc[har.index, 'window']})
results.head()""")
nb.code("""from numpy import log
ratio = results['rv5_fwd'] / results['LSTM']
results['QLIKE_LSTM'] = ratio - log(ratio) - 1
ratio = results['rv5_fwd'] / results['GARCH']
results['QLIKE_GARCH'] = ratio - log(ratio) - 1
ratio = results['rv5_fwd'] / results['HAR']
results['QLIKE_HAR'] = ratio - log(ratio) - 1
results[['QLIKE_LSTM', 'QLIKE_GARCH', 'QLIKE_HAR']].mean().round(3)""")
nb.md("Over all days the LSTM has the **highest** loss (0.463). By window:")
nb.code("""by_window = results.groupby('window')[['QLIKE_LSTM', 'QLIKE_GARCH', 'QLIKE_HAR']].mean()
by_window.round(3)""")
nb.md("""The picture changes with the crisis: the LSTM is the **best** model in COVID-19 (0.562) and 2022 (0.264), but by far the worst in the GFC (1.021) and Brexit.""")
nb.code("""by_window.plot(kind='bar', figsize=(11, 4), color=['green', 'red', 'blue'], title='Mean QLIKE by window (lower is better)')""")
nb.md("## Are the differences significant? (Diebold–Mariano, from notebook 13)")
nb.code(DM_FUNCTION)
nb.code("""print('LSTM vs GARCH:')
diff_garch, p_garch = dm_test(results['QLIKE_LSTM'], results['QLIKE_GARCH'])
print('LSTM vs HAR:')
diff_har, p_har = dm_test(results['QLIKE_LSTM'], results['QLIKE_HAR'])""")
nb.md("""## The verdict on H1
H1 is supported only if the LSTM's loss is **lower** (negative difference) **and** significant (p < 0.05) against **both** benchmarks.""")
nb.code("""if diff_garch < 0 and p_garch < 0.05 and diff_har < 0 and p_har < 0.05:
    print('H1 supported')
else:
    print('H1 NOT supported: the LSTM is not more accurate than both benchmarks')""")
nb.md("""The differences are positive and significant: over 2007–2023 the LSTM is significantly **less** accurate than GARCH (p = 0.003) and HAR (p = 0.04). This is a genuine finding because the hypothesis was written down before the results (pre-registration).""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("19_Learning_From_History", "19. Learning from history — frozen models (research question 3)", r"""
*This time is different?* For each crisis, 5 LSTMs are trained **only on the data before the crisis** and never updated ('frozen'); they forecast the whole crisis window. Their QLIKE is divided by HAR's QLIKE in the same window: **above 1 = worse than HAR**.

**H3** (written before the results, the 'Minsky test'): the GFC model, which had seen only the calm 2003–07 boom, suffers most; then COVID; the Irish-crisis model (which had already seen the GFC) least.

* **Needs:** the forecasts in `work/` (and `work/model_data.csv` if you re-train). **Makes:** nothing new.
""")
nb.code(SETUP)
nb.md("""## The saved results
Training the 15 frozen models takes about 5–10 minutes on Colab; set `run_frozen = True` further below to do it. First the saved table:""")
nb.code("""from pandas import read_csv
frozen_table = read_csv(folder + 'results/evaluation_frozen.csv', index_col=0)
frozen_table[['trained_until', 'days', 'ratio_frozen_to_HAR', 'ratio_refitted_to_HAR']].round(2)""")
nb.md("""* **frozen** = trained only before the crisis; **refitted** = the normal model of notebook 17 (re-trained every year).
* GFC: the frozen model was **4.3 times** worse than HAR. COVID: 0.74 — **better** than HAR, although it had never seen a pandemic.""")
nb.code("""ranking = frozen_table['ratio_frozen_to_HAR'].sort_values(ascending=False)
print('observed ranking (worst first):', list(ranking.index))
print('expected ranking (H3)         :', ['Global Financial Crisis', 'COVID-19', 'Irish sovereign debt crisis'])
if list(ranking.index) == ['Global Financial Crisis', 'COVID-19', 'Irish sovereign debt crisis']:
    print('H3 supported')
else:
    print('H3 NOT supported as ranked (but the GFC is the worst, as expected)')""")
nb.code("""frozen_table[['ratio_frozen_to_HAR', 'ratio_refitted_to_HAR']].plot(kind='bar', figsize=(8, 4), color=['grey', 'green'], title='LSTM QLIKE / HAR QLIKE in each crisis (1 = as good as HAR)')""")
nb.md("""## Optional: train the frozen models here
The same functions as notebook 17, with a different last training day: the day before each window starts.""")
nb.code("""run_frozen = False""")
nb.md("The data, the LSTM blueprint and the two functions — exactly as in notebook 17:")
nb.code("""import torch
from numpy import log
model_data = read_csv(folder + 'work/model_data.csv', index_col=0, parse_dates=True)
x = model_data[['ret', 'log_r2', 'us_ret', 'log_us_r2', 'euro_ret', 'log_euro_r2']]
y = model_data['log_rv5_fwd']
har = read_csv(folder + 'work/har_forecasts.csv', index_col=0, parse_dates=True)
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)""")
nb.code(LSTM_CLASS)
nb.code(FUNCTIONS)
nb.md("The loop: for each crisis, 5 frozen models, their average forecast, and the ratio to HAR.")
nb.code("""if run_frozen:
    last_training_day = {'Global Financial Crisis': '2007-08-08',
                         'Irish sovereign debt crisis': '2010-04-22',
                         'COVID-19': '2020-02-18'}
    for crisis in last_training_day:
        days = har.index[dataset.loc[har.index, 'window'] == crisis]
        frozen = 0
        for seed in [1, 2, 3, 4, 5]:
            m = train_lstm(last_training_day[crisis], seed)
            frozen = frozen + forecast_lstm(m, days) / 5            # the average of 5 seeds
        realised = dataset.loc[days, 'rv5_fwd']
        ratio = realised / frozen
        q_frozen = (ratio - log(ratio) - 1).mean()
        ratio = realised / har.loc[days, 'HAR']
        q_har = (ratio - log(ratio) - 1).mean()
        print(crisis, '| frozen LSTM / HAR =', round(q_frozen / q_har, 2))""")
nb.save()

"""Notebooks 01-08 of the simple series (Part 1: the data; Part 2 starts with forecast errors)."""
from series_common import NB, SETUP

# ------------------------------------------------------------------------------------------------
nb = NB("01_Data_Reading", "01. Data Reading", r"""
**Part 1 — The data.** In this first notebook we only **read** the four price files and **look** at them — exactly like the class notebook *Data_Reading (Employee Dataset)*.

* **Needs:** the folder `ThisTimeIsDifferent` in `MyDrive/DataSets/` on Google Drive.
* **Makes:** nothing new (we only look).
""")
nb.md("## Connect Google Drive\nThe data files are in Google Drive. These lines connect Colab to your Drive and save the folder name in `folder`, so later we can write `folder + 'ISEQ.csv'`.")
nb.code(SETUP)
nb.md("""## Read the whole data
`ISEQ.csv` holds the daily prices of the **ISEQ Overall** index — the Irish stock market, the market whose risk we want to forecast. We read it with `read_csv`, as in class.""")
nb.code("""from pandas import read_csv
data1 = read_csv(folder + 'ISEQ.csv')""")
nb.code("data1")
nb.md("""Each **row is one trading day**. The columns are:

| Column | Meaning |
|---|---|
| `Unnamed: 0` | the date (it has no name in the file; notebook 02 fixes this) |
| `Open` | the price at the start of the day |
| `High` / `Low` | the highest / lowest price during the day |
| `Close` | the price at the end of the day — **the price we use** |
| `Volume` | how much was traded (not used in this study) |""")
nb.md("## Read from head")
nb.code("data1.head()")
nb.md("## Read from tail")
nb.code("data1.tail(3)")
nb.md("## Read a range\nExample: show the rows 100 to 115.")
nb.code("data1[100:116]")
nb.md("## Reading features (columns)")
nb.code("data1['Close']")
nb.code("data1[['Unnamed: 0', 'Close']]")
nb.md("### Example: the closing prices of rows 1500 to 1510")
nb.code("data1['Close'][1500:1511]")
nb.md("## Data size")
nb.code("data1.shape")
nb.code("""a, b = data1.shape
print('rows (trading days):', a)
print('columns:', b)""")
nb.md("## Data information")
nb.code("data1.info()")
nb.code("data1.describe()")
nb.md("""How to read `describe()`: `count` = number of days, `mean` = average, `min` / `max` = the smallest / largest value. The lowest ISEQ close in the file is about 1,900 (March 2009, the bottom of the crisis); the highest is about 13,100 (December 2025). Before the crisis it had peaked near 10,000 (February 2007).""")
nb.md("""## The other three markets
The same reading for the three foreign markets:

| File | Market | Why it is in the study |
|---|---|---|
| `GSPC.csv` | S&P 500 (United States) | the **US channel** — the 2008 crisis started in the US |
| `GDAXI.csv` | DAX (Germany) | the **euro channel** — the core of the euro area |
| `FTSE.csv` | FTSE 100 (United Kingdom) | used only in a robustness check |""")
nb.code("""data2 = read_csv(folder + 'GSPC.csv')
data3 = read_csv(folder + 'GDAXI.csv')
data4 = read_csv(folder + 'FTSE.csv')
print(data2.shape)
print(data3.shape)
print(data4.shape)""")
nb.code("data2.head()")
nb.md("""The four markets have **different numbers of rows** because they close on different holidays (for example, New York is closed on 4 July, Dublin is not). Notebook 05 shows how we line them up.

### What we learned
* Four files, one row per trading day, from October 2002 to December 2025.
* We will use the **closing price** of each market.""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("02_Date_Time", "02. Date and Time", r"""
In the class notebook *DateTime (Bike Rental)* the text column `datetime` was turned into real dates with `to_datetime`, and the year, month and day were taken out with `.dt.year`, `.dt.month`… We do the same with our dates.

* **Needs:** `ISEQ.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
data1 = read_csv(folder + 'ISEQ.csv')
data1.info()""")
nb.md("""The first column has no name (`Unnamed: 0`) and its type is `object` — that means **text**, not a date. First we give it a name.""")
nb.code("""data1 = data1.rename(columns={'Unnamed: 0': 'Date'})
data1['Date']""")
nb.md("## Change the text into dates")
nb.code("""from pandas import to_datetime
data1['Date'] = to_datetime(data1['Date'])
data1.info()""")
nb.md("Now the type of `Date` is `datetime64` — a real date. So we can take out its parts, like in the Bike Rental notebook:")
nb.code("""data1['year'] = data1['Date'].dt.year
data1['month'] = data1['Date'].dt.month
data1['day'] = data1['Date'].dt.day
data1['day_week'] = data1['Date'].dt.dayofweek
data1.head()""")
nb.md("`day_week`: 0 = Monday, 1 = Tuesday, … 4 = Friday. Let us count:")
nb.code("data1['day_week'].value_counts().sort_index()")
nb.md("There is no 5 (Saturday) or 6 (Sunday): the stock market is closed at weekends. How many trading days per year?")
nb.code("data1['year'].value_counts().sort_index()")
nb.md("About 250–255 trading days per year (2002 and 2025 are not complete years in the file).")
nb.md("""## Use the date as the index
If the date becomes the **index** (the row label), pandas can select periods with `.loc`. This is how we will choose training years, test years and crisis windows later.""")
nb.code("""data1 = data1.set_index('Date')
data1.head()""")
nb.md("### Example 1: the week of the Irish bank guarantee (30 September 2008)")
nb.code("data1.loc['2008-09-26':'2008-10-02', ['Close']]")
nb.md("On 29 September 2008 the ISEQ fell from 3,785 to 3,292 (about −13%). The next day the Irish government guaranteed the banks, and the market jumped back.")
nb.md("### Example 2: COVID-19, March 2020")
nb.code("data1.loc['2020-03-09':'2020-03-13', ['Close']]")
nb.md("### Example 3: a whole year")
nb.code("data1.loc['2008'].shape")
nb.md("## A first picture")
nb.code("""data1['Close'].plot(figsize=(12, 4), title='ISEQ Overall, closing price')""")
nb.md("You can see the boom until 2007, the crash of 2008–09, and the falls in 2011, 2016, 2020 and 2022.")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("03_Data_Cleaning", "03. Data Cleaning", r"""
In the Employees notebooks, missing values were filled with the **mean** (`Age`) or the **mode** (`DistanceFromHome`). For prices we check the data in the same way — but we will **not** fill anything, and we will see why.

* **Needs:** the four price files. **Makes:** `work/clean_ISEQ.csv`, `work/clean_GSPC.csv`, `work/clean_GDAXI.csv`, `work/clean_FTSE.csv`.
""")
nb.code(SETUP)
nb.md("## Data Reading (with the dates already as the index)\n`index_col=0` uses the first column as the index and `parse_dates=True` turns it into dates — the work of notebook 02 in one line.")
nb.code("""from pandas import read_csv
data1 = read_csv(folder + 'ISEQ.csv', index_col=0, parse_dates=True)
data2 = read_csv(folder + 'GSPC.csv', index_col=0, parse_dates=True)
data3 = read_csv(folder + 'GDAXI.csv', index_col=0, parse_dates=True)
data4 = read_csv(folder + 'FTSE.csv', index_col=0, parse_dates=True)
data1.head()""")
nb.md("## Missing values")
nb.code("data1.isna().sum()")
nb.code("""print(data2.isna().sum().sum())
print(data3.isna().sum().sum())
print(data4.isna().sum().sum())""")
nb.md("""No missing values. **If** there were, we would **remove** the row instead of filling it: a missing price means the market was closed, and an invented price (for example the mean) would create a fake movement.""")
nb.code("""data1 = data1.dropna(subset=['Close'])
data2 = data2.dropna(subset=['Close'])
data3 = data3.dropna(subset=['Close'])
data4 = data4.dropna(subset=['Close'])
print(data1.shape, data2.shape, data3.shape, data4.shape)""")
nb.md("## Impossible values\nA price can never be zero or negative.")
nb.code("""print((data1['Close'] <= 0).sum())
print((data2['Close'] <= 0).sum())
print((data3['Close'] <= 0).sum())
print((data4['Close'] <= 0).sum())""")
nb.code("""data1 = data1[data1['Close'] > 0]
data2 = data2[data2['Close'] > 0]
data3 = data3[data3['Close'] > 0]
data4 = data4[data4['Close'] > 0]""")
nb.md("## Duplicated dates\nEach date should appear only once.")
nb.code("""print(data1.index.duplicated().sum())
print(data2.index.duplicated().sum())
print(data3.index.duplicated().sum())
print(data4.index.duplicated().sum())""")
nb.md("""## A hidden problem: 'stale' opening prices
There is a better way to measure daily volatility that uses the **opening** price (the Garman–Klass method). But on Yahoo, the ISEQ's opening price is often just **yesterday's closing price copied** — not a real price. Let us count how often.

`shift(1)` moves a column down by one row, so each row sees **yesterday's** close:""")
nb.code("""data1['previous_close'] = data1['Close'].shift(1)
data1[['Open', 'Close', 'previous_close']].loc['2015-03-02':'2015-03-06']""")
nb.md("In this week, on four of the five days the `Open` is exactly the `previous_close`. Let us mark these days with `True`:")
nb.code("""data1['stale_open'] = (data1['Open'] - data1['previous_close']).abs() < 0.000000001
print('2003-2006:', round(data1.loc['2003':'2006', 'stale_open'].mean() * 100, 1), '%')
print('2007-2012:', round(data1.loc['2007':'2012', 'stale_open'].mean() * 100, 1), '%')
print('2013-2019:', round(data1.loc['2013':'2019', 'stale_open'].mean() * 100, 1), '%')
print('2020-2025:', round(data1.loc['2020':'2025', 'stale_open'].mean() * 100, 1), '%')""")
nb.md("""(The mean of `True`/`False` is the share of `True`.) In 2013–2019, three days out of four have a copied opening price. A volatility measure built on these would be wrong. **Decision: we use only closing prices.**""")
nb.md("## Save the clean data\nWe keep the closing price (and, for the ISEQ, the high and low prices, used in a later robustness check). The files go into a sub-folder `work`.")
nb.code("""import os
os.makedirs(folder + 'work', exist_ok=True)

data1[['High', 'Low', 'Close']].to_csv(folder + 'work/clean_ISEQ.csv')
data2[['Close']].to_csv(folder + 'work/clean_GSPC.csv')
data3[['Close']].to_csv(folder + 'work/clean_GDAXI.csv')
data4[['Close']].to_csv(folder + 'work/clean_FTSE.csv')
print('saved')""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("04_Returns_and_Volatility", "04. Returns and Volatility (building the features)", r"""
In class, *Data Encoding* turned text into numbers (`map`, `get_dummies`). Our raw material is prices, and in this notebook we turn them into the numbers the models need: **returns**, **squared returns**, and the **target** — next week's volatility.

* **Needs:** `work/clean_ISEQ.csv`. **Makes:** `work/iseq_features.csv`.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
data1 = read_csv(folder + 'work/clean_ISEQ.csv', index_col=0, parse_dates=True)
data1.head()""")
nb.md("""## 1. The daily return
The **log return** is the daily percentage change: $r_t = \\ln(P_t / P_{t-1})$.

Example by hand: the price moves from 100 to 102 → $\\ln(102/100) = 0.0198$, about +2%.""")
nb.code("""from numpy import log
print(log(102 / 100))""")
nb.md("`log(...)` of the whole column, then `.diff()` (today minus yesterday) gives every daily return at once:")
nb.code("""data1['ret'] = log(data1['Close']).diff()
data1.head()""")
nb.md("The first row is empty (`NaN`): there is no yesterday for the first day.")
nb.md("""## 2. The squared return — the size of the move
The sign of a return says up or down (direction). Its **square** is always positive and says **how big** the move was. A day of −3% and a day of +3% have the same squared return.""")
nb.code("""data1['r2'] = data1['ret'] ** 2
data1[['ret', 'r2']].loc['2008-09-26':'2008-10-02']""")
nb.md("""## 3. The target: next week's realised variance
We want to forecast, after the close of day *t*, **how much the market will move over the next five trading days**:

$RV5_t = r_{t+1}^2 + r_{t+2}^2 + r_{t+3}^2 + r_{t+4}^2 + r_{t+5}^2$

`shift(-1)` moves a column **up** by one row, so each row sees **tomorrow's** value. A small example:""")
nb.code("""demo = data1[['r2']].loc['2008-09-24':'2008-10-03'].copy()
demo['r2_tomorrow'] = demo['r2'].shift(-1)
demo['r2_in_2_days'] = demo['r2'].shift(-2)
demo""")
nb.md("Adding the next five days gives the target:")
nb.code("""data1['rv5_fwd'] = data1['r2'].shift(-1) + data1['r2'].shift(-2) + data1['r2'].shift(-3) + data1['r2'].shift(-4) + data1['r2'].shift(-5)
data1[['r2', 'rv5_fwd']].head(8)""")
nb.md("""## 4. A number we can read: annualised volatility
Variance is hard to read. Volatility (the square root), scaled to a year, is the usual way to talk about risk: **about 15–20% is a normal year**.""")
nb.code("""from numpy import sqrt
data1['vol_annual'] = sqrt(data1['rv5_fwd'] * 252 / 5)
data1['vol_annual'].describe()""")
nb.code("""data1['vol_annual'].loc['2007':'2009'].plot(figsize=(12, 4), title='Next-week volatility of the ISEQ, 2007-2009 (annualised)')""")
nb.md("In October 2008 the next-week volatility reached about 120% — six times a normal year.")
nb.md("""## 5. Why the models learn the logarithm
Volatility is very **skewed**: most weeks are calm (small numbers), a few are huge. In logs it looks much more like a bell curve, which is easier to learn.""")
nb.code("""data1['log_rv5_fwd'] = log(data1['rv5_fwd'])
data1['rv5_fwd'].plot(kind='hist', bins=100, figsize=(6, 3), title='rv5_fwd')""")
nb.code("""data1['log_rv5_fwd'].plot(kind='hist', bins=100, figsize=(6, 3), title='log of rv5_fwd')""")
nb.md("""## 6. The ingredients of the HAR model
The HAR benchmark (notebook 09) uses three averages of past squared returns:
* `rv_d` — **today**,
* `rv_w` — the average of the **last 5 days** (one week),
* `rv_m` — the average of the **last 22 days** (one month).

`rolling(5).mean()` takes the average of the last 5 rows, moving forward one row at a time:""")
nb.code("""data1['rv_d'] = data1['r2']
data1['rv_w'] = data1['r2'].rolling(5).mean()
data1['rv_m'] = data1['r2'].rolling(22).mean()
data1[['r2', 'rv_d', 'rv_w', 'rv_m']].iloc[20:26]""")
nb.md("## Save")
nb.code("""data1 = data1.drop(columns=['vol_annual'])
data1.to_csv(folder + 'work/iseq_features.csv')
data1.info()""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("05_Joining_the_Markets", "05. Joining the three markets", r"""
The models use the Irish market **and** the US and German markets. In this notebook we put them in one table — carefully, so that no information from the future is used.

* **Needs:** `work/iseq_features.csv` and the clean foreign files. **Makes:** `work/dataset.csv`.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
data1 = read_csv(folder + 'work/iseq_features.csv', index_col=0, parse_dates=True)
data2 = read_csv(folder + 'work/clean_GSPC.csv', index_col=0, parse_dates=True)
data3 = read_csv(folder + 'work/clean_GDAXI.csv', index_col=0, parse_dates=True)
data4 = read_csv(folder + 'work/clean_FTSE.csv', index_col=0, parse_dates=True)
print(data1.shape, data2.shape, data3.shape, data4.shape)""")
nb.md("""## Why we must be careful: trading hours (Irish time)

| Market | Closes at |
|---|---|
| Dublin (ISEQ), London (FTSE), Frankfurt (DAX) | about 16:30 |
| New York (S&P 500) | about **21:00** |

The forecast for the next week is made at **08:00 the next morning**. By then the closes of **all** markets on day *t* are known, so we may use them. We may **never** use a value from day *t+1*.""")
nb.md("## Returns of the foreign markets\nThe same two numbers as for the ISEQ: the return and the squared return.")
nb.code("""from numpy import log
data2['us_ret'] = log(data2['Close']).diff()
data2['us_r2'] = data2['us_ret'] ** 2
data3['euro_ret'] = log(data3['Close']).diff()
data3['euro_r2'] = data3['euro_ret'] ** 2
data4['uk_ret'] = log(data4['Close']).diff()
data4['uk_r2'] = data4['uk_ret'] ** 2
data2.head()""")
nb.md("""## A small example of `merge_asof`
`merge_asof` joins two tables by date. For every Irish day it takes the **latest** US value dated **on or before** that day (`direction='backward'`). If New York was closed (4 July), the last known value is used again.""")
nb.code("""from pandas import DataFrame, to_datetime, merge_asof
irish = DataFrame({'Date': to_datetime(['2016-07-01', '2016-07-04', '2016-07-05']), 'irish_value': [1, 2, 3]})
us = DataFrame({'Date': to_datetime(['2016-07-01', '2016-07-05']), 'us_value': [10, 30]})   # 4 July: New York closed
merge_asof(irish, us, on='Date', direction='backward')""")
nb.md("On 4 July the Irish row gets the US value of 1 July. Nothing from the future is used.")
nb.md("## Join the real data")
nb.code("""dataset = merge_asof(data1, data2[['us_ret', 'us_r2']], left_index=True, right_index=True, direction='backward')
dataset = merge_asof(dataset, data3[['euro_ret', 'euro_r2']], left_index=True, right_index=True, direction='backward')
dataset = merge_asof(dataset, data4[['uk_ret', 'uk_r2']], left_index=True, right_index=True, direction='backward')
dataset.info()""")
nb.md("How often did we have to re-use an older foreign value? (`isin` checks if each Irish date is also a US date; `~` means *not*.)")
nb.code("""print('US  :', round((~dataset.index.isin(data2.index)).mean() * 100, 1), '% of Irish days')
print('euro:', round((~dataset.index.isin(data3.index)).mean() * 100, 1), '% of Irish days')
print('UK  :', round((~dataset.index.isin(data4.index)).mean() * 100, 1), '% of Irish days')""")
nb.md("""## The sample starts in January 2003
The data begin in October 2002 only so that the 22-day averages are complete on the first day. The study itself starts after the dot-com crash, in **January 2003**.""")
nb.code("""dataset = dataset.loc['2003-01-01':]
dataset.head()""")
nb.md("## How strongly do the markets move together? (2003 – August 2023)")
nb.code("""dataset.loc[:'2023-08-31', ['ret', 'us_ret', 'euro_ret', 'uk_ret']].corr().round(2)""")
nb.md("""The Irish market moves most with London (0.71) and Frankfurt (0.69), less with New York (0.48). London and Frankfurt move together a lot (0.83) — **too much** to use both as inputs (SHAP could not tell them apart). So the main models use the **DAX**, and the FTSE is kept for a robustness check.""")
nb.code("""dataset.to_csv(folder + 'work/dataset.csv')
print(dataset.shape)""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("06_Crisis_Windows", "06. The crisis windows", r"""
The study compares the models in **calm** times and in **five crisis windows**. Here we label every day with its window, measure how wild each window was, and draw the main picture of the data.

* **Needs:** `work/dataset.csv`. **Makes:** `work/dataset.csv` again, with a new column `window`.
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
print(dataset.shape)""")
nb.md("""## The windows

| Window | From | To | Origin |
|---|---|---|---|
| Global Financial Crisis (GFC) | 9 Aug 2007 (BNP Paribas freezes funds) | 9 Mar 2009 (US market bottom) | US banking, Irish property crash at the same time |
| Irish sovereign debt crisis | 23 Apr 2010 (Greece asks for help) | 26 Jul 2012 (Draghi: 'whatever it takes') | euro-area debt and Irish banks |
| COVID-19 | 19 Feb 2020 | 30 Jun 2020 | a health shock from outside the financial system |
| War / energy shock 2022 | 10 Feb 2022 | 31 Oct 2022 | war in Ukraine, energy prices, interest rates |
| Brexit referendum | 1 Jun 2016 | 31 Jul 2016 | a UK political shock |

**Important:** the window labels are used only to **report** the results afterwards. The models never see them.""")
nb.code("""dataset['window'] = 'calm'
dataset.loc['2007-08-09':'2009-03-09', 'window'] = 'Global Financial Crisis'
dataset.loc['2010-04-23':'2012-07-26', 'window'] = 'Irish sovereign debt crisis'
dataset.loc['2020-02-19':'2020-06-30', 'window'] = 'COVID-19'
dataset.loc['2022-02-10':'2022-10-31', 'window'] = 'War / energy shock 2022'
dataset.loc['2016-06-01':'2016-07-31', 'window'] = 'Brexit referendum'
dataset.loc[:'2023-08-31', 'window'].value_counts()""")
nb.md("## How wild was each window?\nThe median annualised volatility of the next week, in each window (`groupby` = do the calculation separately for each group).")
nb.code("""from numpy import sqrt
dataset['vol_annual'] = sqrt(dataset['rv5_fwd'] * 252 / 5)
(dataset.loc[:'2023-08-31'].groupby('window')['vol_annual'].median() * 100).round(1)""")
nb.md("""## The largest fall inside each window (drawdown)
`cummax()` gives the highest price **so far**; dividing the price by it shows how far below its peak the market is.""")
nb.code("""gfc = dataset.loc['2007-08-09':'2009-03-09', 'Close']
drawdown = gfc / gfc.cummax() - 1
print('GFC largest fall:', round(drawdown.min() * 100, 1), '%')""")
nb.code("""irish = dataset.loc['2010-04-23':'2012-07-26', 'Close']
print('Irish crisis largest fall:', round((irish / irish.cummax() - 1).min() * 100, 1), '%')
covid = dataset.loc['2020-02-19':'2020-06-30', 'Close']
print('COVID largest fall:', round((covid / covid.cummax() - 1).min() * 100, 1), '%')
war = dataset.loc['2022-02-10':'2022-10-31', 'Close']
print('2022 largest fall:', round((war / war.cummax() - 1).min() * 100, 1), '%')
brexit = dataset.loc['2016-06-01':'2016-07-31', 'Close']
print('Brexit largest fall:', round((brexit / brexit.cummax() - 1).min() * 100, 1), '%')""")
nb.md("## The main picture of the data")
nb.code("""import matplotlib.pyplot as plt
from pandas import to_datetime
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax1.plot(dataset['Close'], color='black', linewidth=0.8)
ax1.set_ylabel('ISEQ Overall')
ax2.plot(dataset['vol_annual'], color='black', linewidth=0.5)
ax2.set_ylabel('next-week volatility (annualised)')
ax2.axvspan(to_datetime('2007-08-09'), to_datetime('2009-03-09'), color='blue', alpha=0.2, label='GFC')
ax2.axvspan(to_datetime('2010-04-23'), to_datetime('2012-07-26'), color='green', alpha=0.2, label='Irish debt crisis')
ax2.axvspan(to_datetime('2020-02-19'), to_datetime('2020-06-30'), color='purple', alpha=0.2, label='COVID-19')
ax2.axvspan(to_datetime('2022-02-10'), to_datetime('2022-10-31'), color='red', alpha=0.2, label='2022 shock')
ax2.axvspan(to_datetime('2016-06-01'), to_datetime('2016-07-31'), color='orange', alpha=0.4, label='Brexit')
ax2.legend()
plt.show()""")
nb.code("""dataset = dataset.drop(columns=['vol_annual'])
dataset.to_csv(folder + 'work/dataset.csv')
print('saved with the window column')""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("07_x_y_Scaling_Splitting", "07. Divide data to x and y, scaling and splitting", r"""
The last preparation steps of the class notebooks: **Divide data to x and y**, **Data scaling (x only)**, **Data Splitting** and **Data balancing** — and why time series need a different splitting rule.

* **Needs:** `work/dataset.csv`. **Makes:** `work/model_data.csv` (used by the LSTM notebooks).
""")
nb.code(SETUP)
nb.code("""from pandas import read_csv
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
dataset = dataset.loc[:'2023-08-31']        # main sample: January 2003 - August 2023
print(dataset.shape)""")
nb.md("""## Squared returns in logs
Like the target, the squared returns are very skewed, so the LSTM uses their logarithm. Two days had a return of exactly 0, and $\\ln(0)$ does not exist; `clip(lower=0.00000001)` replaces anything smaller than $10^{-8}$ by $10^{-8}$ first.""")
nb.code("""print('days with a zero return:', (dataset['r2'] == 0).sum())
from numpy import log
dataset['log_r2'] = log(dataset['r2'].clip(lower=0.00000001))
dataset['log_us_r2'] = log(dataset['us_r2'].clip(lower=0.00000001))
dataset['log_euro_r2'] = log(dataset['euro_r2'].clip(lower=0.00000001))""")
nb.md("""## Divide data to x and y
* **x** — six inputs per day, two for each market: the return (**direction**) and the log squared return (**size**).
* **y** — the target: the log of next week's realised variance.""")
nb.code("""x = dataset[['ret', 'log_r2', 'us_ret', 'log_us_r2', 'euro_ret', 'log_euro_r2']]
y = dataset['log_rv5_fwd']
print(x.shape)
print(y.shape)""")
nb.md("""## Data Splitting — why the class method does not work here
In class we wrote `train_test_split(x, y, test_size=0.10, random_state=10)`. It **shuffles** the rows. Look at the first test dates:""")
nb.code("""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)
print(x_test.index[:6])""")
nb.md("""The test days are mixed with the training days: the model would learn from 2009 and be tested on 2008 — it would **know the future**. For employees this does not matter; for time series it does. So we split **by time** with `shuffle=False`:""")
nb.code("""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=False)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print('training ends:', x_train.index[-1].date(), '| testing starts:', x_test.index[0].date())""")
nb.md("""### The walk-forward design used in the study
One split is not enough: the world changes. The study uses a **walk-forward**: train on all years up to 31 December, forecast the whole next year, then add that year to the training data and repeat — 17 times, for the test years 2007 to 2023.""")
nb.code("""for year in range(2007, 2024):
    train = dataset.loc[:str(year - 1) + '-12-31']
    test = dataset.loc[str(year) + '-01-01':str(year) + '-12-31']
    print('test year', year, '| training days:', len(train), '| test days:', len(test))""")
nb.md("""## Data scaling (x only) — fitted on the training years
In class: `x_scaled = StandardScaler().fit_transform(x)` on **all** of x before splitting. With time series this leaks the future: the mean and standard deviation would include the crisis years. Rule: **fit the scaler on the training years only**, then use it on the test year.""")
nb.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x.loc[:'2006-12-31'])         # learn mean and std from 2003-2006 only
from pandas import DataFrame
x_scaled = DataFrame(scaler.transform(x), index=x.index, columns=x.columns)
x_scaled.loc[:'2006'].describe().round(2)""")
nb.md("In the training years the mean is 0 and the standard deviation 1. In the crisis year 2008 the same scale gives much bigger numbers — exactly the information the model should see:")
nb.code("x_scaled.loc['2008'].describe().round(2)")
nb.md("""## Data balancing
In class, `SMOTE` made extra examples of the rare class (employees who left). Our target is a **number** (regression), not a class, so there is nothing to balance. Crisis weeks are rare, but inventing crisis days would break the time order. Instead, the results are reported **separately for each crisis window**.""")
nb.code("""dataset.to_csv(folder + 'work/model_data.csv')
print('saved')""")
nb.save()

# ------------------------------------------------------------------------------------------------
nb = NB("08_Measuring_Forecast_Errors", "08. Measuring forecast errors (QLIKE and MSE)", r"""
**Part 2 — The benchmark models** starts with the scoring rules. In class, a regressor was scored with `r2_score`. For volatility forecasts the study uses **QLIKE** (main score) and **MSE**. This notebook explains them with small numbers first.

* **Needs:** `work/dataset.csv`. **Makes:** nothing new.
""")
nb.code(SETUP)
nb.md("""## MSE — mean squared error
The average of (realised − forecast)². You know it from class.

## QLIKE
$\\text{QLIKE} = \\dfrac{RV}{F} - \\ln\\dfrac{RV}{F} - 1$, where RV = realised variance, F = forecast. It is **0** for a perfect forecast and bigger for worse ones.

### A small example
Three weeks; the third week is a storm (realised 4). Forecaster A says 2 (too low), forecaster B says 8 (too high).""")
nb.code("""from pandas import Series
from numpy import log
realised = Series([1, 1, 4])
forecast_A = Series([1, 1, 2])
forecast_B = Series([1, 1, 8])

MSE_A = ((realised - forecast_A) ** 2).mean()
MSE_B = ((realised - forecast_B) ** 2).mean()
print('MSE   A =', round(MSE_A, 3), '| MSE   B =', round(MSE_B, 3))

ratio_A = realised / forecast_A
ratio_B = realised / forecast_B
QLIKE_A = (ratio_A - log(ratio_A) - 1).mean()
QLIKE_B = (ratio_B - log(ratio_B) - 1).mean()
print('QLIKE A =', round(QLIKE_A, 3), '| QLIKE B =', round(QLIKE_B, 3))""")
nb.md("""MSE says A is better; QLIKE says **B** is better. QLIKE punishes **under-prediction** — saying 'calm' before a storm — more, which is what a risk manager fears most. QLIKE also ranks forecasts correctly even when the 'true' volatility is only measured with noise (Patton, 2011). That is why it is the **main score** of the study.""")
nb.md("""## A real example: the naive forecast
The simplest possible forecast: **next week will be like last week** (the last 5-day average × 5). It is not a real model, only a reference.""")
nb.code("""from pandas import read_csv
dataset = read_csv(folder + 'work/dataset.csv', index_col=0, parse_dates=True)
test = dataset.loc['2007-01-01':'2023-08-31'].dropna(subset=['rv5_fwd']).copy()
y_test = test['rv5_fwd']
y_pred_naive = test['rv_w'] * 5
print(len(y_test), 'test days')""")
nb.code("""ratio = y_test / y_pred_naive
QLIKE = (ratio - log(ratio) - 1).mean()
MSE = ((y_test - y_pred_naive) ** 2).mean()
print('QLIKE (naive) =', round(QLIKE, 3))
print('MSE (naive)   =', MSE)""")
nb.md("The MSE is a tiny number because variances are tiny (0.0004 = a normal week). That is why results tables show MSE × 1,000,000.")
nb.md("## The score in each window\nWe keep the daily QLIKE values in a column and average them per window with `groupby`:")
nb.code("""test['QLIKE_naive'] = ratio - log(ratio) - 1
test.groupby('window')['QLIKE_naive'].mean().round(3)""")
nb.md("The naive forecast is poor in every window. (In notebook 13 you will see that in the short Brexit window it still beats the real models: it simply copied the spike of the referendum day afterwards.) The statistical test that compares two models (Diebold–Mariano) comes in notebook 13.")
nb.save()

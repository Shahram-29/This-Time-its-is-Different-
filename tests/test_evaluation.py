"""
test_evaluation.py
Checks that the shared evaluation code (code/evaluation.py) and the processed dataset behave as the
Methodology chapter says. These tests add no analysis and change no result.

Run from the project folder:  py -3.13 -m pytest
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluation import dm_hln, mse, qlike, window_of  # noqa: E402

DATA = ROOT / "data" / "processed" / "dataset.csv"
RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not DATA.exists(),
                                reason="data/ is rebuilt by code/01_data_pipeline.py and is not in git")


def _reference_dm():
    """Independent DM-HLN implementation (John Tsang, 2017, MIT licence), kept unmodified in tests/reference."""
    spec = importlib.util.spec_from_file_location("dm_reference", ROOT / "tests" / "reference" / "dm_reference.py")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)   # its 2017 regex string '\d' warns on Python 3.12+
        spec.loader.exec_module(module)
    return module.dm_test


# ---- Loss functions -------------------------------------------------------------------------------

def test_qlike_is_zero_for_a_perfect_forecast():
    assert np.allclose(qlike([1e-4, 3e-4], [1e-4, 3e-4]), 0)


def test_qlike_known_values():
    assert qlike(2.0, 1.0) == pytest.approx(1 - np.log(2))      # RV/F = 2:   2 - ln 2 - 1
    assert qlike(1.0, 2.0) == pytest.approx(np.log(2) - 0.5)    # RV/F = 0.5: 0.5 - ln 0.5 - 1


def test_qlike_punishes_under_prediction_more():
    assert qlike(2.0, 1.0) > qlike(1.0, 2.0)


def test_qlike_depends_only_on_the_ratio():
    assert qlike(2e-4, 1e-4) == pytest.approx(qlike(2.0, 1.0))


def test_mse_known_values():
    assert np.allclose(mse([1.0, 2.0], [0.5, 4.0]), [0.25, 4.0])


# ---- Diebold-Mariano test with the HLN correction -------------------------------------------------

@pytest.mark.parametrize("h", [1, 5])
def test_dm_hln_matches_an_independent_implementation(h):
    rng = np.random.default_rng(2026)
    n = 300
    actual = np.round(rng.uniform(1, 3, n), 6)           # plain decimals: the reference rejects 1e-05 notation
    pred1 = np.round(actual + rng.normal(0, 0.30, n), 6)
    pred2 = np.round(actual + rng.normal(0.05, 0.35, n), 6)
    ours = dm_hln(mse(actual, pred1), mse(actual, pred2), h=h)
    ref = _reference_dm()(actual.tolist(), pred1.tolist(), pred2.tolist(), h=h, crit="MSE")
    assert ours["stat"] == pytest.approx(ref.DM, rel=1e-9)
    assert ours["p_value"] == pytest.approx(ref.p_value, rel=1e-9)


def test_dm_hln_sign_convention_and_symmetry():
    rng = np.random.default_rng(7)
    loss_a = rng.uniform(0.1, 1.0, 200)
    loss_b = loss_a + 0.2 + rng.normal(0, 0.1, 200)       # model A clearly more accurate
    a_vs_b, b_vs_a = dm_hln(loss_a, loss_b), dm_hln(loss_b, loss_a)
    assert a_vs_b["mean_diff"] < 0 and a_vs_b["stat"] < 0    # negative = model A more accurate
    assert a_vs_b["p_value"] < 0.01
    assert b_vs_a["stat"] == pytest.approx(-a_vs_b["stat"])
    assert b_vs_a["p_value"] == pytest.approx(a_vs_b["p_value"])


def test_dm_hln_drops_missing_days():
    rng = np.random.default_rng(1)
    loss_a = rng.uniform(0.1, 1.0, 50)
    loss_b = rng.uniform(0.1, 1.0, 50)
    loss_a[[3, 17]] = np.nan
    assert dm_hln(loss_a, loss_b, h=1)["n"] == 48


# ---- Evaluation windows ---------------------------------------------------------------------------

def test_window_boundaries_are_inclusive_and_other_days_are_calm():
    dates = pd.DatetimeIndex(["2007-08-08", "2007-08-09", "2009-03-09", "2009-03-10", "2015-06-01"])
    assert window_of(dates).tolist() == ["calm", "Global Financial Crisis", "Global Financial Crisis",
                                         "calm", "calm"]


# ---- No look-ahead in the processed dataset (recomputed independently from the raw cache) ---------

def _raw(name):
    df = pd.read_csv(RAW / f"{name}.csv", index_col=0, parse_dates=True).dropna(subset=["Close"])
    return df[df["Close"] > 0].sort_index()


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA, index_col=0, parse_dates=True)


@needs_data
def test_target_is_realised_variance_of_the_next_five_days_only(dataset):
    iseq = _raw("ISEQ")
    r2 = (np.log(iseq["Close"]).diff() ** 2).to_numpy()
    fwd = np.full(len(r2), np.nan)
    for i in range(len(r2) - 5):
        fwd[i] = r2[i + 1:i + 6].sum()                    # days t+1..t+5, never day t
    fwd = pd.Series(fwd, index=iseq.index).reindex(dataset.index)
    ok = dataset["rv5_fwd"].notna()
    assert ok.sum() > 5000
    assert np.allclose(dataset.loc[ok, "rv5_fwd"], fwd[ok], rtol=1e-10, atol=0)


@needs_data
def test_har_regressors_use_only_day_t_and_earlier(dataset):
    iseq = _raw("ISEQ")
    r2 = (np.log(iseq["Close"]).diff() ** 2).to_numpy()
    week, month = np.full(len(r2), np.nan), np.full(len(r2), np.nan)
    for i in range(len(r2)):
        if i >= 5:
            week[i] = r2[i - 4:i + 1].mean()             # days t-4..t
        if i >= 22:
            month[i] = r2[i - 21:i + 1].mean()           # days t-21..t
    expected = pd.DataFrame({"rv_w": week, "rv_m": month}, index=iseq.index).reindex(dataset.index)
    for col in ["rv_w", "rv_m"]:
        assert np.allclose(dataset[col], expected[col], rtol=1e-10, atol=0, equal_nan=True)


@needs_data
@pytest.mark.parametrize("raw_name, column", [("GSPC", "us_ret"), ("GDAXI", "euro_ret"), ("FTSE", "uk_ret")])
def test_foreign_returns_are_dated_on_or_before_the_irish_day(dataset, raw_name, column):
    foreign = _raw(raw_name)
    returns = np.log(foreign["Close"]).diff().to_numpy()
    last_known = foreign.index.searchsorted(dataset.index, side="right") - 1   # last foreign day <= t
    assert (last_known >= 0).all()
    assert np.allclose(dataset[column].to_numpy(), returns[last_known], rtol=1e-10, atol=0, equal_nan=True)

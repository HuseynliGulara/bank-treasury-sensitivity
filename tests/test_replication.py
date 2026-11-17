"""Check that the code reproduces the numbers printed in the paper.

Run with:  pytest -q
Values are copied from the SSRN version (November 2026) at the precision printed there.
"""
import pytest

from src import config as cfg
from src import data, tables


@pytest.fixture(scope="module")
def sample():
    return data.load_sample()


@pytest.fixture(scope="module")
def t3(sample):
    return tables.table3_betas(sample)


def test_sample_size(sample):
    assert len(sample) == 2961
    assert str(sample.index[0].date()) == "2015-01-05"
    assert str(sample.index[-1].date()) == "2026-11-12"


def test_table3_model1_betas(t3):
    paper = {"JPM": 0.0635, "BAC": 0.0780, "WFC": 0.0668, "C": 0.0569,
             "GS": 0.0348, "MS": 0.0519, "KRE": 0.0642, "SP500TR": 0.0053}
    for a, v in paper.items():
        assert round(t3[0].loc[a, "M1_beta"], 4) == v, a


def test_table3_model2_betas(t3):
    paper = {"JPM": 0.0688, "BAC": 0.0841, "WFC": 0.0724, "C": 0.0634,
             "GS": 0.0408, "MS": 0.0584, "KRE": 0.0703}
    for a, v in paper.items():
        assert round(t3[0].loc[a, "M2_beta"], 4) == v, a


def test_table4_wald(t3):
    _, m1, m2 = t3
    w = tables.table4_wald(m1, m2)
    assert round(w.loc["All banks equal (6 d.f.)", ("Model 1", "Wald")], 2) == 46.62
    assert round(w.loc["Commercial - investment", ("Model 1", "Wald")], 2) == 30.48
    assert round(w.loc["BAC - market", ("Model 2", "Wald")], 2) == 111.47
    assert round(w.loc["GS - market", ("Model 1", "Wald")], 2) == 10.69


def test_table5_break(sample):
    t = tables.table5_break(sample)
    assert round(t.loc["BAC", "beta_before"], 4) == 0.1519
    assert round(t.loc["BAC", "beta_after"], 4) == 0.0164
    assert round(t.loc["WFC", "decline_pct"], 1) == 67.7
    assert round(t.loc["GS", "decline_pct"], 1) == 97.2
    assert int(t.loc["JPM", "N_before"]) == 1799
    assert int(t.loc["JPM", "N_after"]) == 1162


def test_table7_market_vs_bank(sample):
    a, b, joint, _ = tables.table7_market_vs_bank(sample)
    assert round(a["beta_before"], 4) == 0.0327
    assert round(a["beta_after"], 4) == -0.0142
    assert round(a["chow_F"], 2) == 34.08
    assert round(joint["Wald"], 2) == 138.86
    assert round(b.loc["BAC", "change_in_gap"], 4) == -0.1401


def test_table11_groupings(t3):
    _, m1, m2 = t3
    t = tables.table11_groupings(m1, m2)
    assert round(t.loc[("Model 1", "BAC, WFC vs GS, MS"), "difference"], 4) == 0.0291
    assert round(t.loc[("Model 1", "JPM, BAC, WFC vs GS, MS (no Citigroup)"), "difference"], 4) == 0.0261

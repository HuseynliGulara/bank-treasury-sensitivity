"""One function per table in the paper. Each returns a DataFrame.

Table numbers follow the SSRN version of the paper (November 2026).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from . import config as cfg
from .econometrics import (break_test, fit_all, fmt, joint_hac_cov, ols_hac, regressors,
                           rolling_betas, stars, wald, wald_table, add_post, group_mean)

M = cfg.MARKET
R = cfg.RATE


def exclude_covid(sample):
    return sample.drop(sample.loc[cfg.COVID_START:cfg.COVID_END].index)


# --- Section 3 ---------------------------------------------------------------
def table1_summary(sample):
    t = sample.describe().T[["mean", "std", "min", "max"]]
    t["N"] = len(sample)
    return t


def table2_adf(sample):
    rows = [(c, *adfuller(sample[c], autolag="AIC", result_object=False)[:2])
            for c in sample.columns]
    return pd.DataFrame(rows, columns=["variable", "ADF stat", "p-value"]).set_index("variable")


# --- Section 5.1-5.2 ---------------------------------------------------------
def table3_betas(sample):
    m1, m2 = fit_all(sample, with_market=True), fit_all(sample, with_market=False)
    rows = []
    for a in cfg.ASSETS:
        f1, f2 = m1[a], m2[a]
        rows.append({
            "asset": a,
            "M1_beta": f1.params[R], "M1_se": f1.bse[R], "M1_p": f1.pvalues[R],
            "M1_sig": stars(f1.pvalues[R]),
            "M1_delta": f1.params.get(f"ret_{M}", np.nan), "M1_R2": f1.rsquared,
            "M2_beta": f2.params[R], "M2_se": f2.bse[R], "M2_p": f2.pvalues[R],
            "M2_sig": stars(f2.pvalues[R]), "M2_R2": f2.rsquared,
            "N": int(f1.nobs),
        })
    return pd.DataFrame(rows).set_index("asset"), m1, m2


def table4_wald(m1, m2):
    w1, w2 = wald_table(m1), wald_table(m2)
    return pd.concat({"Model 1": w1, "Model 2": w2}, axis=1)


# --- Section 5.3 -------------------------------------------------------------
def table5_break(sample):
    rows = {b: break_test(sample, f"ret_{b}", cfg.X_MODEL1) for b in cfg.BANKS}
    return pd.DataFrame(rows).T


def table6_subperiods(sample):
    out = {}
    for label, (s, e) in cfg.SUBPERIODS.items():
        d = sample.loc[s:e]
        out[label] = {b: fmt(ols_hac(d, f"ret_{b}", cfg.X_MODEL1), R) for b in cfg.BANKS}
        out[label]["N"] = len(d)
    return pd.DataFrame(out)


def table7_market_vs_bank(sample):
    """Panel A: break for the S&P 500. Panel B: Model 2 gaps. Panel C: by subperiod."""
    X = cfg.X_MODEL2

    # Panel A
    a = break_test(sample, f"ret_{M}", X)
    panel_a = pd.Series(a, name=f"{M} (Model 2)")

    # Panel B: stacked system with post-break interactions, Model 2 for every asset
    z, inter = add_post(sample, X)
    fits = {x: ols_hac(z, f"ret_{x}", X + inter) for x in cfg.ASSETS}
    p, V, nm = joint_hac_cov(fits)
    BX = f"{R}_x_post"
    mkt_change = a["change"]
    rows = []
    for b in cfg.BANKS:
        tests = {
            "gap_before": [{f"{b}:{R}": 1, f"{M}:{R}": -1}],
            "gap_after": [{f"{b}:{R}": 1, f"{b}:{BX}": 1, f"{M}:{R}": -1, f"{M}:{BX}": -1}],
            "change_in_gap": [{f"{b}:{BX}": 1, f"{M}:{BX}": -1}],
        }
        r = {"bank": b}
        for name, res in tests.items():
            d, W, q, pv = wald(p, V, nm, res)
            r[name] = d[0]
            r[f"p_{name}"] = pv
        r["sp500_share_pct"] = mkt_change / (mkt_change + r["change_in_gap"]) * 100
        rows.append(r)
    panel_b = pd.DataFrame(rows).set_index("bank")
    _, W, q, pv = wald(p, V, nm, [{f"{b}:{BX}": 1, f"{M}:{BX}": -1} for b in cfg.BANKS])
    joint = pd.Series({"Wald": W, "df": q, "p": pv}, name="joint change in gap = 0")

    # Panel C
    mkt_row, gaps = {}, {}
    for label, (s, e) in cfg.SUBPERIODS.items():
        d = sample.loc[s:e]
        fl = {x: ols_hac(d, f"ret_{x}", X) for x in cfg.ASSETS}
        mkt_row[label] = fmt(fl[M], R)
        ps, Vs, nms = joint_hac_cov(fl)
        gaps[label] = {}
        for b in cfg.BANKS:
            dd, _, _, pv = wald(ps, Vs, nms, [{f"{b}:{R}": 1, f"{M}:{R}": -1}])
            gaps[label][f"{b} - market"] = f"{dd[0]:.3f}{stars(pv)}"
    panel_c = pd.concat([pd.DataFrame({"S&P 500 TR (yield beta)": mkt_row}).T,
                         pd.DataFrame(gaps)])
    return panel_a, panel_b, joint, panel_c


# --- Section 6 ---------------------------------------------------------------
def table8_robustness(sample):
    ex = exclude_covid(sample)
    out = {}
    for rate, lab in [("d_DGS10", "10y"), ("d_DGS2", "2y"), ("d_DGS3MO", "3m")]:
        for name, data in [("full", sample), ("ex-COVID", ex)]:
            out[f"{lab}, {name}"] = {a: fmt(ols_hac(data, f"ret_{a}", regressors(a, rate)), rate)
                                     for a in cfg.ASSETS}
    return pd.DataFrame(out)


def table9_break_ex_covid(sample):
    ex = exclude_covid(sample)
    rows = {}
    for b in cfg.BANKS:
        z, inter = add_post(ex, cfg.X_MODEL1)
        f = ols_hac(z, f"ret_{b}", cfg.X_MODEL1 + inter)
        pre, ch = f.params[R], f.params[f"{R}_x_post"]
        rows[b] = {"beta_before": pre, "beta_after": pre + ch,
                   "decline_pct": (1 - (pre + ch) / pre) * 100,
                   "p_change": f.pvalues[f"{R}_x_post"]}
    return pd.DataFrame(rows).T


def table10_level_slope(sample):
    s = sample.copy()
    s["d_SLOPE"] = s["d_DGS10"] - s["d_DGS2"]
    rows = {}
    for a in cfg.ASSETS:
        X = ["d_DGS2", "d_SLOPE", "d_VIX"] + ([f"ret_{M}"] if a != M else [])
        f = ols_hac(s, f"ret_{a}", X)
        w = f.wald_test("d_DGS2 = d_SLOPE", scalar=True)
        rows[a] = {"level": fmt(f, "d_DGS2"), "slope": fmt(f, "d_SLOPE"),
                   "R2": f.rsquared,
                   "level_minus_slope": f.params["d_DGS2"] - f.params["d_SLOPE"],
                   "Wald_F_equal": float(w.statistic), "p_equal": float(w.pvalue)}
    return pd.DataFrame(rows).T


def table11_groupings(m1, m2):
    splits = [
        (["JPM", "BAC", "WFC", "C"], ["GS", "MS"], "JPM, BAC, WFC, C vs GS, MS (baseline)"),
        (["BAC", "WFC"], ["GS", "MS"], "BAC, WFC vs GS, MS"),
        (["JPM", "BAC", "WFC"], ["GS", "MS"], "JPM, BAC, WFC vs GS, MS (no Citigroup)"),
    ]
    rows = []
    for model, fits in [("Model 1", m1), ("Model 2", m2)]:
        p, V, nm = joint_hac_cov(fits)
        for g1, g2, label in splits:
            d, W, q, pv = wald(p, V, nm, [{**group_mean(g1, R), **group_mean(g2, R, -1)}])
            rows.append({"model": model, "grouping": label, "difference": d[0],
                         "Wald": W, "p": pv, "sig": stars(pv)})
    return pd.DataFrame(rows).set_index(["model", "grouping"])


# --- Robustness reported in the text -----------------------------------------
def hac_lag_sensitivity(sample):
    T = len(sample)
    nw_rule = int(np.floor(4 * (T / 100) ** (2 / 9)))
    rows = []
    for L in [cfg.HAC_LAGS, nw_rule, 10]:
        f1 = fit_all(sample, with_market=True, lags=L)
        w2 = wald_table(fit_all(sample, with_market=False, lags=L), lags=L)
        for a in cfg.ASSETS:
            rows.append({"lags": L, "row": a, "se": f1[a].bse[R], "p": f1[a].pvalues[R]})
        rows.append({"lags": L, "row": "Wald M2: commercial - investment", "se": np.nan,
                     "p": w2.loc["Commercial - investment", "p"]})
        rows.append({"lags": L, "row": "Wald M2: all banks equal", "se": np.nan,
                     "p": w2.loc["All banks equal (6 d.f.)", "p"]})
    return pd.DataFrame(rows).pivot(index="row", columns="lags", values=["se", "p"])


def wald_2y(sample):
    return wald_table(fit_all(sample, rate="d_DGS2", with_market=False), rate="d_DGS2")


def rolling_peaks(sample):
    full, ex = rolling_betas(sample), rolling_betas(exclude_covid(sample))
    return pd.DataFrame({
        "peak_full": full.max(), "date_full": full.idxmax().dt.date,
        "peak_ex_covid": ex.max(), "date_ex_covid": ex.idxmax().dt.date})

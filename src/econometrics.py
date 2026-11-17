"""Estimators and tests used throughout the paper."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.rolling import RollingOLS

from . import config as cfg


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(fit, var):
    """Coefficient with significance stars, e.g. '0.063***'."""
    return f"{fit.params[var]:.3f}{stars(fit.pvalues[var])}"


def ols_hac(data, y, X, lags=cfg.HAC_LAGS):
    """OLS with Newey-West (HAC) standard errors."""
    return sm.OLS(data[y], sm.add_constant(data[X])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def regressors(asset, rate=cfg.RATE, with_market=True):
    """Model 1 adds the market return for banks; the market never gets it."""
    X = [rate, "d_VIX"]
    if with_market and asset != cfg.MARKET:
        X.append(f"ret_{cfg.MARKET}")
    return X


def fit_all(data, rate=cfg.RATE, with_market=True, lags=cfg.HAC_LAGS):
    """One regression per asset (seven banks and the market)."""
    return {a: ols_hac(data, f"ret_{a}", regressors(a, rate, with_market), lags)
            for a in cfg.ASSETS}


# --- Joint covariance and Wald tests -----------------------------------------
def joint_hac_cov(fits, lags=cfg.HAC_LAGS):
    """Cross-equation HAC covariance from stacked moment conditions (Bartlett kernel).

    Each equation is estimated by OLS. With identical regressors across
    equations the point estimates equal SUR.
    """
    assets = list(fits)
    Xs = [fits[a].model.exog for a in assets]
    es = [fits[a].resid.values for a in assets]
    G = np.hstack([X * e[:, None] for X, e in zip(Xs, es)])
    T = len(G)
    S = G.T @ G / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        Gam = G[L:].T @ G[:-L] / T
        S += w * (Gam + Gam.T)
    k = [X.shape[1] for X in Xs]
    B = np.zeros((sum(k), sum(k)))
    o = 0
    for X, ki in zip(Xs, k):
        B[o:o + ki, o:o + ki] = np.linalg.inv(X.T @ X / T)
        o += ki
    V = B @ S @ B / T
    params = np.concatenate([fits[a].params.values for a in assets])
    names = [f"{a}:{p}" for a in assets for p in fits[a].params.index]
    return params, V, names


def wald(params, V, names, restrictions):
    """Test R @ params = 0.

    restrictions: list of dicts {"ASSET:variable": coefficient}, one dict per row.
    Returns (R @ params, Wald statistic, degrees of freedom, p-value).
    """
    R = np.zeros((len(restrictions), len(params)))
    for i, res in enumerate(restrictions):
        for n, c in res.items():
            R[i, names.index(n)] = c
    d = R @ params
    W = float(d @ np.linalg.solve(R @ V @ R.T, d))
    q = len(restrictions)
    return d, W, q, float(1 - stats.chi2.cdf(W, q))


def group_mean(assets, rate, sign=1.0):
    return {f"{a}:{rate}": sign / len(assets) for a in assets}


def wald_table(fits, rate=cfg.RATE, lags=cfg.HAC_LAGS):
    """H2 and H3 tests: each bank vs market, all banks equal, groups."""
    p, V, nm = joint_hac_cov(fits, lags)
    b = lambda a: f"{a}:{rate}"
    tests = {f"{a} - market": [{b(a): 1, b(cfg.MARKET): -1}] for a in cfg.BANKS}
    tests["All banks equal (6 d.f.)"] = [{b(cfg.BANKS[0]): 1, b(a): -1} for a in cfg.BANKS[1:]]
    tests["Commercial - investment"] = [{**group_mean(cfg.COMMERCIAL, rate),
                                         **group_mean(cfg.INVESTMENT, rate, -1)}]
    tests["KRE - six large banks"] = [{b("KRE"): 1, **group_mean(cfg.BIG_SIX, rate, -1)}]

    rows = []
    for name, res in tests.items():
        d, W, q, pv = wald(p, V, nm, res)
        rows.append({"test": name, "difference": d[0] if q == 1 else np.nan,
                     "Wald": W, "df": q, "p": pv, "sig": stars(pv)})
    return pd.DataFrame(rows).set_index("test")


# --- Structural break --------------------------------------------------------
def add_post(data, cols, break_date=cfg.BREAK_DATE):
    """Post-break dummy and its interaction with each regressor."""
    z = data.copy()
    z["post"] = (z.index >= break_date).astype(float)
    for c in cols:
        z[f"{c}_x_post"] = z[c] * z["post"]
    return z, ["post"] + [f"{c}_x_post" for c in cols]


def chow_f(data, y, X, break_date=cfg.BREAK_DATE):
    """Classic Chow F-test (assumes homoskedastic errors)."""
    pre, post = data[data.index < break_date], data[data.index >= break_date]
    rss = lambda d: sm.OLS(d[y], sm.add_constant(d[X])).fit().ssr
    k, n = len(X) + 1, len(data)
    F = ((rss(data) - rss(pre) - rss(post)) / k) / ((rss(pre) + rss(post)) / (n - 2 * k))
    return F, float(1 - stats.f.cdf(F, k, n - 2 * k)), len(pre), len(post)


def break_test(data, y, X, break_date=cfg.BREAK_DATE):
    """Chow test plus HAC-robust tests on all interactions and on the rate beta alone."""
    F, pF, n_pre, n_post = chow_f(data, y, X, break_date)
    z, inter = add_post(data, X, break_date)
    f = ols_hac(z, y, X + inter)
    w_all = f.wald_test(", ".join(f"{c} = 0" for c in inter), scalar=True)
    w_beta = f.wald_test(f"{cfg.RATE}_x_post = 0", scalar=True)
    b_pre = f.params[cfg.RATE]
    change = f.params[f"{cfg.RATE}_x_post"]
    return {
        "beta_before": b_pre, "beta_after": b_pre + change, "change": change,
        "decline_pct": (1 - (b_pre + change) / b_pre) * 100,
        "chow_F": F, "chow_p": pF,
        "robust_F_all": float(w_all.statistic), "robust_p_all": float(w_all.pvalue),
        "robust_F_beta": float(w_beta.statistic), "robust_p_beta": float(w_beta.pvalue),
        "N_before": n_pre, "N_after": n_post,
    }


# --- Rolling betas -----------------------------------------------------------
def rolling_beta(data, y, X, window=cfg.ROLL_WINDOW):
    return (RollingOLS(data[y], sm.add_constant(data[X]), window=window)
            .fit().params[cfg.RATE].dropna())


def rolling_betas(data, X=cfg.X_MODEL1):
    return pd.DataFrame({b: rolling_beta(data, f"ret_{b}", X) for b in cfg.BANKS}).dropna()

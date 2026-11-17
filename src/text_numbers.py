"""Numbers quoted in the text of the paper that do not appear in a table."""
import pandas as pd

from . import config as cfg
from .econometrics import ols_hac, rolling_beta, rolling_betas
from .tables import exclude_covid


def text_numbers(sample, build_info, dividend_gap):
    out = {}
    out["Sample: N"] = build_info["N"]
    out["Sample: first / last observation"] = f"{build_info['first']} / {build_info['last']}"
    out["Sample: days dropped (Treasury market closed)"] = build_info["dropped_days"]
    out["Footnote 4: SP500TR minus GSPC annualized return (pp)"] = round(dividend_gap * 100, 2)

    covid = sample.loc[cfg.COVID_START:cfg.COVID_END]
    out["COVID exclusion: trading days removed"] = len(covid)

    # Section 5.3: 2024 days with negative Model 1 rolling beta
    roll = rolling_betas(sample).loc["2024-01-01":"2024-12-31"]
    for b in cfg.BANKS:
        neg = roll[b] < 0
        if neg.any():
            out[f"2024 negative rolling beta: {b}"] = (
                f"{int(neg.sum())} days, {neg.idxmax().date()} to {neg[::-1].idxmax().date()}, "
                f"min {roll[b].min():.3f}")
        else:
            out[f"2024 negative rolling beta: {b}"] = f"never (min {roll[b].min():.3f})"

    # Section 6: daily std of the 3-month yield change, in basis points
    d3 = sample["d_DGS3MO"] * 100
    for lab, (s, e) in {"2015-2016": ("2015", "2016"),
                        "Jun 2020 - Dec 2021": ("2020-06-01", "2021-12-31"),
                        "2022-2026": ("2022", "2026")}.items():
        out[f"Std of daily ΔDGS3MO, {lab} (bp)"] = round(d3.loc[s:e].std(), 2)

    # Table 7 notes and Figure A1 notes: S&P 500 with and without Mar-Apr 2020
    ex = exclude_covid(sample)
    y, X = f"ret_{cfg.MARKET}", cfg.X_MODEL2
    for lab, d in {"full": sample, "ex-COVID": ex}.items():
        f = ols_hac(d.loc["2020-01-01":"2022-03-15"], y, X)
        out[f"S&P 500 beta 2020 - Mar 2022, {lab}"] = (
            f"{f.params[cfg.RATE]:.3f} (p = {f.pvalues[cfg.RATE]:.3f})")
        r = rolling_beta(d, y, X)
        peak = r.loc["2020-01-01":"2021-12-31"]
        out[f"S&P 500 rolling beta 2020-21 peak, {lab}"] = (
            f"{peak.max():.3f} on {peak.idxmax().date()}")
        out[f"S&P 500 rolling beta 2019 mean, {lab}"] = round(r.loc["2019"].mean(), 3)

    return pd.Series(out, name="value")

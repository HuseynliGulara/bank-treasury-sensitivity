"""Reproduce every table and figure in the paper.

    python run_all.py              # use the raw files in data/raw
    python run_all.py --download   # download fresh data first (see README)
"""
import argparse
import time

import pandas as pd

from src import config as cfg
from src import data, figures, tables
from src.text_numbers import text_numbers


def save(df, name):
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.TABLES_DIR / f"{name}.csv", float_format="%.10g")
    print(f"  results/tables/{name}.csv")


def main(download=False):
    t0 = time.time()
    if download:
        print("Downloading data")
        data.download_yahoo()
        data.download_fred()

    if data.raw_prices_available():
        print("Building the sample from data/raw")
        sample, info = data.build_sample()
        dividend_gap = data.dividend_check()
    else:
        print("Raw Yahoo prices not found; using data/processed/sample.csv "
              "(run with --download to rebuild from raw data)")
        sample = data.load_sample()
        info = {"N": len(sample), "first": sample.index[0].date(),
                "last": sample.index[-1].date(), "dropped_days": "n/a"}
        dividend_gap = float("nan")
    print(f"  N = {info['N']}, {info['first']} to {info['last']}")

    print("Tables")
    save(tables.table1_summary(sample), "table01_summary_statistics")
    save(tables.table2_adf(sample), "table02_adf_tests")
    t3, m1, m2 = tables.table3_betas(sample)
    save(t3, "table03_betas_model1_model2")
    save(tables.table4_wald(m1, m2), "table04_wald_tests")
    save(tables.table5_break(sample), "table05_structural_break")
    save(tables.table6_subperiods(sample), "table06_subperiods")
    a, b, joint, c = tables.table7_market_vs_bank(sample)
    save(a.to_frame(), "table07a_sp500_break")
    save(b, "table07b_gap_before_after")
    save(joint.to_frame(), "table07b_joint_test")
    save(c, "table07c_gap_by_subperiod")
    save(tables.table8_robustness(sample), "table08_alternative_maturities")
    save(tables.table9_break_ex_covid(sample), "table09_break_ex_covid")
    save(tables.table10_level_slope(sample), "table10_level_slope")
    save(tables.table11_groupings(m1, m2), "table11_bank_groupings")

    print("Robustness reported in the text")
    save(tables.hac_lag_sensitivity(sample), "text_hac_lags")
    save(tables.wald_2y(sample), "text_wald_2y_model2")
    save(tables.rolling_peaks(sample), "text_rolling_peaks_ex_covid")
    save(text_numbers(sample, info, dividend_gap).to_frame(), "text_numbers")

    print("Figures")
    figures.figure1(sample)
    figures.figure2(sample)
    figures.figureA1(sample)
    print("  results/figures/ (PNG and PDF)")

    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true",
                   help="download fresh data from Yahoo Finance and FRED first")
    main(p.parse_args().download)

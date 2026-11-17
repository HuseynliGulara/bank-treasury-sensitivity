"""Settings shared by every step of the replication.

All dates and choices below are the ones used in the paper. Change them here,
not inside the analysis code.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

# --- Sample ----------------------------------------------------------------
START = "2015-01-01"
END_EXCLUSIVE = "2026-11-13"        # last observation: 12 November 2026

# --- Assets ----------------------------------------------------------------
BANKS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "KRE"]
MARKET = "SP500TR"                  # S&P 500 Total Return index (^SP500TR)
PRICE_INDEX = "GSPC"                # S&P 500 price index, only for the dividend check
ASSETS = BANKS + [MARKET]

COMMERCIAL = ["JPM", "BAC", "WFC", "C"]
INVESTMENT = ["GS", "MS"]
BIG_SIX = COMMERCIAL + INVESTMENT

FRED_SERIES = ["DGS10", "DGS2", "DGS3MO", "VIXCLS"]

# --- Estimation --------------------------------------------------------------
HAC_LAGS = 5                        # Newey-West lags for every standard error
ROLL_WINDOW = 252                   # rolling window, about one trading year

BREAK_DATE = "2022-03-16"           # FOMC: first rate increase since 2018
SVB_DATE = "2023-03-10"             # Silicon Valley Bank failure (marked on figures only)
COVID_START, COVID_END = "2020-03-01", "2020-04-30"   # excluded in robustness checks

SUBPERIODS = {
    "2015-2019":           ("2015-01-01", "2019-12-31"),
    "2020 - Mar 2022":     ("2020-01-01", "2022-03-15"),
    "Mar 2022 - Jul 2023": ("2022-03-16", "2023-07-31"),
    "Aug 2023 - Nov 2026": ("2023-08-01", "2026-11-12"),
}

# Regressors
RATE = "d_DGS10"
X_MODEL1 = [RATE, "d_VIX", f"ret_{MARKET}"]   # with market return
X_MODEL2 = [RATE, "d_VIX"]                    # without market return

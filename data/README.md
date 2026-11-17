# Data

## `raw/`

| File | Source | Description |
|---|---|---|
| `DGS10.csv`, `DGS2.csv`, `DGS3MO.csv` | FRED | Constant-maturity Treasury yields, percent |
| `VIXCLS.csv` | FRED | CBOE Volatility Index |
| `yahoo_*.csv` | Yahoo Finance | Daily prices adjusted for splits and dividends (not included, see below) |

The Yahoo Finance price files are not stored in this repository because of Yahoo's terms of
use. `python run_all.py --download` downloads them with `yfinance` into this folder.

## `processed/sample.csv`

The estimation sample used in the paper: 2,961 trading days, 5 January 2015 to
12 November 2026.

| Column | Definition |
|---|---|
| `ret_<ticker>` | Daily log return, ln P<sub>t</sub> − ln P<sub>t−1</sub> |
| `d_DGS10`, `d_DGS2`, `d_DGS3MO` | Daily change in the yield, percentage points (0.01 = 1 bp) |
| `d_VIX` | Daily change in the VIX |

Only days on which every series is observed are kept. This drops 22 days on which the stock
market was open but the Treasury market was closed (mostly Columbus Day and Veterans Day).

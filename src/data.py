"""Download the raw data and build the estimation sample."""
import numpy as np
import pandas as pd

from . import config as cfg


# --- Download ----------------------------------------------------------------
def download_yahoo():
    """Adjusted daily closes from Yahoo Finance (splits and dividends)."""
    import yfinance as yf

    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = cfg.BANKS + ["^" + cfg.MARKET, "^" + cfg.PRICE_INDEX]
    for t in tickers:
        df = yf.download(t, start=cfg.START, end=cfg.END_EXCLUSIVE,
                         auto_adjust=True, progress=False)
        name = t.replace("^", "")
        df.to_csv(cfg.RAW_DIR / f"yahoo_{name}.csv")
        print(f"  {t:10s} {len(df):5d} rows  {df.index[0].date()} -> {df.index[-1].date()}")


def download_fred():
    """Treasury yields and the VIX from FRED (no API key needed)."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp(cfg.END_EXCLUSIVE) - pd.Timedelta(days=1)).date()
    for s in cfg.FRED_SERIES:
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={s}&cosd={cfg.START}&coed={end}")
        df = pd.read_csv(url)
        df.to_csv(cfg.RAW_DIR / f"{s}.csv", index=False)
        print(f"  {s:10s} {len(df):5d} rows")


# --- Load --------------------------------------------------------------------
def raw_prices_available():
    names = cfg.ASSETS + [cfg.PRICE_INDEX]
    return all((cfg.RAW_DIR / f"yahoo_{n}.csv").exists() for n in names)


def load_yahoo(name):
    df = pd.read_csv(cfg.RAW_DIR / f"yahoo_{name}.csv",
                     header=[0, 1], index_col=0, parse_dates=True)
    return df["Close"].iloc[:, 0].rename(name)


def load_fred(series):
    return (pd.read_csv(cfg.RAW_DIR / f"{series}.csv", index_col=0, parse_dates=True)
            .iloc[:, 0].rename(series))


def load_prices():
    return pd.concat([load_yahoo(n) for n in cfg.ASSETS + [cfg.PRICE_INDEX]], axis=1)


def load_fred_all():
    return pd.concat([load_fred(s) for s in cfg.FRED_SERIES], axis=1)


# --- Build -------------------------------------------------------------------
def build_sample(save=True):
    """Daily log returns and first differences of yields and the VIX.

    Keeps only days on which every series is observed (drops days when the
    stock market was open but the Treasury market was closed), then differences.
    """
    prices = load_prices()
    fred = load_fred_all()

    merged = prices[cfg.ASSETS].join(fred, how="inner")
    n_joined = len(merged)
    merged = merged.dropna()

    sample = pd.DataFrame(index=merged.index)
    for c in cfg.ASSETS:
        sample[f"ret_{c}"] = np.log(merged[c]).diff()
    for c in ["DGS10", "DGS2", "DGS3MO"]:
        sample[f"d_{c}"] = merged[c].diff()
    sample["d_VIX"] = merged["VIXCLS"].diff()
    sample = sample.dropna()
    sample.index.name = "date"

    info = {
        "dropped_days": int(n_joined - len(merged)),
        "N": len(sample),
        "first": sample.index[0].date(),
        "last": sample.index[-1].date(),
    }
    if save:
        cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        sample.to_csv(cfg.PROCESSED_DIR / "sample.csv", float_format="%.12g")
    return sample, info


def load_sample():
    return pd.read_csv(cfg.PROCESSED_DIR / "sample.csv", index_col=0, parse_dates=True)


def dividend_check():
    """Annualized return gap between the total-return and price index (footnote 4)."""
    p = load_prices()[[cfg.MARKET, cfg.PRICE_INDEX]].dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    return float(cagr[cfg.MARKET] - cagr[cfg.PRICE_INDEX])

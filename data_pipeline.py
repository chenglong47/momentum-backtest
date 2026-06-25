"""
Week 1: Data Pipeline
=====================
Fetches S&P 500 constituent price history, cleans it, and computes
monthly returns + momentum signals ready for the backtest engine.

Dependencies:
    pip install yfinance pandas numpy requests beautifulsoup4
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE   = "2014-01-01"   # 10 years of history
END_DATE     = "2024-12-31"
DATA_DIR     = "data"         # where we cache raw + processed files
LOOKBACK     = 12             # momentum lookback in months (12-1 = 11 usable)
SKIP_RECENT  = 1              # skip most recent month (avoids short-term reversal)

os.makedirs(DATA_DIR, exist_ok=True)


# ── STEP 1: GET S&P 500 TICKERS FROM WIKIPEDIA ────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """
    Scrapes current S&P 500 constituents from Wikipedia.
    Returns a list of ticker strings (e.g. ['AAPL', 'MSFT', ...]).

    Note for interviews: in production you'd use a point-in-time
    constituent list to avoid survivorship bias. This is a known
    simplification — be ready to discuss it.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if cols:
            ticker = cols[0].text.strip().replace(".", "-")  # BRK.B → BRK-B
            tickers.append(ticker)
    print(f"Found {len(tickers)} tickers")
    return tickers


# ── STEP 2: DOWNLOAD PRICE DATA ────────────────────────────────────────────────
def download_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for all tickers.
    Uses yfinance's batch download (much faster than one-by-one).
    Caches to disk so you don't re-download on every run.

    Returns a DataFrame: rows = dates, columns = tickers.
    """
    cache_path = os.path.join(DATA_DIR, "prices_raw.parquet")
    if os.path.exists(cache_path):
        print("Loading prices from cache...")
        return pd.read_parquet(cache_path)

    print(f"Downloading prices for {len(tickers)} tickers ({START_DATE} → {END_DATE})...")
    # yfinance batch download — group=ticker gives MultiIndex columns
    raw = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts for splits + dividends automatically
        progress=True,
    )

    # Extract just the "Close" prices
    prices = raw["Close"]
    prices.to_parquet(cache_path)
    print(f"Downloaded and cached: {prices.shape}")
    return prices


# ── STEP 3: CLEAN THE DATA ────────────────────────────────────────────────────
def clean_prices(prices: pd.DataFrame, min_history_pct: float = 0.8) -> pd.DataFrame:
    """
    Removes tickers with too much missing data and forward-fills
    short gaps (e.g. trading halts, delistings).

    min_history_pct: drop any ticker missing more than this fraction of days.

    Interview note: data quality is a huge deal in quant finance.
    Real shops spend enormous effort on cleaning pipelines.
    """
    total_days = len(prices)
    min_days = int(total_days * min_history_pct)

    # Count non-NaN observations per ticker
    valid_counts = prices.notna().sum()
    good_tickers = valid_counts[valid_counts >= min_days].index
    dropped = len(prices.columns) - len(good_tickers)
    print(f"Dropped {dropped} tickers with insufficient history")

    prices = prices[good_tickers]

    # Forward fill up to 5 business days (handles short gaps)
    prices = prices.ffill(limit=5)

    # Drop any remaining rows that are all NaN
    prices = prices.dropna(how="all")

    print(f"Clean price matrix: {prices.shape[0]} days × {prices.shape[1]} stocks")
    return prices


# ── STEP 4: COMPUTE MONTHLY RETURNS ──────────────────────────────────────────
def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily prices to month-end and computes simple monthly returns.

    We use month-end ('ME') resampling — each row represents the return
    earned during that calendar month.

    Returns DataFrame: rows = month-end dates, columns = tickers.
    """
    # Take last price of each month
    monthly_prices = prices.resample("ME").last()

    # Pct change: (P_t - P_{t-1}) / P_{t-1}
    monthly_returns = monthly_prices.pct_change()

    # Drop first row (NaN — no prior month to compare)
    monthly_returns = monthly_returns.iloc[1:]

    print(f"Monthly returns: {monthly_returns.shape[0]} months × {monthly_returns.shape[1]} stocks")
    return monthly_returns


# ── STEP 5: COMPUTE MOMENTUM SIGNAL ──────────────────────────────────────────
def compute_momentum(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the classic 12-1 momentum signal for each stock each month.

    Formula: cumulative return from t-12 to t-2
    (skipping t-1 to avoid short-term reversal — a well-documented effect)

    The result at month t tells you: "over the past year (minus last month),
    how much did this stock return?" Stocks in the top decile are 'winners',
    bottom decile are 'losers'.

    Interview note: Jegadeesh & Titman (1993) is the foundational paper.
    D.E. Shaw interviewers will respect that you know the academic basis.
    """
    # rolling_product(1 + r) - 1 over a window gives cumulative return
    # Window = LOOKBACK months, but we shift by SKIP_RECENT to exclude last month
    def cum_return(r):
        return (1 + r).prod() - 1

    # For each month t, compute cumulative return from t-LOOKBACK to t-SKIP_RECENT-1
    momentum = monthly_returns.shift(SKIP_RECENT).rolling(
        window=LOOKBACK - SKIP_RECENT
    ).apply(cum_return, raw=True)

    print(f"Momentum signal computed: {momentum.notna().sum().sum():,} valid observations")
    return momentum


# ── STEP 6: COMPUTE CROSS-SECTIONAL RANKS ────────────────────────────────────
def compute_signal_ranks(momentum: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw momentum values to cross-sectional percentile ranks each month.

    Instead of using raw returns, we rank stocks against each other.
    This makes the signal robust to market-wide moves and outliers.

    Rank of 1.0 = top momentum stock that month
    Rank of 0.0 = bottom momentum stock that month
    """
    # axis=1 means rank across columns (stocks) for each row (month)
    ranks = momentum.rank(axis=1, pct=True)
    return ranks


# ── MAIN ──────────────────────────────────────────────────────────────────────
def build_pipeline() -> dict:
    """
    Runs the full pipeline and returns all intermediate outputs.
    This is what the backtest engine (Week 2) will import.
    """
    print("=" * 60)
    print("MOMENTUM BACKTEST — WEEK 1: DATA PIPELINE")
    print("=" * 60)

    tickers        = get_sp500_tickers()
    prices_raw     = download_prices(tickers)
    prices_clean   = clean_prices(prices_raw)
    monthly_ret    = compute_monthly_returns(prices_clean)
    momentum       = compute_momentum(monthly_ret)
    signal_ranks   = compute_signal_ranks(momentum)

    # Save processed data for Week 2
    monthly_ret.to_parquet(os.path.join(DATA_DIR, "monthly_returns.parquet"))
    momentum.to_parquet(os.path.join(DATA_DIR, "momentum_signal.parquet"))
    signal_ranks.to_parquet(os.path.join(DATA_DIR, "signal_ranks.parquet"))
    print(f"\nSaved processed data to ./{DATA_DIR}/")

    # Quick sanity check — print top 5 momentum stocks for most recent month
    latest_month = signal_ranks.index[-1]
    top5 = signal_ranks.loc[latest_month].nlargest(5)
    print(f"\nTop 5 momentum stocks as of {latest_month.strftime('%b %Y')}:")
    for ticker, rank in top5.items():
        print(f"  {ticker:8s}  rank percentile: {rank:.2f}")

    return {
        "prices":        prices_clean,
        "monthly_ret":   monthly_ret,
        "momentum":      momentum,
        "signal_ranks":  signal_ranks,
    }


if __name__ == "__main__":
    data = build_pipeline()
    print("\nPipeline complete. Ready for Week 2: Backtest Engine.")
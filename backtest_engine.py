"""

Constructs a long/short momentum portfolio from signal ranks,
applies transaction costs, and computes monthly P&L.
Run after data_pipeline.py has populated the data/ directory.
Dependencies:
    pip install yfinance pandas numpy requests beautifulsoup4 pyarrow
"""

import pandas as pd
import numpy as np
import os

# CONFIG
DATA_DIR          = "data"
TOP_DECILE        = 0.9      # long stocks ranked above 90th percentile
BOTTOM_DECILE     = 0.1      # short stocks ranked below 10th percentile
TRANSACTION_COST  = 0.001    # 10 bps per side (0.1%) — conservative estimate
BENCHMARK_TICKER  = "^GSPC"  # S&P 500 index


# STEP 1: LOAD PIPELINE OUTPUTS 
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads monthly returns and signal ranks produced by data_pipeline.py.
    """
    monthly_ret   = pd.read_parquet(os.path.join(DATA_DIR, "monthly_returns.parquet"))
    signal_ranks  = pd.read_parquet(os.path.join(DATA_DIR, "signal_ranks.parquet"))
    print(f"Loaded {len(monthly_ret)} months of returns, {monthly_ret.shape[1]} stocks")
    return monthly_ret, signal_ranks


# STEP 2: CONSTRUCT PORTFOLIO WEIGHTS
def compute_weights(signal_ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Each month, assigns portfolio weights based on momentum rank:
      - Top decile stocks: equal weight long  (+1 / n_long)
      - Bottom decile stocks: equal weight short (-1 / n_short)
      - Everyone else: 0

    Equal weighting within deciles is a deliberate choice — it avoids
    concentration risk and is standard in academic momentum studies.
    """
    weights = pd.DataFrame(0.0, index=signal_ranks.index, columns=signal_ranks.columns)

    for date, row in signal_ranks.iterrows():
        valid = row.dropna()
        if len(valid) < 20:
            continue  # skip months with too few stocks

        long_mask  = valid >= TOP_DECILE
        short_mask = valid <= BOTTOM_DECILE

        n_long  = long_mask.sum()
        n_short = short_mask.sum()

        if n_long > 0:
            weights.loc[date, valid[long_mask].index]  =  1.0 / n_long

    long_counts  = (weights > 0).sum(axis=1)
    short_counts = (weights < 0).sum(axis=1)
    print(f"Avg long positions per month:  {long_counts[long_counts > 0].mean():.1f}")
    print(f"Avg short positions per month: {short_counts[short_counts > 0].mean():.1f}")
    return weights


# ── STEP 3: COMPUTE TURNOVER AND TRANSACTION COSTS ────────────────────────────
def compute_transaction_costs(weights: pd.DataFrame) -> pd.Series:
    """
    Transaction cost = 10 bps × absolute change in each position each month.

    Turnover measures how much the portfolio changes each rebalance.
    High turnover erodes returns — this is why momentum strategies
    are typically run monthly, not weekly.
    """
    # Weight change = new weight - old weight (shift by 1 month)
    weight_changes = weights.diff().abs()

    # Cost per month = sum of |Δweight| × cost per unit traded
    costs = weight_changes.sum(axis=1) * TRANSACTION_COST

    avg_monthly_turnover = weights.diff().abs().sum(axis=1).mean()
    print(f"Avg monthly turnover:          {avg_monthly_turnover:.2%}")
    print(f"Avg monthly transaction cost:  {costs.mean():.4%}")
    return costs


# STEP 4: COMPUTE PORTFOLIO RETURNS
def compute_portfolio_returns(
    weights: pd.DataFrame,
    monthly_ret: pd.DataFrame,
    costs: pd.Series
) -> pd.Series:
    """
    Portfolio return for month t = sum(weight_{t-1} × return_t) - cost_t

    We use lagged weights (shift by 1) because weights are determined at
    end of month t-1 and earn returns during month t.

    This is a critical detail — using same-month weights introduces
    look-ahead bias, a common backtest error.
    """
    # Align weights and returns on shared columns and dates
    shared_cols  = weights.columns.intersection(monthly_ret.columns)
    w = weights[shared_cols].shift(1)   # lag weights by 1 month
    r = monthly_ret[shared_cols]

    # Gross return: dot product of weights and returns, row by row
    gross_returns = (w * r).sum(axis=1)

    # Net return: subtract transaction costs
    net_returns = gross_returns - costs

    # Drop first row (NaN from lag) and warmup period
    net_returns = net_returns.dropna()
    net_returns = net_returns[net_returns.index >= net_returns.index[12]]

    print(f"Backtest period: {net_returns.index[0].strftime('%b %Y')} → {net_returns.index[-1].strftime('%b %Y')}")
    print(f"Total months:    {len(net_returns)}")
    return net_returns


# STEP 5: FETCH BENCHMARK RETURNS 
def get_benchmark_returns(start: str, end: str) -> pd.Series:
    """
    Downloads S&P 500 monthly returns as the benchmark.
    We compare our strategy against this to compute alpha.
    """
    import yfinance as yf
    raw = yf.download(BENCHMARK_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    monthly = raw["Close"].resample("ME").last().pct_change().dropna()
    monthly.name = "SP500"
    return monthly


#STEP 6: PERFORMANCE METRICS 
def compute_metrics(returns: pd.Series, benchmark: pd.Series) -> dict:
    """
    Computes the core performance statistics used by quant funds
    to evaluate a strategy.

    Sharpe Ratio: risk-adjusted return. Above 1.0 is good, above 1.5 is strong.
    Max Drawdown: worst peak-to-trough loss. Tells you the worst case experience.
    Alpha:        return unexplained by market exposure (the "edge").
    Beta:         sensitivity to market moves. Near 0 is ideal for a L/S strategy.
    """
    # Align on shared dates
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    strat   = aligned.iloc[:, 0]
    bench   = aligned.iloc[:, 1]

    # Annualized return
    ann_return = (1 + strat).prod() ** (12 / len(strat)) - 1

    # Annualized volatility
    ann_vol = strat.std() * np.sqrt(12)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Maximum drawdown
    cumulative  = (1 + strat).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Alpha and Beta via OLS regression
    cov_matrix = np.cov(strat, bench)
    beta       = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha_monthly = strat.mean() - beta * bench.mean()
    alpha_annual  = alpha_monthly * 12

    # Win rate
    win_rate = (strat > 0).mean()

    # Benchmark annualized return for comparison
    bench_ann = (1 + bench).prod() ** (12 / len(bench)) - 1

    metrics = {
        "Annualized Return":     ann_return,
        "Benchmark Return":      bench_ann,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio":          sharpe,
        "Max Drawdown":          max_dd,
        "Alpha (annualized)":    alpha_annual,
        "Beta":                  beta,
        "Win Rate":              win_rate,
        "Total Months":          len(strat),
    }

    return metrics, strat, bench


# MAIN 
def run_backtest() -> dict:
    print("=" * 60)
    print("MOMENTUM BACKTEST —  BACKTEST ENGINE")
    print("=" * 60)

    monthly_ret, signal_ranks = load_data()
    weights    = compute_weights(signal_ranks)
    costs      = compute_transaction_costs(weights)
    net_ret    = compute_portfolio_returns(weights, monthly_ret, costs)
    benchmark  = get_benchmark_returns(
        str(net_ret.index[0].date()),
        str(net_ret.index[-1].date())
    )

    metrics, strat, bench = compute_metrics(net_ret, benchmark)

    print("\n" + "─" * 40)
    print("PERFORMANCE SUMMARY")
    print("─" * 40)
    for k, v in metrics.items():
        if k == "Total Months":
            print(f"  {k:<28} {v}")
        elif k in ("Beta",):
            print(f"  {k:<28} {v:.3f}")
        else:
            print(f"  {k:<28} {v:.2%}")

    # Save outputs for Week 3 analytics
    net_ret.to_csv(os.path.join(DATA_DIR, "strategy_returns.csv"))
    benchmark.to_csv(os.path.join(DATA_DIR, "benchmark_returns.csv"))
    weights.to_parquet(os.path.join(DATA_DIR, "portfolio_weights.parquet"))
    print(f"\nSaved backtest outputs to ./{DATA_DIR}/")

    return {"metrics": metrics, "strategy": strat, "benchmark": bench, "weights": weights}


if __name__ == "__main__":
    results = run_backtest()
    print("\nBacktest complete. Run analytics.py for charts.")

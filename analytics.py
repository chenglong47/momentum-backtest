"""
Loads backtest results and produces publication-quality charts:
  1. Cumulative returns vs S&P 500 benchmark
  2. Monthly return distribution
  3. Drawdown over time
  4. Rolling 12-month Sharpe ratio
Saves all charts to charts/ directory.

Dependencies:
    pip install matplotlib seaborn pandas numpy pyarrow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

DATA_DIR   = "data"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

#STYLE 
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "text.color":        "#c9d1d9",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "sans-serif",
})

BLUE   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
GRAY   = "#8b949e"


#  LOAD DATA 
def load_results() -> tuple[pd.Series, pd.Series]:
    strat = pd.read_csv(
        os.path.join(DATA_DIR, "strategy_returns.csv"),
        index_col=0, parse_dates=True
    ).squeeze()
    bench = pd.read_csv(
        os.path.join(DATA_DIR, "benchmark_returns.csv"),
        index_col=0, parse_dates=True
    ).squeeze()
    aligned = pd.concat([strat, bench], axis=1).dropna()
    aligned.columns = ["Strategy", "S&P 500"]
    return aligned["Strategy"], aligned["S&P 500"]


#HELPER: DRAWDOWN SERIES 
def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


#HELPER: ROLLING SHARPE 
def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    roll_mean = returns.rolling(window).mean() * 12
    roll_std  = returns.rolling(window).std() * np.sqrt(12)
    return roll_mean / roll_std


# CHART 1: CUMULATIVE RETURNS 
def plot_cumulative_returns(strat: pd.Series, bench: pd.Series):
    cum_strat = (1 + strat).cumprod()
    cum_bench = (1 + bench).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(cum_strat.index, cum_strat.values, color=BLUE,  lw=2,   label="Momentum Strategy")
    ax.plot(cum_bench.index, cum_bench.values, color=GRAY,  lw=1.5, label="S&P 500", linestyle="--")
    ax.fill_between(cum_strat.index, cum_strat.values, cum_bench.values,
                    where=cum_strat.values >= cum_bench.values,
                    alpha=0.15, color=GREEN, label="Outperformance")
    ax.fill_between(cum_strat.index, cum_strat.values, cum_bench.values,
                    where=cum_strat.values < cum_bench.values,
                    alpha=0.15, color=RED, label="Underperformance")

    # Annotate final values
    ax.annotate(f"{cum_strat.iloc[-1]:.2f}x",
                xy=(cum_strat.index[-1], cum_strat.iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=BLUE, fontsize=10, va="center")
    ax.annotate(f"{cum_bench.iloc[-1]:.2f}x",
                xy=(cum_bench.index[-1], cum_bench.iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=GRAY, fontsize=10, va="center")

    ax.set_title("Cumulative Returns: Momentum Strategy vs S&P 500")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1fx"))
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "1_cumulative_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


#  CHART 2: MONTHLY RETURN DISTRIBUTION 
def plot_return_distribution(strat: pd.Series, bench: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(
        min(strat.min(), bench.min()) - 0.01,
        max(strat.max(), bench.max()) + 0.01,
        50
    )

    ax.hist(bench.values,  bins=bins, alpha=0.5, color=GRAY,  label="S&P 500",           density=True)
    ax.hist(strat.values,  bins=bins, alpha=0.6, color=BLUE,  label="Momentum Strategy",  density=True)

    # Vertical lines for means
    ax.axvline(strat.mean(), color=BLUE, lw=1.5, linestyle="--",
               label=f"Strategy mean: {strat.mean():.2%}")
    ax.axvline(bench.mean(), color=GRAY, lw=1.5, linestyle="--",
               label=f"S&P 500 mean:  {bench.mean():.2%}")
    ax.axvline(0, color="white", lw=0.8, alpha=0.4)

    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("Monthly Return")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "2_return_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# CHART 3: DRAWDOWN 
def plot_drawdown(strat: pd.Series, bench: pd.Series):
    dd_strat = drawdown_series(strat)
    dd_bench = drawdown_series(bench)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(dd_strat.index, dd_strat.values, 0, color=BLUE, alpha=0.4, label="Momentum Strategy")
    ax.fill_between(dd_bench.index, dd_bench.values, 0, color=GRAY, alpha=0.3, label="S&P 500")
    ax.plot(dd_strat.index, dd_strat.values, color=BLUE, lw=1)
    ax.plot(dd_bench.index, dd_bench.values, color=GRAY, lw=1)

    # Annotate worst drawdown
    worst_idx = dd_strat.idxmin()
    ax.annotate(f"Max DD: {dd_strat.min():.1%}",
                xy=(worst_idx, dd_strat.min()),
                xytext=(0, -20), textcoords="offset points",
                color=RED, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    ax.set_title("Drawdown Over Time")
    ax.set_ylabel("Drawdown from Peak")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "3_drawdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# HART 4: ROLLING 12-MONTH SHARPE 
def plot_rolling_sharpe(strat: pd.Series):
    roll = rolling_sharpe(strat, window=12).dropna()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(roll.index, roll.values, color=YELLOW, lw=1.5)
    ax.fill_between(roll.index, roll.values, 0,
                    where=roll.values >= 0, color=GREEN, alpha=0.2)
    ax.fill_between(roll.index, roll.values, 0,
                    where=roll.values < 0,  color=RED,   alpha=0.2)
    ax.axhline(0,   color="white", lw=0.8, alpha=0.4)
    ax.axhline(1.0, color=GREEN,   lw=0.8, linestyle="--", alpha=0.6, label="Sharpe = 1.0")

    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "4_rolling_sharpe.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# CHART 5: ANNUAL RETURNS BAR CHART
def plot_annual_returns(strat: pd.Series, bench: pd.Series):
    ann_strat = strat.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    ann_bench = bench.resample("YE").apply(lambda r: (1 + r).prod() - 1)

    years = ann_strat.index.year
    x = np.arange(len(years))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_s = ax.bar(x - w/2, ann_strat.values, w, label="Momentum Strategy", color=BLUE,  alpha=0.85)
    bars_b = ax.bar(x + w/2, ann_bench.values, w, label="S&P 500",           color=GRAY,  alpha=0.7)

    # Color bars by positive/negative
    for bar, val in zip(bars_s, ann_strat.values):
        bar.set_color(GREEN if val >= 0 else RED)
        bar.set_alpha(0.85)

    ax.axhline(0, color="white", lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Annual Returns: Momentum Strategy vs S&P 500")
    ax.set_ylabel("Annual Return")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "5_annual_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# SUMMARY TABLE 
def print_summary(strat: pd.Series, bench: pd.Series):
    """Prints a clean summary table to terminal."""
    def ann_ret(r): return (1 + r).prod() ** (12 / len(r)) - 1
    def ann_vol(r): return r.std() * np.sqrt(12)
    def sharpe(r):  return ann_ret(r) / ann_vol(r)
    def max_dd(r):
        c = (1 + r).cumprod()
        return ((c - c.cummax()) / c.cummax()).min()

    aligned = pd.concat([strat, bench], axis=1).dropna()
    s, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

    cov    = np.cov(s, b)
    beta   = cov[0, 1] / cov[1, 1]
    alpha  = (ann_ret(s) - beta * ann_ret(b))

    print("\n" + "═" * 52)
    print(f"{'FINAL PERFORMANCE SUMMARY':^52}")
    print("═" * 52)
    print(f"  {'Metric':<28} {'Strategy':>10} {'S&P 500':>10}")
    print("─" * 52)
    print(f"  {'Annualized Return':<28} {ann_ret(s):>10.2%} {ann_ret(b):>10.2%}")
    print(f"  {'Annualized Volatility':<28} {ann_vol(s):>10.2%} {ann_vol(b):>10.2%}")
    print(f"  {'Sharpe Ratio':<28} {sharpe(s):>10.2f} {sharpe(b):>10.2f}")
    print(f"  {'Max Drawdown':<28} {max_dd(s):>10.2%} {max_dd(b):>10.2%}")
    print(f"  {'Win Rate (monthly)':<28} {(s > 0).mean():>10.2%} {(b > 0).mean():>10.2%}")
    print("─" * 52)
    print(f"  {'Alpha (annualized)':<28} {alpha:>10.2%}")
    print(f"  {'Beta':<28} {beta:>10.3f}")
    print(f"  {'Backtest Period':<28} {s.index[0].strftime('%b %Y'):>10} {'→':>3} {s.index[-1].strftime('%b %Y'):<8}")
    print("═" * 52)


# MAIN 
if __name__ == "__main__":
    print("=" * 60)
    print("MOMENTUM BACKTEST — ANALYTICS & VISUALIZATION")
    print("=" * 60)

    strat, bench = load_results()

    print("\nGenerating charts...")
    plot_cumulative_returns(strat, bench)
    plot_return_distribution(strat, bench)
    plot_drawdown(strat, bench)
    plot_rolling_sharpe(strat)
    plot_annual_returns(strat, bench)

    print_summary(strat, bench)
    print(f"\nAll charts saved to ./{CHARTS_DIR}/")
    print("Project complete.")

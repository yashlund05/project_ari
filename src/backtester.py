"""
backtester.py
-------------
Portfolio backtesting engine for project_ari.

Loads trading signals, computes a full suite of risk-adjusted performance
metrics, generates equity curve and drawdown series, and persists a
professional performance report alongside the numerical outputs.

Project structure expected:
    project_ari/
    ├── data/
    │   ├── processed/
    │   │   └── trading_signals.csv   ← input
    │   └── results/
    │       ├── equity_curve.csv      ← output
    │       ├── drawdown_series.csv   ← output
    │       └── performance_summary.txt ← output
    └── src/
        └── backtester.py

Usage:
    Run directly:   python src/backtester.py
    Import:         from src.backtester import backtest_pipeline
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

INPUT_PATH       = _PROJECT_ROOT / "data" / "processed" / "trading_signals.csv"
RESULTS_DIR      = _PROJECT_ROOT / "data" / "results"
EQUITY_PATH      = RESULTS_DIR / "equity_curve.csv"
DRAWDOWN_PATH    = RESULTS_DIR / "drawdown_series.csv"
SUMMARY_PATH     = RESULTS_DIR / "performance_summary.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int   = 252
RISK_FREE_RATE_DAILY: float  = 0.0                 # assume 0 % for crypto
REQUIRED_COLUMNS: set[str]   = {
    "date", "close", "signal", "position_size", "strategy_return",
}


# ---------------------------------------------------------------------------
# 1. load_signals
# ---------------------------------------------------------------------------
def load_signals(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the trading signals dataset produced by strategy_engine.py.

    Parameters
    ----------
    filepath : Path or str
        Path to trading_signals.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with *date* as ``datetime64[ns]``, sorted chronologically.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Trading signals file not found: {filepath}\n"
            "Run strategy_engine.py first to generate trading_signals.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Signals dataset is missing required column(s): {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    active = (df["signal"] != "HOLD").sum()
    print(
        f"[load_signals]  Loaded {len(df):,} rows from '{filepath}'  |  "
        f"Active signals (non-HOLD): {active:,}"
    )
    return df


# ---------------------------------------------------------------------------
# 2. compute_equity_curve
# ---------------------------------------------------------------------------
def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build the portfolio equity curve from per-row strategy returns.

    The curve starts at 1.0 and compounds daily:
        equity_t = Π(1 + strategy_return_i)  for i in [0, t]

    Parameters
    ----------
    df : pd.DataFrame
        Signals DataFrame containing *strategy_return*.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with *equity* column appended.
    """
    df = df.copy()
    df["equity"] = (1 + df["strategy_return"]).cumprod()

    start = df["equity"].iloc[0]
    end   = df["equity"].iloc[-1]
    print(
        f"[compute_equity_curve]  Equity curve built — "
        f"start: {start:.4f}  →  end: {end:.4f}  "
        f"(total: {(end - 1) * 100:+.2f} %)"
    )
    return df


# ---------------------------------------------------------------------------
# 3. compute_drawdown
# ---------------------------------------------------------------------------
def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the rolling drawdown series from the equity curve.

    Formula
    -------
        rolling_max_t = max(equity_0 … equity_t)
        drawdown_t    = (equity_t - rolling_max_t) / rolling_max_t

    Values are always ≤ 0; the minimum value is the maximum drawdown.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing *equity* column.

    Returns
    -------
    pd.DataFrame
        DataFrame with *rolling_max* and *drawdown* columns appended.
    """
    df = df.copy()
    df["rolling_max"] = df["equity"].cummax()
    df["drawdown"]    = (df["equity"] - df["rolling_max"]) / df["rolling_max"]

    max_dd   = df["drawdown"].min()
    avg_dd   = df["drawdown"].mean()
    under_wt = (df["drawdown"] < 0).sum()
    print(
        f"[compute_drawdown]  Drawdown series built — "
        f"max drawdown: {max_dd * 100:.2f} %  |  "
        f"avg drawdown: {avg_dd * 100:.2f} %  |  "
        f"rows underwater: {under_wt:,} / {len(df):,}"
    )
    return df


# ---------------------------------------------------------------------------
# 4. compute_metrics
# ---------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute a comprehensive suite of risk-adjusted performance metrics.

    Metrics
    -------
    cumulative_return  : Total return over the full period (decimal).
    annual_return      : CAGR approximated by geometric scaling.
    annual_volatility  : Annualised std of daily strategy returns.
    sharpe_ratio       : (annual_return - Rf) / annual_volatility.
    sortino_ratio      : annual_return / downside_deviation.
    max_drawdown       : Largest peak-to-trough decline (negative decimal).
    calmar_ratio       : annual_return / |max_drawdown|.
    win_rate           : Fraction of non-zero days with positive return.
    profit_factor      : Gross profit / gross loss (absolute).
    total_trades       : Count of BUY + SELL signals (non-HOLD rows).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *strategy_return*, *equity*, *drawdown*, *signal*.

    Returns
    -------
    dict
        Mapping of metric name → value.
    """
    r       = df["strategy_return"]
    equity  = df["equity"]
    n_days  = len(r)

    # ── Return metrics ──────────────────────────────────────────────────────
    cumulative_return = equity.iloc[-1] - 1.0
    years             = n_days / TRADING_DAYS_PER_YEAR
    annual_return     = (1 + cumulative_return) ** (1 / max(years, 1e-9)) - 1

    # ── Volatility ─────────────────────────────────────────────────────────
    excess            = r - RISK_FREE_RATE_DAILY
    annual_volatility = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Sharpe ─────────────────────────────────────────────────────────────
    sharpe_ratio = (
        (annual_return - RISK_FREE_RATE_DAILY * TRADING_DAYS_PER_YEAR)
        / annual_volatility
        if annual_volatility > 0 else np.nan
    )

    # ── Sortino ────────────────────────────────────────────────────────────
    downside      = r[r < 0]
    downside_std  = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = (
        annual_return / downside_std
        if downside_std > 0 else np.nan
    )

    # ── Drawdown ───────────────────────────────────────────────────────────
    max_drawdown  = df["drawdown"].min()
    calmar_ratio  = (
        annual_return / abs(max_drawdown)
        if max_drawdown < 0 else np.nan
    )

    # ── Trade statistics ───────────────────────────────────────────────────
    active_returns = r[r != 0]
    total_trades   = int((df["signal"] != "HOLD").sum())

    win_rate = (
        float((active_returns > 0).sum() / len(active_returns))
        if len(active_returns) > 0 else 0.0
    )

    gross_profit = active_returns[active_returns > 0].sum()
    gross_loss   = abs(active_returns[active_returns < 0].sum())
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.nan
    )

    # ── Date range ─────────────────────────────────────────────────────────
    date_start = str(df["date"].min().date())
    date_end   = str(df["date"].max().date())

    metrics = {
        "date_start"        : date_start,
        "date_end"          : date_end,
        "total_days"        : n_days,
        "total_trades"      : total_trades,
        "cumulative_return" : round(cumulative_return,  6),
        "annual_return"     : round(annual_return,      6),
        "annual_volatility" : round(annual_volatility,  6),
        "sharpe_ratio"      : round(sharpe_ratio,       4) if not np.isnan(sharpe_ratio)  else "N/A",
        "sortino_ratio"     : round(sortino_ratio,      4) if not np.isnan(sortino_ratio) else "N/A",
        "max_drawdown"      : round(max_drawdown,       6),
        "calmar_ratio"      : round(calmar_ratio,       4) if not np.isnan(calmar_ratio)  else "N/A",
        "win_rate"          : round(win_rate,           4),
        "profit_factor"     : round(profit_factor,      4) if not np.isnan(profit_factor) else "N/A",
    }

    print(f"[compute_metrics]  Metrics computed over {n_days:,} days "
          f"({date_start} → {date_end}).")
    return metrics


# ---------------------------------------------------------------------------
# 5. save_results
# ---------------------------------------------------------------------------
def save_results(
    df:           pd.DataFrame,
    metrics:      dict,
    equity_path:   Path | str = EQUITY_PATH,
    drawdown_path: Path | str = DRAWDOWN_PATH,
    summary_path:  Path | str = SUMMARY_PATH,
) -> None:
    """Persist equity curve, drawdown series, and performance summary to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Full signals DataFrame with *equity* and *drawdown* columns.
    metrics : dict
        Performance metrics dictionary from :func:`compute_metrics`.
    equity_path, drawdown_path, summary_path : Path or str
        Destination file paths.
    """
    equity_path   = Path(equity_path)
    drawdown_path = Path(drawdown_path)
    summary_path  = Path(summary_path)

    for path in (equity_path, drawdown_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    # ── Equity curve ───────────────────────────────────────────────────────
    equity_df = df[["date", "close", "strategy_return", "equity"]].copy()
    equity_df.to_csv(equity_path, index=False)
    print(f"[save_results]  Equity curve saved      → '{equity_path}'")

    # ── Drawdown series ────────────────────────────────────────────────────
    dd_df = df[["date", "equity", "rolling_max", "drawdown"]].copy()
    dd_df.to_csv(drawdown_path, index=False)
    print(f"[save_results]  Drawdown series saved   → '{drawdown_path}'")

    # ── Performance summary (txt) ──────────────────────────────────────────
    report = _build_report(metrics)
    summary_path.write_text(report, encoding="utf-8")
    print(f"[save_results]  Performance report saved → '{summary_path}'")


# ---------------------------------------------------------------------------
# 6. backtest_pipeline
# ---------------------------------------------------------------------------
def backtest_pipeline(
    input_path:    Path | str = INPUT_PATH,
    equity_path:   Path | str = EQUITY_PATH,
    drawdown_path: Path | str = DRAWDOWN_PATH,
    summary_path:  Path | str = SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Execute the complete backtesting pipeline end-to-end.

    Steps
    -----
    1. Load trading signals        (:func:`load_signals`).
    2. Build equity curve          (:func:`compute_equity_curve`).
    3. Compute drawdown series     (:func:`compute_drawdown`).
    4. Calculate performance metrics (:func:`compute_metrics`).
    5. Print professional report.
    6. Persist all outputs         (:func:`save_results`).

    Parameters
    ----------
    input_path    : Path or str
    equity_path   : Path or str
    drawdown_path : Path or str
    summary_path  : Path or str

    Returns
    -------
    df : pd.DataFrame
        Enriched signals DataFrame with equity and drawdown columns.
    metrics : dict
        Full performance metrics dictionary.
    """
    df = load_signals(input_path)
    df = compute_equity_curve(df)
    df = compute_drawdown(df)

    metrics = compute_metrics(df)

    _print_report(metrics)

    save_results(
        df, metrics,
        equity_path   = equity_path,
        drawdown_path = drawdown_path,
        summary_path  = summary_path,
    )
    return df, metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _fmt_pct(value, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    if isinstance(value, str):
        return value
    return f"{value * 100:+.{decimals}f} %"


def _fmt_float(value, decimals: int = 4) -> str:
    """Format a plain float."""
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}"


def _build_report(metrics: dict) -> str:
    """Render the performance metrics as a structured text report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w  = 54   # report width

    lines = [
        "=" * w,
        " PROJECT ARI — BACKTEST PERFORMANCE REPORT".center(w),
        f" Generated: {ts}".center(w),
        "=" * w,
        "",
        "  PERIOD",
        f"  {'Start date':<28} {metrics['date_start']}",
        f"  {'End date':<28} {metrics['date_end']}",
        f"  {'Total trading days':<28} {metrics['total_days']:,}",
        f"  {'Total trades':<28} {metrics['total_trades']:,}",
        "",
        "  RETURNS",
        f"  {'Cumulative return':<28} {_fmt_pct(metrics['cumulative_return'])}",
        f"  {'Annualised return (CAGR)':<28} {_fmt_pct(metrics['annual_return'])}",
        f"  {'Annualised volatility':<28} {_fmt_pct(metrics['annual_volatility'])}",
        "",
        "  RISK-ADJUSTED METRICS",
        f"  {'Sharpe ratio':<28} {_fmt_float(metrics['sharpe_ratio'])}",
        f"  {'Sortino ratio':<28} {_fmt_float(metrics['sortino_ratio'])}",
        f"  {'Calmar ratio':<28} {_fmt_float(metrics['calmar_ratio'])}",
        "",
        "  DRAWDOWN",
        f"  {'Maximum drawdown':<28} {_fmt_pct(metrics['max_drawdown'])}",
        "",
        "  TRADE STATISTICS",
        f"  {'Win rate':<28} {_fmt_pct(metrics['win_rate'])}",
        f"  {'Profit factor':<28} {_fmt_float(metrics['profit_factor'])}",
        "",
        "=" * w,
    ]
    return "\n".join(lines)


def _print_report(metrics: dict) -> None:
    """Print the formatted performance report to stdout."""
    print()
    print(_build_report(metrics))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: run the backtesting pipeline."""
    print("=" * 60)
    print("  Backtester Pipeline — project_ari")
    print("=" * 60)
    try:
        df, metrics = backtest_pipeline()
        print(
            f"Backtest complete.  "
            f"Cumulative return: {metrics['cumulative_return'] * 100:+.2f} %  |  "
            f"Sharpe: {metrics['sharpe_ratio']}\n"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
backtester.py
-------------
Competition-grade portfolio backtesting engine for project_ari.
Upgraded for Techkriti competition requirements.

Enhancements over v1
--------------------
  • Transaction cost model  — 0.15 % fee per trade deducted from returns.
  • Stop-loss rule          — exit (zero return) when cumulative trade
                              return falls below -5 %.
  • Buy-and-hold benchmark  — BTC return over the same period.
  • Profit breakdown        — Gross Profit / Gross Loss / Net Profit.
  • Competition metrics     — full Techkriti scorecard output.
  • Full dataset backtest   — 2020–2023 coverage enforced.

Project structure expected:
    project_ari/
    ├── data/
    │   ├── processed/
    │   │   └── trading_signals.csv      ← input
    │   └── results/
    │       ├── equity_curve.csv         ← output
    │       ├── drawdown_series.csv      ← output
    │       └── performance_summary.txt  ← output
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

INPUT_PATH    = _PROJECT_ROOT / "data" / "processed" / "trading_signals.csv"
RESULTS_DIR   = _PROJECT_ROOT / "data" / "results"
EQUITY_PATH   = RESULTS_DIR / "equity_curve.csv"
DRAWDOWN_PATH = RESULTS_DIR / "drawdown_series.csv"
SUMMARY_PATH  = RESULTS_DIR / "performance_summary.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int  = 252
RISK_FREE_RATE_DAILY: float = 0.0          # 0 % risk-free rate for crypto

TRANSACTION_FEE_RATE: float = 0.0015       # 0.15 % per trade (one-way)
STOP_LOSS_THRESHOLD: float  = -0.05        # exit trade if cumulative return < -5 %

REQUIRED_COLUMNS: set[str] = {
    "date", "close", "signal", "position_size", "strategy_return", "target_return",
}


# ===========================================================================
# 1. load_signals
# ===========================================================================
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

    active   = (df["signal"] != "HOLD").sum()
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    print(
        f"[load_signals]  Loaded {len(df):,} rows from '{filepath}'  |  "
        f"Period: {date_min} → {date_max}  |  "
        f"Active signals (non-HOLD): {active:,}"
    )
    return df


# ===========================================================================
# 2. compute_positions  (new)
# ===========================================================================
def compute_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Build a continuous position column from the signal series.

    The strategy uses position carry: a BUY opens a long (position = 1),
    a SELL closes it (position = 0), and HOLD rows carry the previous
    position forward unchanged.  This reflects real holding behaviour —
    the strategy stays exposed through multi-day uptrends without exiting
    and re-entering on every HOLD row.

    Position encoding
    -----------------
    1   → long (BUY signal received, or carried from prior BUY)
    0   → flat (SELL signal received, or no position yet opened)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a *signal* column (BUY / SELL / HOLD).

    Returns
    -------
    pd.DataFrame
        DataFrame with *position* column appended (int, 0 or 1).
    """
    df = df.copy()

    position  = 0
    positions = []

    for sig in df["signal"]:
        if sig == "BUY":
            position = 1
        elif sig == "SELL":
            position = 0
        # HOLD → keep previous position unchanged

        positions.append(position)

    df["position"] = positions

    long_days = int(sum(p == 1 for p in positions))
    flat_days = int(sum(p == 0 for p in positions))
    # Count position changes (transitions between 0 and 1)
    position_changes = int(sum(
        1 for i in range(1, len(positions))
        if positions[i] != positions[i - 1]
    ))

    print(
        f"[positions]  long_days: {long_days:,}  |  "
        f"flat_days: {flat_days:,}  |  "
        f"position_changes: {position_changes:,}"
    )
    return df


# ===========================================================================
# 3. compute_strategy_returns  (new)
# ===========================================================================
def compute_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute strategy_return as position × market_return.

    Overwrites the *strategy_return* column that was pre-computed by
    strategy_engine.py with a return series that correctly reflects
    carry exposure: on every day the position is long (position = 1),
    the strategy earns the market's daily return; on flat days
    (position = 0) it earns nothing.

    The *position_size* column scales the exposure so that sizing logic
    from the strategy engine is honoured.

    Formula
    -------
        strategy_return = position × position_size × target_return

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *position*, *position_size*, and *target_return*.

    Returns
    -------
    pd.DataFrame
        DataFrame with *strategy_return* recomputed.
    """
    df = df.copy()
    df["strategy_return"] = df["position"] * df["position_size"] * df["target_return"]

    cumulative = df["strategy_return"].sum()
    print(
        f"[compute_strategy_returns]  "
        f"Cumulative gross return: {cumulative:+.4f}  "
        f"({cumulative * 100:+.2f} %)"
    )
    return df


# ===========================================================================
# 4. apply_transaction_costs  (modified — charges on position changes only)
# ===========================================================================
def apply_transaction_costs(
    df: pd.DataFrame,
    fee_rate: float = TRANSACTION_FEE_RATE,
) -> pd.DataFrame:
    """Deduct transaction costs only when the position changes.

    A fee of *fee_rate* (0.15 %) is subtracted from *strategy_return*
    on rows where the *position* column changes value relative to the
    previous row (i.e. an actual trade took place).  HOLD rows that carry
    an existing position are unaffected — no new trade means no new cost.

    The raw pre-cost return is preserved in *gross_return*.

    Parameters
    ----------
    df : pd.DataFrame
        Signals DataFrame with *position*, *position_size*, and
        *strategy_return* columns.
    fee_rate : float
        One-way transaction fee as a decimal fraction.  Default 0.0015.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
          • *gross_return*     — strategy_return before cost deduction.
          • *transaction_cost* — fee charged on position-change rows (0 elsewhere).
          • *strategy_return*  — net return after cost deduction (in-place).
    """
    df = df.copy()
    df["gross_return"] = df["strategy_return"].copy()

    # A cost is incurred only when position changes (a real trade occurs)
    prev_position = df["position"].shift(1).fillna(0).astype(int)
    position_changed = df["position"] != prev_position

    df["transaction_cost"] = 0.0
    df.loc[position_changed, "transaction_cost"] = (
        fee_rate * df.loc[position_changed, "position_size"]
    )
    df["strategy_return"] = df["gross_return"] - df["transaction_cost"]

    total_cost = df["transaction_cost"].sum()
    n_charged  = position_changed.sum()
    print(
        f"[apply_transaction_costs]  Fee rate: {fee_rate * 100:.3f} %  |  "
        f"Trades charged: {n_charged:,}  |  "
        f"Total cost drag: {total_cost * 100:.4f} %"
    )
    return df


# ===========================================================================
# 5. apply_stop_loss
# ===========================================================================
def apply_stop_loss(
    df: pd.DataFrame,
    threshold: float = STOP_LOSS_THRESHOLD,
) -> pd.DataFrame:
    """Zero out returns once a cumulative stop-loss level is breached.

    Logic
    -----
    A running cumulative return is tracked across consecutive active
    (non-HOLD) rows.  Once the running sum falls below *threshold*
    (-5 %), the position is considered exited — the current and all
    subsequent rows in that active run are zeroed out.  The counter
    resets when a HOLD row is encountered (new trade begins).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *signal* and *strategy_return* columns.
    threshold : float
        Stop-loss level as a negative decimal fraction.  Default -0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame with *strategy_return* zeroed on stopped-out rows and a
        boolean *stop_loss_triggered* column appended.
    """
    df = df.copy()
    df["stop_loss_triggered"] = False

    cum_return  = 0.0
    stopped     = False
    triggered_n = 0

    for idx in df.index:
        sig = df.at[idx, "signal"]

        if sig == "HOLD":
            cum_return = 0.0
            stopped    = False
            continue

        if stopped:
            df.at[idx, "strategy_return"]     = 0.0
            df.at[idx, "stop_loss_triggered"] = True
            triggered_n += 1
            continue

        cum_return += df.at[idx, "strategy_return"]

        if cum_return < threshold:
            df.at[idx, "strategy_return"]     = 0.0
            df.at[idx, "stop_loss_triggered"] = True
            stopped      = True
            triggered_n += 1

    print(
        f"[apply_stop_loss]  Threshold: {threshold * 100:.1f} %  |  "
        f"Rows zeroed by stop-loss: {triggered_n:,}"
    )
    return df


# ===========================================================================
# 6. compute_buy_and_hold
# ===========================================================================
def compute_buy_and_hold(df: pd.DataFrame) -> float:
    """Compute the passive BTC buy-and-hold return over the backtest period.

    Formula
    -------
        bnh = (close_last − close_first) / close_first

    Parameters
    ----------
    df : pd.DataFrame
        Signals DataFrame containing *close* prices.

    Returns
    -------
    float
        Buy-and-hold return as a decimal fraction.
    """
    close_first = df["close"].iloc[0]
    close_last  = df["close"].iloc[-1]
    bnh         = (close_last - close_first) / close_first

    print(
        f"[compute_buy_and_hold]  BTC close: "
        f"{close_first:,.2f} → {close_last:,.2f}  |  "
        f"Buy-and-hold return: {bnh * 100:+.2f} %"
    )
    return float(bnh)


# ===========================================================================
# 7. compute_equity_curve
# ===========================================================================
def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build the portfolio equity curve from net (post-cost) strategy returns.

    Equity starts at 1.0 and compounds daily:
        equity_t = Π(1 + strategy_return_i)  for i ∈ [0, t]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *strategy_return* (net of costs and stop-loss).

    Returns
    -------
    pd.DataFrame
        DataFrame with *equity* column appended.
    """
    df = df.copy()
    df["equity"] = (1 + df["strategy_return"]).cumprod()

    start = df["equity"].iloc[0]
    end   = df["equity"].iloc[-1]
    print(
        f"[compute_equity_curve]  Equity curve built — "
        f"start: {start:.4f}  →  end: {end:.4f}  "
        f"(net total: {(end - 1) * 100:+.2f} %)"
    )
    return df


# ===========================================================================
# 8. compute_drawdown
# ===========================================================================
def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the rolling drawdown series from the equity curve.

    Formula
    -------
        rolling_max_t = max(equity_0 … equity_t)
        drawdown_t    = (equity_t − rolling_max_t) / rolling_max_t

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *equity* column.

    Returns
    -------
    pd.DataFrame
        DataFrame with *rolling_max* and *drawdown* columns appended.
    """
    df = df.copy()
    df["rolling_max"] = df["equity"].cummax()
    df["drawdown"]    = (df["equity"] - df["rolling_max"]) / df["rolling_max"]

    max_dd     = df["drawdown"].min()
    avg_dd     = df["drawdown"].mean()
    underwater = (df["drawdown"] < 0).sum()
    print(
        f"[compute_drawdown]  Drawdown series built — "
        f"max: {max_dd * 100:.2f} %  |  "
        f"avg: {avg_dd * 100:.2f} %  |  "
        f"underwater rows: {underwater:,} / {len(df):,}"
    )
    return df


# ===========================================================================
# 9. compute_metrics
# ===========================================================================
def compute_metrics(
    df: pd.DataFrame,
    bnh_return: float,
) -> dict[str, float | int | str]:
    """Compute the full Techkriti competition scorecard metrics.

    Metrics
    -------
    gross_profit         : Sum of positive net strategy returns.
    gross_loss           : Absolute sum of negative net strategy returns.
    net_profit           : gross_profit − gross_loss.
    total_cost_drag      : Total transaction costs deducted.
    buy_hold_return      : Passive BTC return over the same period.
    alpha                : Strategy return minus buy-and-hold return.
    cumulative_return    : Total net return (decimal).
    annual_return        : CAGR over the backtest period.
    annual_volatility    : Annualised std of daily net returns.
    sharpe_ratio         : Annualised Sharpe (Rf = 0).
    sortino_ratio        : Annualised Sortino using downside deviation only.
    calmar_ratio         : annual_return / |max_drawdown|.
    max_drawdown         : Largest peak-to-trough equity decline.
    total_closed_trades  : Count of BUY + SELL signal rows.
    stopped_trades       : Rows zeroed by stop-loss rule.
    win_rate             : Fraction of active rows with positive net return.
    profit_factor        : gross_profit / gross_loss.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched signals DataFrame with all computed columns.
    bnh_return : float
        Buy-and-hold return from :func:`compute_buy_and_hold`.

    Returns
    -------
    dict
        Complete Techkriti competition metrics mapping.
    """
    r      = df["strategy_return"]
    equity = df["equity"]
    n_days = len(r)

    # ── Profit breakdown ────────────────────────────────────────────────────
    active_net   = r[df["signal"] != "HOLD"]
    gross_profit = float(active_net[active_net > 0].sum())
    gross_loss   = float(abs(active_net[active_net < 0].sum()))
    net_profit   = gross_profit - gross_loss
    total_cost   = float(df["transaction_cost"].sum()) if "transaction_cost" in df.columns else 0.0

    # ── Return metrics ──────────────────────────────────────────────────────
    cumulative_return = float(equity.iloc[-1] - 1.0)
    years             = n_days / TRADING_DAYS_PER_YEAR
    annual_return     = float((1 + cumulative_return) ** (1 / max(years, 1e-9)) - 1)
    alpha             = cumulative_return - bnh_return

    # ── Volatility ─────────────────────────────────────────────────────────
    annual_volatility = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ── Sharpe ─────────────────────────────────────────────────────────────
    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility > 0 else np.nan
    )

    # ── Sortino ────────────────────────────────────────────────────────────
    downside_std  = float(r[r < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino_ratio = (
        annual_return / downside_std
        if downside_std > 0 else np.nan
    )

    # ── Drawdown / Calmar ───────────────────────────────────────────────────
    max_drawdown = float(df["drawdown"].min())
    calmar_ratio = (
        annual_return / abs(max_drawdown)
        if max_drawdown < 0 else np.nan
    )

    # ── Trade statistics ────────────────────────────────────────────────────
    total_closed_trades = int((df["signal"] != "HOLD").sum())
    stopped_trades      = (
        int(df["stop_loss_triggered"].sum())
        if "stop_loss_triggered" in df.columns else 0
    )
    win_rate = (
        float((active_net > 0).sum() / len(active_net))
        if len(active_net) > 0 else 0.0
    )
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.nan
    )

    # ── Date range ──────────────────────────────────────────────────────────
    date_start = str(df["date"].min().date())
    date_end   = str(df["date"].max().date())

    def _safe(v, d=4):
        return round(v, d) if (isinstance(v, float) and not np.isnan(v)) else "N/A"

    metrics = {
        "date_start"          : date_start,
        "date_end"            : date_end,
        "total_days"          : n_days,
        # ── Competition scorecard ────────────────────────────────────────
        "gross_profit"        : round(gross_profit,      6),
        "gross_loss"          : round(gross_loss,        6),
        "net_profit"          : round(net_profit,        6),
        "total_cost_drag"     : round(total_cost,        6),
        "buy_hold_return"     : round(bnh_return,        6),
        "alpha"               : round(alpha,             6),
        "cumulative_return"   : round(cumulative_return, 6),
        "annual_return"       : round(annual_return,     6),
        "annual_volatility"   : round(annual_volatility, 6),
        "sharpe_ratio"        : _safe(sharpe_ratio),
        "sortino_ratio"       : _safe(sortino_ratio),
        "calmar_ratio"        : _safe(calmar_ratio),
        "max_drawdown"        : round(max_drawdown,      6),
        "total_closed_trades" : total_closed_trades,
        "stopped_trades"      : stopped_trades,
        "win_rate"            : round(win_rate,          4),
        "profit_factor"       : _safe(profit_factor),
    }

    print(
        f"[compute_metrics]  {n_days:,} days  ({date_start} → {date_end})  |  "
        f"Net profit: {net_profit * 100:+.2f} %  |  "
        f"B&H: {bnh_return * 100:+.2f} %  |  "
        f"Alpha: {alpha * 100:+.2f} %"
    )
    return metrics


# ===========================================================================
# 10. save_results
# ===========================================================================
def save_results(
    df:            pd.DataFrame,
    metrics:       dict,
    equity_path:   Path | str = EQUITY_PATH,
    drawdown_path: Path | str = DRAWDOWN_PATH,
    summary_path:  Path | str = SUMMARY_PATH,
) -> None:
    """Persist equity curve, drawdown series, and competition report to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched signals DataFrame.
    metrics : dict
        Performance metrics from :func:`compute_metrics`.
    equity_path, drawdown_path, summary_path : Path or str
        Output file paths.
    """
    for path in map(Path, (equity_path, drawdown_path, summary_path)):
        path.parent.mkdir(parents=True, exist_ok=True)

    # ── Equity curve ───────────────────────────────────────────────────────
    eq_want = ["date", "close", "signal", "position", "gross_return",
               "transaction_cost", "strategy_return", "equity"]
    eq_cols = [c for c in eq_want if c in df.columns]
    df[eq_cols].to_csv(equity_path, index=False)
    print(f"[save_results]  Equity curve saved       → '{equity_path}'")

    # ── Drawdown series ────────────────────────────────────────────────────
    dd_want = ["date", "equity", "rolling_max", "drawdown"]
    dd_cols = [c for c in dd_want if c in df.columns]
    df[dd_cols].to_csv(drawdown_path, index=False)
    print(f"[save_results]  Drawdown series saved    → '{drawdown_path}'")

    # ── Performance report ─────────────────────────────────────────────────
    report = _build_report(metrics)
    Path(summary_path).write_text(report, encoding="utf-8")
    print(f"[save_results]  Performance report saved → '{summary_path}'")


# ===========================================================================
# 11. backtest_pipeline
# ===========================================================================
def backtest_pipeline(
    input_path:    Path | str = INPUT_PATH,
    equity_path:   Path | str = EQUITY_PATH,
    drawdown_path: Path | str = DRAWDOWN_PATH,
    summary_path:  Path | str = SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Execute the complete Techkriti-grade backtesting pipeline.

    Steps
    -----
    1.  Load trading signals                   (:func:`load_signals`).
    2.  Build position series from signals     (:func:`compute_positions`).
    3.  Recompute strategy returns             (:func:`compute_strategy_returns`).
    4.  Apply transaction costs on changes     (:func:`apply_transaction_costs`).
    5.  Apply stop-loss rule (−5 %)            (:func:`apply_stop_loss`).
    6.  Compute buy-and-hold benchmark         (:func:`compute_buy_and_hold`).
    7.  Build equity curve                     (:func:`compute_equity_curve`).
    8.  Compute drawdown series                (:func:`compute_drawdown`).
    9.  Calculate competition metrics          (:func:`compute_metrics`).
    10. Print professional scorecard.
    11. Persist all outputs                    (:func:`save_results`).

    Returns
    -------
    df : pd.DataFrame
        Fully enriched signals DataFrame.
    metrics : dict
        Techkriti competition scorecard.
    """
    df         = load_signals(input_path)
    df         = compute_positions(df)
    df         = compute_strategy_returns(df)
    df         = apply_transaction_costs(df)
    df         = apply_stop_loss(df)
    bnh_return = compute_buy_and_hold(df)
    df         = compute_equity_curve(df)
    df         = compute_drawdown(df)
    metrics    = compute_metrics(df, bnh_return)

    _print_report(metrics)

    save_results(
        df, metrics,
        equity_path   = equity_path,
        drawdown_path = drawdown_path,
        summary_path  = summary_path,
    )
    return df, metrics


# ===========================================================================
# Private helpers
# ===========================================================================
def _fmt_pct(value, decimals: int = 2) -> str:
    """Format a decimal fraction as a signed percentage string."""
    if isinstance(value, str):
        return value
    return f"{value * 100:+.{decimals}f} %"


def _fmt_float(value, decimals: int = 4) -> str:
    """Format a plain float to *decimals* places."""
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}"


def _build_report(metrics: dict) -> str:
    """Render the Techkriti competition scorecard as a structured text report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w  = 62

    lines = [
        "=" * w,
        "   PROJECT ARI — TECHKRITI COMPETITION SCORECARD".center(w),
        f"   Generated : {ts}".center(w),
        "=" * w,
        "",
        "  ── PERIOD ─────────────────────────────────────────────────",
        f"  {'Start date':<32} {metrics['date_start']}",
        f"  {'End date':<32} {metrics['date_end']}",
        f"  {'Total trading days':<32} {metrics['total_days']:,}",
        "",
        "  ── PROFIT BREAKDOWN ────────────────────────────────────────",
        f"  {'Gross Profit':<32} {_fmt_pct(metrics['gross_profit'])}",
        f"  {'Gross Loss':<32} {_fmt_pct(metrics['gross_loss'])}",
        f"  {'Net Profit':<32} {_fmt_pct(metrics['net_profit'])}",
        f"  {'Transaction Cost Drag':<32} {_fmt_pct(metrics['total_cost_drag'])}",
        "",
        "  ── BENCHMARK COMPARISON ────────────────────────────────────",
        f"  {'Buy & Hold Return (BTC)':<32} {_fmt_pct(metrics['buy_hold_return'])}",
        f"  {'Strategy Net Return':<32} {_fmt_pct(metrics['cumulative_return'])}",
        f"  {'Alpha vs B&H':<32} {_fmt_pct(metrics['alpha'])}",
        f"  {'Annualised Return (CAGR)':<32} {_fmt_pct(metrics['annual_return'])}",
        f"  {'Annualised Volatility':<32} {_fmt_pct(metrics['annual_volatility'])}",
        "",
        "  ── RISK-ADJUSTED METRICS ───────────────────────────────────",
        f"  {'Sharpe Ratio':<32} {_fmt_float(metrics['sharpe_ratio'])}",
        f"  {'Sortino Ratio':<32} {_fmt_float(metrics['sortino_ratio'])}",
        f"  {'Calmar Ratio':<32} {_fmt_float(metrics['calmar_ratio'])}",
        f"  {'Max Drawdown':<32} {_fmt_pct(metrics['max_drawdown'])}",
        "",
        "  ── TRADE STATISTICS ────────────────────────────────────────",
        f"  {'Total Closed Trades':<32} {metrics['total_closed_trades']:,}",
        f"  {'Stop-Loss Triggered (rows)':<32} {metrics['stopped_trades']:,}",
        f"  {'Win Rate':<32} {_fmt_pct(metrics['win_rate'])}",
        f"  {'Profit Factor':<32} {_fmt_float(metrics['profit_factor'])}",
        "",
        "=" * w,
    ]
    return "\n".join(lines)


def _print_report(metrics: dict) -> None:
    """Print the formatted competition scorecard to stdout."""
    print()
    print(_build_report(metrics))
    print()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """CLI entry point: run the competition-grade backtesting pipeline."""
    print("=" * 62)
    print("  Backtester Pipeline (Techkriti) — project_ari")
    print("=" * 62)
    try:
        df, metrics = backtest_pipeline()
        print(
            f"Backtest complete.  "
            f"Net profit: {metrics['net_profit'] * 100:+.2f} %  |  "
            f"B&H: {metrics['buy_hold_return'] * 100:+.2f} %  |  "
            f"Alpha: {metrics['alpha'] * 100:+.2f} %  |  "
            f"Sharpe: {metrics['sharpe_ratio']}\n"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
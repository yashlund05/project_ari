"""
backtester.py
-------------
Competition-grade portfolio backtesting engine for Project ARI.

v3 — Regime-Gated Alignment
-----------------------------
Updated to handle the outputs of the v13 Regime-Gated Strategy Engine.
All four requirements from the ARI specification are addressed:

  1. Strict Position Alignment
     compute_positions() treats rows where signal=="HOLD" AND position_size==0
     as a Hard Cash Lock: position is forced to 0 (not carried forward).
     This correctly represents that the regime gate physically moves the
     portfolio to cash, not to a "held long with zero size" state.

  2. Corrected Transaction Fee Logic
     apply_transaction_costs() only charges a fee when the position change
     was caused by an actual BUY or SELL signal.  Hard Cash Lock transitions
     (signal="HOLD", position_size=0) are explicitly excluded so the regime
     gate does not generate spurious round-trip costs.

  3. Institutional Scorecard — Regime-Gated Efficiency section
     compute_metrics() and _build_report() now include:
       • Total Signals Suppressed by Alpha Filter  (entropy gate + regime gate)
       • Equity Saved from High-Entropy Periods    (counterfactual loss avoided)

  4. Equity Curve Sanity
     compute_equity_curve() builds three parallel curves saved to equity_curve.csv:
       • lgbm_naive_equity  — pure LightGBM signal, no regime filter, no fees.
                              Baseline that shows what following raw predictions
                              alone would have yielded.
       • gross_equity       — regime-gated strategy, before 0.15 % transaction fees.
       • equity (net)       — regime-gated strategy, after all fees and stop-loss.
     Comparing lgbm_naive vs gross vs net isolates the contribution of each layer.

Enhancements retained from v2
------------------------------
  • 0.15 % per-trade transaction cost model.
  • Stop-loss rule: exit when cumulative trade return < -5 %.
  • Buy-and-hold BTC benchmark.
  • Profit breakdown: Gross Profit / Gross Loss / Net Profit.
  • Full Techkriti competition scorecard.

Project structure expected
--------------------------
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

Usage
-----
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
RISK_FREE_RATE_DAILY: float = 0.0         # 0 % risk-free rate for crypto

TRANSACTION_FEE_RATE: float = 0.0015      # 0.15 % per trade (one-way)
STOP_LOSS_THRESHOLD:  float = -0.05       # exit when cumulative trade return < -5 %

# Minimum |predicted_return| required for a suppressed row to count as
# a "would-have-been active" signal.  Matches CONFIDENCE_THRESHOLD in v13.
SIGNAL_CONFIDENCE_FLOOR: float = 0.002

# Regime labels that are hard-locked to cash by the Regime Gate (v13).
CASH_LOCKED_REGIMES: frozenset[str] = frozenset({"BULL", "RECOVERY"})

REQUIRED_COLUMNS: set[str] = {
    "date", "close", "signal", "position_size", "strategy_return", "target_return",
}


# ===========================================================================
# 1. load_signals
# ===========================================================================
def load_signals(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the trading signals dataset produced by strategy_engine.py v13.

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

    # Optional v13 columns — report availability for downstream steps.
    optional_present = [
        c for c in ("predicted_return", "regime_entropy", "regime_label")
        if c in df.columns
    ]
    print(
        f"[load_signals]  Loaded {len(df):,} rows from '{filepath}'  |  "
        f"Period: {date_min} → {date_max}  |  "
        f"Active signals (non-HOLD): {active:,}  |  "
        f"Optional v13 cols present: {optional_present}"
    )
    return df


# ===========================================================================
# 2. compute_positions   (v3: Hard Cash Lock override)
# ===========================================================================
def compute_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Build a continuous position column, with Hard Cash Lock support.

    Standard carry logic
    --------------------
    BUY  → position = 1  (enter long)
    SELL → position = 0  (exit to flat)
    HOLD → carry previous position forward

    Hard Cash Lock override (v3 addition)
    --------------------------------------
    When the v13 Regime Gate forces ``signal = "HOLD"`` on a BULL or
    RECOVERY row it also sets ``position_size = 0``.  Carrying the prior
    position through such rows would misrepresent the portfolio state:
    the strategy is *in cash*, not in a zero-sized long.

    Rule:  if  signal == "HOLD"  AND  position_size == 0
           then position = 0  (override carry; hard cash lock)

    This has two downstream effects:
      • compute_strategy_returns: position(0) × position_size(0) × target = 0  ✓
        (already correct either way, but semantically unambiguous)
      • apply_transaction_costs: when the lock lifts and a new BUY fires,
        the position transitions 0→1, which is detected as a real entry
        and charged a single round-trip fee.  Without this override the
        position stays at 1 through the lock and the re-entry is invisible.

    The ``is_hard_cash_lock`` boolean column is carried through the pipeline
    so ``apply_transaction_costs`` can exclude lock-boundary transitions
    from fee charges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *signal* and *position_size* columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with *position* (0 or 1) and *is_hard_cash_lock* columns.
    """
    df = df.copy()

    # Pre-compute the cash-lock mask (vectorised, no loop needed here).
    hard_cash_lock_mask = (df["signal"] == "HOLD") & (df["position_size"] == 0.0)

    position  = 0
    positions = []

    for idx, (sig, is_locked) in enumerate(
        zip(df["signal"], hard_cash_lock_mask)
    ):
        if sig == "BUY":
            position = 1
        elif sig == "SELL":
            position = 0
        elif is_locked:
            # Hard Cash Lock: override carry → force flat regardless of prior state.
            position = 0
        # Plain HOLD (position_size > 0): carry previous position unchanged.

        positions.append(position)

    df["position"]         = positions
    df["is_hard_cash_lock"] = hard_cash_lock_mask.values

    long_days      = int(sum(p == 1 for p in positions))
    flat_days      = int(sum(p == 0 for p in positions))
    lock_days      = int(hard_cash_lock_mask.sum())
    pos_changes    = int(sum(
        1 for i in range(1, len(positions))
        if positions[i] != positions[i - 1]
    ))

    print(
        f"[compute_positions]  long_days: {long_days:,}  |  "
        f"flat_days: {flat_days:,}  |  "
        f"hard_cash_lock_days: {lock_days:,}  |  "
        f"position_changes: {pos_changes:,}"
    )
    return df


# ===========================================================================
# 3. compute_strategy_returns
# ===========================================================================
def compute_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute strategy_return as position × position_size × target_return.

    With the v3 Hard Cash Lock override in compute_positions:
      • Cash-locked rows have position = 0 AND position_size = 0  → return = 0
      • Active rows have position = 1 (or 0 for SELL) AND non-zero position_size

    The *position_size* column is the primary return multiplier.  It already
    encodes all v13 sizing decisions:
      • Regime scale (BEAR ×1.20, SIDEWAYS ×1.10, BULL/RECOVERY ×0.00)
      • Inverse-volatility base sizing
      • 30-day vol-targeting adjustment
      • Confidence-floor scalar (0.6 – 1.5)
      • Heatmap Vol-Entropy 50 % reduction scalar

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *position*, *position_size*, and *target_return*.

    Returns
    -------
    pd.DataFrame
        DataFrame with *strategy_return* recomputed in-place.
    """
    df = df.copy()
    df["strategy_return"] = df["position"] * df["position_size"] * df["target_return"]

    cumulative = df["strategy_return"].sum()
    print(
        f"[compute_strategy_returns]  "
        f"Cumulative gross return (pre-fee): {cumulative:+.4f} "
        f"({cumulative * 100:+.2f} %)"
    )
    return df


# ===========================================================================
# 4. apply_transaction_costs   (v3: fee gated to real BUY/SELL signals only)
# ===========================================================================
def apply_transaction_costs(
    df:       pd.DataFrame,
    fee_rate: float = TRANSACTION_FEE_RATE,
) -> pd.DataFrame:
    """Deduct transaction costs only when a real BUY or SELL signal fires.

    Fee logic (v3)
    --------------
    A fee is charged when ALL THREE of the following hold simultaneously:

      1. The *position* column changes value relative to the prior row.
         (A real trade changed the portfolio's direction.)

      2. The current row's *signal* is BUY or SELL — not HOLD.
         This is the primary guard against Hard Cash Lock transitions.
         When the regime gate overrides a row to HOLD (position_size=0),
         the position may still change (0→1 or 1→0 at lock boundaries),
         but no real order was submitted, so no fee should be charged.

      3. The row is not flagged as *is_hard_cash_lock*.
         Second-level guard: ensures that even an edge case where a lock
         boundary coincides with a genuine signal cannot silently slip
         through without the double-check.

    The raw pre-cost return is preserved in *gross_return*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *position*, *position_size*, *signal*,
        *is_hard_cash_lock*, and *strategy_return* columns.
    fee_rate : float
        One-way transaction fee as a decimal fraction.  Default 0.0015.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
          • *gross_return*     — strategy_return before cost deduction.
          • *transaction_cost* — fee charged on real-trade rows (0 elsewhere).
          • *strategy_return*  — net return after cost deduction.
    """
    df = df.copy()
    df["gross_return"] = df["strategy_return"].copy()

    # ── Condition 1: position changed ────────────────────────────────────────
    prev_position    = df["position"].shift(1).fillna(0).astype(int)
    position_changed = df["position"] != prev_position

    # ── Condition 2: signal is a real trade directive (not HOLD) ─────────────
    real_signal = df["signal"].isin({"BUY", "SELL"})

    # ── Condition 3: not a Hard Cash Lock row ─────────────────────────────────
    not_locked = ~df["is_hard_cash_lock"]

    # Fee is charged only when all three conditions hold.
    fee_mask = position_changed & real_signal & not_locked

    n_lock_excluded = int((position_changed & ~real_signal).sum())
    n_charged       = int(fee_mask.sum())

    df["transaction_cost"] = 0.0
    df.loc[fee_mask, "transaction_cost"] = (
        fee_rate * df.loc[fee_mask, "position_size"]
    )
    df["strategy_return"] = df["gross_return"] - df["transaction_cost"]

    total_cost = df["transaction_cost"].sum()
    print(
        f"[apply_transaction_costs]  Fee rate: {fee_rate * 100:.3f} %  |  "
        f"Real-trade rows charged: {n_charged:,}  |  "
        f"Hard-cash-lock transitions excluded: {n_lock_excluded:,}  |  "
        f"Total cost drag: {total_cost * 100:.4f} %"
    )
    return df


# ===========================================================================
# 5. apply_stop_loss
# ===========================================================================
def apply_stop_loss(
    df:        pd.DataFrame,
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

    Hard Cash Lock rows (signal=HOLD) reset the stop-loss counter
    identically to ordinary HOLD rows: the regime gate forces a full
    exit to cash, so any prior trade's running loss is cleared.

    Parameters
    ----------
    df : pd.DataFrame
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
            # Both ordinary HOLDs and Hard Cash Lock HOLDs reset the counter.
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
# 7. compute_equity_curve   (v3: gross_equity + lgbm_naive_equity added)
# ===========================================================================
def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build three parallel equity curves for layered performance attribution.

    Curves
    ------
    lgbm_naive_equity (v3 NEW)
        Equity from naively following the sign of LightGBM's predicted_return
        with a fixed unit position and no regime filter, no fees, no stop-loss.
        Computed as:  cumprod(1 + sign(predicted_return) × target_return)
        Purpose: isolates the raw model signal's contribution before any
        risk management layer is applied.  If this is worse than gross_equity,
        the regime gate is adding value; if better, it is not.

    gross_equity (v3 NEW)
        Equity from the regime-gated strategy *before* 0.15 % transaction fees
        and before stop-loss zeroing.
        Computed as:  cumprod(1 + gross_return)
        Purpose: isolates the regime-filter + sizing contribution.
        The gap between gross_equity and net equity quantifies pure fee drag.

    equity (net)
        Final strategy equity after all filters, fees, and stop-loss.
        Computed as:  cumprod(1 + strategy_return)
        This is the authoritative performance number.

    Comparing the three curves at any point in time tells you:
        lgbm_naive → gross  : value added by regime gating and sizing
        gross → net         : cost of transaction fees and stop-loss exits

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *strategy_return* and *gross_return*.
        Optionally contains *predicted_return* and *target_return* for
        the LightGBM naive curve (skipped with a warning if absent).

    Returns
    -------
    pd.DataFrame
        DataFrame with *equity*, *gross_equity*, and (if available)
        *lgbm_naive_equity* columns appended.
    """
    df = df.copy()

    # ── Net equity (authoritative) ────────────────────────────────────────
    df["equity"] = (1.0 + df["strategy_return"]).cumprod()

    # ── Gross equity (pre-fee, post regime-gate) ──────────────────────────
    gross_col = "gross_return" if "gross_return" in df.columns else "strategy_return"
    df["gross_equity"] = (1.0 + df[gross_col]).cumprod()

    # ── LightGBM naive equity (no regime filter, no fees) ─────────────────
    if "predicted_return" in df.columns and "target_return" in df.columns:
        # direction: +1 if model predicts up, -1 if down, 0 if no signal
        lgbm_dir = np.sign(df["predicted_return"].fillna(0))
        lgbm_daily = lgbm_dir * df["target_return"]
        df["lgbm_naive_equity"] = (1.0 + lgbm_daily).cumprod()
        lgbm_end = float(df["lgbm_naive_equity"].iloc[-1])
        lgbm_note = f"  |  LightGBM naive end: {lgbm_end:.4f} ({(lgbm_end-1)*100:+.2f} %)"
    else:
        df["lgbm_naive_equity"] = np.nan
        lgbm_note = "  |  LightGBM naive curve: skipped (predicted_return not in dataset)"

    net_end   = float(df["equity"].iloc[-1])
    gross_end = float(df["gross_equity"].iloc[-1])

    print(
        f"[compute_equity_curve]  "
        f"Net end: {net_end:.4f} ({(net_end-1)*100:+.2f} %)  |  "
        f"Gross end: {gross_end:.4f} ({(gross_end-1)*100:+.2f} %)"
        + lgbm_note
    )
    return df


# ===========================================================================
# 8. compute_drawdown
# ===========================================================================
def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the rolling drawdown series from the net equity curve.

    Formula
    -------
        rolling_max_t = max(equity_0 … equity_t)
        drawdown_t    = (equity_t − rolling_max_t) / rolling_max_t

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *equity* (net equity) column.

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
        f"[compute_drawdown]  "
        f"max: {max_dd * 100:.2f} %  |  "
        f"avg: {avg_dd * 100:.2f} %  |  "
        f"underwater rows: {underwater:,} / {len(df):,}"
    )
    return df


# ===========================================================================
# 9. compute_metrics   (v3: Regime-Gated Efficiency metrics added)
# ===========================================================================
def compute_metrics(
    df:         pd.DataFrame,
    bnh_return: float,
) -> dict:
    """Compute the full competition scorecard including Regime-Gated Efficiency.

    Standard Techkriti metrics (unchanged from v2)
    -----------------------------------------------
    gross_profit, gross_loss, net_profit, total_cost_drag,
    buy_hold_return, alpha, cumulative_return, annual_return,
    annual_volatility, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, total_closed_trades, stopped_trades, win_rate,
    profit_factor.

    Regime-Gated Efficiency metrics (v3 NEW)
    -----------------------------------------
    n_entropy_suppressed
        Count of rows where regime_entropy ≥ 0.50 AND signal == "HOLD" AND
        |predicted_return| ≥ SIGNAL_CONFIDENCE_FLOOR.  These are rows that
        survived the base-signal confidence filter but were suppressed by
        the Entropy Kill-Switch (Gate 1).

    n_regime_suppressed
        Count of rows where regime_label ∈ {BULL, RECOVERY} AND
        signal == "HOLD" AND regime_entropy < 0.50 AND
        |predicted_return| ≥ SIGNAL_CONFIDENCE_FLOOR.  These survived the
        entropy gate but were suppressed by the Regime Gate (Gate 2).

    total_alpha_suppressed
        n_entropy_suppressed + n_regime_suppressed.

    equity_saved_high_entropy
        Counterfactual equity preserved by the Entropy Kill-Switch.
        For each entropy-suppressed row, a naive signal return is simulated:
            sim_return = sign(predicted_return) × avg_active_position_size × target_return
        Equity saved = cumulative loss that would have been incurred
                     = −sum(sim_return[sim_return < 0])
        A positive number means the entropy gate prevented that much loss.

    n_hard_cash_lock_days
        Total rows with position=0 due to Hard Cash Lock (not ordinary HOLD).

    pct_active_days
        Fraction of total rows where the strategy held an active position.

    Equity curve comparison terminals (v3 NEW)
    -------------------------------------------
    lgbm_naive_final, gross_equity_final, net_equity_final
        Terminal equity values of the three parallel curves, enabling
        quick attribution without opening equity_curve.csv.

    Parameters
    ----------
    df : pd.DataFrame
    bnh_return : float

    Returns
    -------
    dict
    """
    r      = df["strategy_return"]
    equity = df["equity"]
    n_days = len(r)

    def _safe(v, d=4):
        return round(v, d) if (isinstance(v, float) and not np.isnan(v)) else "N/A"

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

    # ── Trade statistics ─────────────────────────────────────────────────
    total_closed_trades = int((df["signal"] != "HOLD").sum())
    stopped_trades = (
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

    # ── Regime-Gated Efficiency  (v3 NEW) ──────────────────────────────────
    # All three optional columns (predicted_return, regime_entropy,
    # regime_label) should be present in trading_signals.csv from v13.
    # If missing (e.g. running on a v12 signals file), skip gracefully.

    has_pred    = "predicted_return" in df.columns
    has_entropy = "regime_entropy"   in df.columns
    has_label   = "regime_label"     in df.columns

    n_entropy_suppressed  = 0
    n_regime_suppressed   = 0
    equity_saved_entropy  = 0.0
    n_hard_cash_lock_days = 0
    pct_active_days       = 0.0

    if has_pred and has_entropy:
        # Entropy-suppressed: signal=HOLD, entropy≥threshold, model had a real signal
        ent_mask = (
            (df["signal"] == "HOLD")
            & (df["regime_entropy"] >= 0.50)
            & (df["predicted_return"].abs() >= SIGNAL_CONFIDENCE_FLOOR)
        )
        n_entropy_suppressed = int(ent_mask.sum())

        # Equity saved from high-entropy rows: counterfactual loss avoided.
        # Use the mean position size of actually-traded rows as a proxy for
        # what position size would have been taken in those suppressed rows.
        avg_active_pos = float(
            df.loc[df["signal"] != "HOLD", "position_size"].mean()
        )
        if n_entropy_suppressed > 0 and avg_active_pos > 0:
            ent_rows = df[ent_mask].copy()
            sim_dir  = np.sign(ent_rows["predicted_return"].fillna(0))
            sim_ret  = sim_dir * avg_active_pos * ent_rows["target_return"]
            # Equity saved = losses prevented (negative sim_ret values)
            equity_saved_entropy = float(-sim_ret[sim_ret < 0].sum())

    if has_pred and has_entropy and has_label:
        # Regime-suppressed: BULL/RECOVERY rows that survived entropy gate
        reg_mask = (
            (df["signal"] == "HOLD")
            & (df["regime_label"].isin(CASH_LOCKED_REGIMES))
            & (df["regime_entropy"] < 0.50)
            & (df["predicted_return"].abs() >= SIGNAL_CONFIDENCE_FLOOR)
        )
        n_regime_suppressed = int(reg_mask.sum())

    total_alpha_suppressed = n_entropy_suppressed + n_regime_suppressed

    if "is_hard_cash_lock" in df.columns:
        n_hard_cash_lock_days = int(df["is_hard_cash_lock"].sum())

    long_days       = int((df.get("position", pd.Series(0)) == 1).sum())
    pct_active_days = float(long_days / max(n_days, 1))

    # ── Equity curve terminals ───────────────────────────────────────────────
    lgbm_naive_final  = (
        float(df["lgbm_naive_equity"].iloc[-1])
        if "lgbm_naive_equity" in df.columns and df["lgbm_naive_equity"].notna().any()
        else np.nan
    )
    gross_equity_final = (
        float(df["gross_equity"].iloc[-1])
        if "gross_equity" in df.columns else np.nan
    )
    net_equity_final = float(equity.iloc[-1])

    print(
        f"[compute_metrics]  {n_days:,} days  ({date_start} → {date_end})  |  "
        f"Net profit: {net_profit * 100:+.2f} %  |  "
        f"B&H: {bnh_return * 100:+.2f} %  |  "
        f"Alpha: {alpha * 100:+.2f} %  |  "
        f"Suppressed by alpha filter: {total_alpha_suppressed:,}"
    )

    metrics = {
        # ── Period ──────────────────────────────────────────────────────────
        "date_start"              : date_start,
        "date_end"                : date_end,
        "total_days"              : n_days,
        # ── Profit breakdown ────────────────────────────────────────────────
        "gross_profit"            : round(gross_profit,      6),
        "gross_loss"              : round(gross_loss,        6),
        "net_profit"              : round(net_profit,        6),
        "total_cost_drag"         : round(total_cost,        6),
        # ── Benchmark comparison ─────────────────────────────────────────────
        "buy_hold_return"         : round(bnh_return,        6),
        "alpha"                   : round(alpha,             6),
        "cumulative_return"       : round(cumulative_return, 6),
        "annual_return"           : round(annual_return,     6),
        "annual_volatility"       : round(annual_volatility, 6),
        # ── Risk-adjusted metrics ─────────────────────────────────────────────
        "sharpe_ratio"            : _safe(sharpe_ratio),
        "sortino_ratio"           : _safe(sortino_ratio),
        "calmar_ratio"            : _safe(calmar_ratio),
        "max_drawdown"            : round(max_drawdown,      6),
        # ── Trade statistics ─────────────────────────────────────────────────
        "total_closed_trades"     : total_closed_trades,
        "stopped_trades"          : stopped_trades,
        "win_rate"                : round(win_rate,          4),
        "profit_factor"           : _safe(profit_factor),
        # ── Regime-Gated Efficiency (v3 NEW) ─────────────────────────────────
        "n_entropy_suppressed"    : n_entropy_suppressed,
        "n_regime_suppressed"     : n_regime_suppressed,
        "total_alpha_suppressed"  : total_alpha_suppressed,
        "equity_saved_entropy"    : round(equity_saved_entropy, 6),
        "n_hard_cash_lock_days"   : n_hard_cash_lock_days,
        "pct_active_days"         : round(pct_active_days,   4),
        # ── Equity curve terminals (v3 NEW) ──────────────────────────────────
        "lgbm_naive_final"        : _safe(lgbm_naive_final),
        "gross_equity_final"      : _safe(gross_equity_final),
        "net_equity_final"        : _safe(net_equity_final),
    }
    return metrics


# ===========================================================================
# 10. save_results   (v3: equity_curve.csv extended with gross/LightGBM curves)
# ===========================================================================
def save_results(
    df:            pd.DataFrame,
    metrics:       dict,
    equity_path:   Path | str = EQUITY_PATH,
    drawdown_path: Path | str = DRAWDOWN_PATH,
    summary_path:  Path | str = SUMMARY_PATH,
) -> None:
    """Persist equity curve, drawdown series, and competition report to disk.

    equity_curve.csv columns (v3)
    ------------------------------
    date, close, signal, position, regime_label (if present),
    regime_entropy (if present),
    predicted_return (if present) — raw LightGBM model output,
    gross_return                   — pre-fee strategy return,
    transaction_cost               — fee deducted on real-trade rows,
    strategy_return                — net return (post-fee, post-stop-loss),
    lgbm_naive_equity              — cumulative equity: naive LightGBM signal,
    gross_equity                   — cumulative equity: regime-gated, pre-fee,
    equity                         — cumulative equity: regime-gated, post-fee.

    The three equity columns allow direct visual comparison of:
        LightGBM raw signal → regime filter value → fee cost
    without any further computation.

    Parameters
    ----------
    df : pd.DataFrame
    metrics : dict
    equity_path, drawdown_path, summary_path : Path or str
    """
    for path in map(Path, (equity_path, drawdown_path, summary_path)):
        path.parent.mkdir(parents=True, exist_ok=True)

    # ── Equity curve CSV ───────────────────────────────────────────────────
    eq_want = [
        "date", "close", "signal", "position",
        "regime_label", "regime_entropy",        # optional v13 columns
        "predicted_return",                      # LightGBM raw output
        "gross_return",                          # pre-fee strategy return
        "transaction_cost",
        "strategy_return",                       # net return
        "lgbm_naive_equity",                     # naive curve
        "gross_equity",                          # pre-fee compounded curve
        "equity",                                # net compounded curve
    ]
    eq_cols = [c for c in eq_want if c in df.columns]
    df[eq_cols].to_csv(equity_path, index=False)
    print(f"[save_results]  Equity curve saved       → '{equity_path}'  "
          f"(columns: {eq_cols})")

    # ── Drawdown series ────────────────────────────────────────────────────
    dd_want = ["date", "equity", "gross_equity", "rolling_max", "drawdown"]
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
    """Execute the complete v3 regime-gated backtesting pipeline.

    Steps
    -----
    1.  Load trading signals                   (:func:`load_signals`).
    2.  Build position series (Hard Cash Lock) (:func:`compute_positions`).
    3.  Recompute strategy returns             (:func:`compute_strategy_returns`).
    4.  Apply transaction costs (real trades)  (:func:`apply_transaction_costs`).
    5.  Apply stop-loss rule (−5 %)            (:func:`apply_stop_loss`).
    6.  Compute buy-and-hold benchmark         (:func:`compute_buy_and_hold`).
    7.  Build equity curves (net/gross/naive)  (:func:`compute_equity_curve`).
    8.  Compute drawdown series                (:func:`compute_drawdown`).
    9.  Calculate competition metrics          (:func:`compute_metrics`).
    10. Print professional scorecard.
    11. Persist all outputs                    (:func:`save_results`).

    Returns
    -------
    df : pd.DataFrame
        Fully enriched signals DataFrame.
    metrics : dict
        Complete competition scorecard.
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
    if isinstance(value, str):
        return value
    return f"{value * 100:+.{decimals}f} %"


def _fmt_float(value, decimals: int = 4) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}"


def _fmt_int(value) -> str:
    if isinstance(value, str):
        return value
    return f"{int(value):,}"


def _build_report(metrics: dict) -> str:
    """Render the Techkriti competition scorecard as a structured text report.

    v3 addition: Regime-Gated Efficiency section and Equity Attribution section.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w  = 66

    def _row(label, value, width=66):
        return f"  {label:<36} {value}"

    lines = [
        "=" * w,
        "  PROJECT ARI — TECHKRITI COMPETITION SCORECARD".center(w),
        f"  Generated : {ts}".center(w),
        "=" * w,
        "",
        "  ── PERIOD ─────────────────────────────────────────────────────",
        _row("Start date",          metrics["date_start"]),
        _row("End date",            metrics["date_end"]),
        _row("Total trading days",  _fmt_int(metrics["total_days"])),
        "",
        "  ── PROFIT BREAKDOWN ────────────────────────────────────────────",
        _row("Gross Profit",        _fmt_pct(metrics["gross_profit"])),
        _row("Gross Loss",          _fmt_pct(metrics["gross_loss"])),
        _row("Net Profit",          _fmt_pct(metrics["net_profit"])),
        _row("Transaction Cost Drag", _fmt_pct(metrics["total_cost_drag"])),
        "",
        "  ── BENCHMARK COMPARISON ────────────────────────────────────────",
        _row("Buy & Hold Return (BTC)",    _fmt_pct(metrics["buy_hold_return"])),
        _row("Strategy Net Return",        _fmt_pct(metrics["cumulative_return"])),
        _row("Alpha vs B&H",               _fmt_pct(metrics["alpha"])),
        _row("Annualised Return (CAGR)",   _fmt_pct(metrics["annual_return"])),
        _row("Annualised Volatility",      _fmt_pct(metrics["annual_volatility"])),
        "",
        "  ── RISK-ADJUSTED METRICS ───────────────────────────────────────",
        _row("Sharpe Ratio",        _fmt_float(metrics["sharpe_ratio"])),
        _row("Sortino Ratio",       _fmt_float(metrics["sortino_ratio"])),
        _row("Calmar Ratio",        _fmt_float(metrics["calmar_ratio"])),
        _row("Max Drawdown",        _fmt_pct(metrics["max_drawdown"])),
        "",
        "  ── TRADE STATISTICS ────────────────────────────────────────────",
        _row("Total Closed Trades",         _fmt_int(metrics["total_closed_trades"])),
        _row("Stop-Loss Triggered (rows)",  _fmt_int(metrics["stopped_trades"])),
        _row("Days Active (% of period)",
             f"{metrics['pct_active_days'] * 100:.1f} %"),
        _row("Win Rate",                    _fmt_pct(metrics["win_rate"])),
        _row("Profit Factor",               _fmt_float(metrics["profit_factor"])),
        "",
        # ── Regime-Gated Efficiency (v3 NEW) ──────────────────────────────
        "  ── REGIME-GATED EFFICIENCY ────────────────────────────────────",
        "  (Alpha Filter = Entropy Kill-Switch + Regime Gate)",
        "",
        _row("Total Signals Suppressed (Alpha Filter)",
             _fmt_int(metrics["total_alpha_suppressed"])),
        _row("  — by Entropy Kill-Switch  (entropy ≥ 0.50)",
             _fmt_int(metrics["n_entropy_suppressed"])),
        _row("  — by Regime Gate          (BULL / RECOVERY)",
             _fmt_int(metrics["n_regime_suppressed"])),
        "",
        _row("Equity Saved from High-Entropy Periods",
             _fmt_pct(metrics["equity_saved_entropy"])),
        _row("Hard Cash Lock Days (regime-forced)",
             _fmt_int(metrics["n_hard_cash_lock_days"])),
        "",
        # ── Equity attribution (v3 NEW) ───────────────────────────────────
        "  ── EQUITY ATTRIBUTION (terminal equity, start = 1.00) ─────────",
        _row("LightGBM Naive Equity  (no filter, no fee)",
             _fmt_float(metrics["lgbm_naive_final"])),
        _row("Gross Equity  (regime-gated, pre-fee)",
             _fmt_float(metrics["gross_equity_final"])),
        _row("Net Equity    (regime-gated, post-fee)",
             _fmt_float(metrics["net_equity_final"])),
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
    """CLI entry point: run the v3 regime-gated backtesting pipeline."""
    print("=" * 66)
    print("  Backtester v3 — Regime-Gated  |  Project ARI")
    print("=" * 66)
    try:
        df, metrics = backtest_pipeline()
        print(
            f"Backtest complete.\n"
            f"  Net profit      : {metrics['net_profit'] * 100:+.2f} %\n"
            f"  B&H             : {metrics['buy_hold_return'] * 100:+.2f} %\n"
            f"  Alpha           : {metrics['alpha'] * 100:+.2f} %\n"
            f"  Sharpe          : {metrics['sharpe_ratio']}\n"
            f"  Suppressed      : {metrics['total_alpha_suppressed']:,} signals\n"
            f"  Equity saved    : {metrics['equity_saved_entropy'] * 100:+.2f} %\n"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
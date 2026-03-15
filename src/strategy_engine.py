"""
strategy_engine.py
------------------
Adaptive rule-based trading strategy engine for project_ari.

v3 improvements — percentile-based signal generation
-----------------------------------------------------
Signal generation now uses the *percentile rank* of predicted_return
within the full test-set distribution rather than fixed return thresholds.
This makes the strategy robust to distribution shift across regimes:
regardless of the absolute level of predictions, the top/bottom
percentile buckets always fire a signal.

Per-regime percentile thresholds
---------------------------------
  Bull      : BUY if pct >= 60  |  SELL if pct <= 10
  Recovery  : BUY if pct >= 70  |  SELL if pct <= 20
  Sideways  : BUY if pct >= 80  |  SELL if pct <= 20
  Crash     : BUY if pct >= 90  |  SELL if pct <= 40
  Middle percentiles → HOLD

Preserved from v2
-----------------
• Minimum holding period of 3 days
• Entropy filter          (high-uncertainty → HOLD)
• Inverse-volatility position sizing
• Regime scale factors    (Bull 1.0 / Recovery 0.5 / Sideways 0.25 / Crash 0.0)
• All logging and diagnostics
• File outputs (trading_signals.csv)

Pipeline:
    1. Load predictions dataset.
    2. Compute global percentile rank of predicted_return.
    3. Generate percentile-based regime-dependent signals.
    4. Apply minimum holding period filter.
    5. Apply entropy filter.
    6. Apply regime scale / allowed-signal filter.
    7. Size positions via inverse-volatility scaling.
    8. Compute strategy returns.
    9. Print diagnostics and save output.

Project structure expected:
    project_ari/
    ├── data/
    │   └── processed/
    │       ├── predictions.csv       ← input
    │       └── trading_signals.csv   ← output
    └── src/
        └── strategy_engine.py

Usage:
    Run directly:   python src/strategy_engine.py
    Import:         from src.strategy_engine import strategy_pipeline
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "predictions.csv"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "trading_signals.csv"

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

# ── v4: Per-regime percentile thresholds ───────────────────────────────────
REGIME_PERCENTILE_THRESHOLDS: dict[str, dict[str, float]] = {
    "Bull"     : {"buy_pct": 55.0, "sell_pct":  2.0},
    "Recovery" : {"buy_pct": 65.0, "sell_pct": 10.0},
    "Sideways" : {"buy_pct": 80.0, "sell_pct": 20.0},
    "Crash"    : {"buy_pct": 95.0, "sell_pct": 40.0},
}
DEFAULT_PERCENTILE_THRESHOLD: dict[str, float] = {"buy_pct": 65.0, "sell_pct": 35.0}

CONFIDENCE_THRESHOLD: float = 0.003
MIN_HOLDING_DAYS: int = 5
ENTROPY_THRESHOLD: float = 0.5
TREND_MA_WINDOW: int = 50

RISK_TARGET:   float = 0.02
POSITION_MIN:  float = 0.0
POSITION_MAX:  float = 1.0

TARGET_VOL:        float = 0.02
VOL_ADJUST_MIN:    float = 0.3
VOL_ADJUST_MAX:    float = 1.5

REGIME_RULES: dict[str, dict] = {
    "Bull"     : {"allowed": {"BUY", "SELL"}, "scale": 1.00},
    "Recovery" : {"allowed": {"BUY"},          "scale": 0.50},
    "Sideways" : {"allowed": {"BUY", "SELL"},  "scale": 0.25},
    "Crash"    : {"allowed": set(),            "scale": 0.00},
}
DEFAULT_REGIME_RULE: dict = {"allowed": {"BUY", "SELL"}, "scale": 1.00}

# ── v11: Trend hold ─────────────────────────────────────────────────────────
# 20-day MA window used to detect short-term upward momentum for the
# trend-hold filter.  Separate from the 50-day MA used by the trend filter.
TREND_HOLD_MA_WINDOW: int = 20   # rolling window for MA20 computation
TREND_HOLD_SLOPE_LOOKBACK: int = 3   # days used to assess whether MA20 is rising


# ===========================================================================
# 1. load_predictions
# ===========================================================================
def load_predictions(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Predictions dataset not found: {filepath}\n"
            "Run lightgbm_model.py first to generate predictions.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    required = {
        "date", "close", "predicted_return", "market_regime",
        "regime_entropy", "volatility_7", "target_return",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Predictions dataset is missing required column(s): {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[df["predicted_return"].notna()].copy()
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[load_predictions]  Loaded {len(df):,} test-set rows from '{filepath}'.")
    return df


# ===========================================================================
# 2. compute_percentile_rank
# ===========================================================================
def compute_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_percentile"] = (
        df["predicted_return"]
        .rank(method="average", pct=True) * 100
    )

    p10 = df["pred_percentile"].quantile(0.10)
    p50 = df["pred_percentile"].quantile(0.50)
    p90 = df["pred_percentile"].quantile(0.90)
    print(
        f"[compute_percentile_rank]  Percentile distribution — "
        f"p10: {p10:.1f}  p50: {p50:.1f}  p90: {p90:.1f}  "
        f"(n={len(df):,})"
    )
    return df


# ===========================================================================
# 3. generate_base_signal
# ===========================================================================
def generate_base_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate base trading signals from predicted returns using
    percentile ranking and a confidence filter.

    BUY  → top 20% predictions
    SELL → bottom 20% predictions
    HOLD → middle 60%

    Any prediction with |return| < CONFIDENCE_THRESHOLD is forced to HOLD.
    """

    if "predicted_return" not in df.columns:
        raise ValueError("Column 'predicted_return' not found in DataFrame.")

    df = df.copy()

    # Step 1 — Percentile ranking of predictions
    df["pred_percentile"] = df["predicted_return"].rank(method="average", pct=True)

    conditions = [
        df["pred_percentile"] >= 0.80,
        df["pred_percentile"] <= 0.20
    ]

    choices = ["BUY", "SELL"]

    df["signal"] = np.select(conditions, choices, default="HOLD")

    # Step 2 — Confidence filter
    CONFIDENCE_THRESHOLD = 0.002

    weak_mask = df["predicted_return"].abs() < CONFIDENCE_THRESHOLD
    df.loc[weak_mask, "signal"] = "HOLD"

    # Diagnostics
    print("\n[generate_base_signal]")
    print("Signal distribution before filters:")
    print(df["signal"].value_counts())

    suppressed = weak_mask.sum()
    print(f"Confidence filter suppressed {suppressed} signals.")

    print("Final signal distribution:")
    print(df["signal"].value_counts())

    return df


# ===========================================================================
# 4. apply_confidence_filter
# ===========================================================================
def apply_confidence_filter(
    df: pd.DataFrame,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    df = df.copy()
    low_conviction_mask = df["predicted_return"].abs() < confidence_threshold
    n_overridden        = (low_conviction_mask & (df["signal"] != "HOLD")).sum()

    df.loc[low_conviction_mask, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[apply_confidence_filter]  Threshold: |pred| >= {confidence_threshold}  |  "
        f"Signals suppressed: {n_overridden:,}  |  "
        f"Post-confidence -- BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 5. apply_signal_persistence
# ===========================================================================
def apply_signal_persistence(
    df: pd.DataFrame,
    required_repeats: int = 2,
) -> pd.DataFrame:
    df = df.copy()
    raw_signals = df["signal"].tolist()

    confirmed      = "HOLD"
    candidate      = "HOLD"
    candidate_run  = 0

    output    = []
    pending   = []

    for raw in raw_signals:
        if raw == "HOLD":
            output.append(confirmed)
            pending.append("")
            continue

        if raw == confirmed:
            candidate     = confirmed
            candidate_run = 0
            output.append(confirmed)
            pending.append("")

        elif raw == candidate:
            candidate_run += 1
            if candidate_run >= required_repeats:
                confirmed     = candidate
                candidate_run = 0
                output.append(confirmed)
                pending.append("")
            else:
                output.append(confirmed if confirmed != "HOLD" else raw)
                pending.append(candidate)

        else:
            candidate     = raw
            candidate_run = 1
            output.append(confirmed if confirmed != "HOLD" else raw)
            pending.append(candidate)

    df["signal"]              = output
    df["persistence_pending"] = pending

    raw_counts = pd.Series(raw_signals).value_counts()
    out_counts = pd.Series(output).value_counts()
    n_flips_raw = sum(
        1 for i in range(1, len(raw_signals))
        if raw_signals[i] != raw_signals[i - 1]
        and raw_signals[i] != "HOLD"
        and raw_signals[i - 1] != "HOLD"
    )
    n_flips_out = sum(
        1 for i in range(1, len(output))
        if output[i] != output[i - 1]
        and output[i] != "HOLD"
        and output[i - 1] != "HOLD"
    )
    print(
        f"[apply_signal_persistence]  Required repeats: {required_repeats}  |  "
        f"Direction flips — before: {n_flips_raw:,}  after: {n_flips_out:,}  "
        f"(suppressed: {n_flips_raw - n_flips_out:,})"
    )
    print(
        f"[apply_signal_persistence]  Post-persistence — "
        f"BUY: {out_counts.get('BUY', 0):,}  "
        f"SELL: {out_counts.get('SELL', 0):,}  "
        f"HOLD: {out_counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 6. apply_holding_period
# ===========================================================================
def apply_holding_period(
    df: pd.DataFrame,
    min_days: int = MIN_HOLDING_DAYS,
) -> pd.DataFrame:
    df = df.copy()
    signals  = df["signal"].tolist()
    filtered = []
    hold_days_col = []

    days_held      = 0
    active_signal  = "HOLD"

    for raw_sig in signals:
        if active_signal != "HOLD":
            if days_held < min_days:
                filtered.append(active_signal)
                hold_days_col.append(days_held)
                days_held += 1
            else:
                if raw_sig != "HOLD":
                    active_signal = raw_sig
                    days_held     = 1
                else:
                    active_signal = "HOLD"
                    days_held     = 0
                filtered.append(active_signal)
                hold_days_col.append(0)
        else:
            if raw_sig != "HOLD":
                active_signal = raw_sig
                days_held     = 1
            filtered.append(active_signal)
            hold_days_col.append(0)

    df["signal"]   = filtered
    df["hold_day"] = hold_days_col

    counts      = pd.Series(filtered).value_counts()
    n_suppressed = sum(
        1 for orig, filt in zip(signals, filtered)
        if orig != filt
    )
    print(
        f"[apply_holding_period]  Min hold: {min_days} days  |  "
        f"Signals suppressed: {n_suppressed:,}  |  "
        f"Post-hold — BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 7. compute_entropy_confidence
# ===========================================================================
def compute_entropy_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise regime entropy into a per-row confidence score [0, 1]."""
    df = df.copy()

    e     = df["regime_entropy"]
    e_min = e.min()
    e_max = e.max()
    e_range = e_max - e_min

    if e_range > 0:
        normalized_entropy = (e - e_min) / e_range
    else:
        normalized_entropy = pd.Series(0.0, index=df.index)

    df["confidence"] = 1.0 - normalized_entropy

    print(
        f"[entropy_diagnostics]  avg_entropy: {float(e.mean()):.4f}  |  "
        f"min_entropy: {float(e_min):.4f}  |  "
        f"max_entropy: {float(e_max):.4f}  |  "
        f"avg_confidence: {float(df['confidence'].mean()):.4f}"
    )
    return df


# ===========================================================================
# 8. apply_trend_filter
# ===========================================================================
def apply_trend_filter(
    df: pd.DataFrame,
    ma_window: int = TREND_MA_WINDOW,
) -> pd.DataFrame:
    """Symmetric trend filter: confirm signals against the 50-day MA."""
    df = df.copy()

    df["trend_ma50"] = df["close"].rolling(window=ma_window, min_periods=ma_window).mean()

    ma        = df["trend_ma50"]
    ma_valid  = ma.notna()
    above_ma  = df["close"] > ma
    below_ma  = df["close"] < ma

    buy_confirmed  = (df["signal"] == "BUY")  & above_ma & ma_valid
    sell_confirmed = (df["signal"] == "SELL") & below_ma & ma_valid
    disagreement   = ma_valid & ~buy_confirmed & ~sell_confirmed & (df["signal"] != "HOLD")

    n_buy_kept   = int(buy_confirmed.sum())
    n_sell_kept  = int(sell_confirmed.sum())
    n_suppressed = int(disagreement.sum())
    n_ma_na      = int((~ma_valid).sum())

    df.loc[disagreement, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[apply_trend_filter]  MA window: {ma_window} days  |  "
        f"Warm-up rows (unfiltered): {n_ma_na:,}  |  "
        f"BUY confirmed (close>MA): {n_buy_kept:,}  |  "
        f"SELL confirmed (close<MA): {n_sell_kept:,}  |  "
        f"Suppressed (trend disagreement): {n_suppressed:,}"
    )
    print(
        f"[apply_trend_filter]  Post-filter — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Bull Regime Protection Filter ---
# ===========================================================================
def apply_bull_regime_protection(
    df: pd.DataFrame,
    regime_col: str = "market_regime",
) -> pd.DataFrame:
    """Convert SELL signals to HOLD during bull market regimes."""
    df = df.copy()

    if regime_col not in df.columns:
        fallback = "regime_label"
        if fallback in df.columns:
            regime_col = fallback
        else:
            print(
                f"[bull_regime_protection]  WARNING: column '{regime_col}' not found — "
                "filter skipped, signals unchanged."
            )
            return df

    raw = df[regime_col]

    def _is_bull(val) -> bool:
        if isinstance(val, str):
            return val.strip().lower() in {"bull", "bull market", "bullish"}
        if isinstance(val, (int, float)) and not pd.isna(val):
            return int(val) == 0
        return False

    bull_mask    = raw.apply(_is_bull)
    sell_on_bull = bull_mask & (df["signal"] == "SELL")
    n_converted  = int(sell_on_bull.sum())
    n_bull_rows  = int(bull_mask.sum())

    df.loc[sell_on_bull, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[bull_regime_protection]  Bull rows: {n_bull_rows:,}  |  "
        f"SELL → HOLD conversions: {n_converted:,}  |  "
        f"Regime column: '{regime_col}'"
    )
    print(
        f"[bull_regime_protection]  Post-filter — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Regime Adaptive Strategy ---
# ===========================================================================
def apply_regime_adaptive_strategy(
    df: pd.DataFrame,
    regime_col: str = "market_regime",
) -> pd.DataFrame:
    """Apply regime-specific signal override rules."""
    df = df.copy()

    if regime_col not in df.columns:
        fallback = "regime_label"
        if fallback in df.columns:
            regime_col = fallback
        else:
            print(
                f"[regime_adaptive]  WARNING: column '{regime_col}' not found — "
                "adaptive rules skipped, signals unchanged."
            )
            return df

    ma_col = "ma50"
    if ma_col not in df.columns:
        print(
            "[regime_adaptive]  'ma50' column not found — "
            "computing 50-day MA from 'close' on the fly."
        )
        df[ma_col] = df["close"].rolling(window=50, min_periods=50).mean()

    _BULL_STRINGS      = {"bull", "bullish", "bull market"}
    _RECOVERY_STRINGS  = {"recovery"}
    _SIDEWAYS_STRINGS  = {"sideways", "neutral", "ranging"}
    _STRESS_STRINGS    = {"stress", "crash", "bear", "bearish"}

    def _to_bucket(val) -> str:
        if isinstance(val, str):
            v = val.strip().lower()
            if v in _BULL_STRINGS:      return "BULL"
            if v in _RECOVERY_STRINGS:  return "RECOVERY"
            if v in _SIDEWAYS_STRINGS:  return "SIDEWAYS"
            if v in _STRESS_STRINGS:    return "STRESS"
        elif isinstance(val, (int, float)) and not pd.isna(val):
            code = int(val)
            if code == 0: return "BULL"
            if code == 1: return "RECOVERY"
            if code == 2: return "SIDEWAYS"
            if code == 3: return "STRESS"
        return "SIDEWAYS"

    df["regime_bucket"] = df[regime_col].apply(_to_bucket)

    signals      = df["signal"].tolist()
    close_vals   = df["close"].tolist()
    ma_vals      = df[ma_col].tolist()
    buckets      = df["regime_bucket"].tolist()
    new_signals  = []

    n_forced_buy   = 0
    n_blocked_sell = 0
    n_blocked_buy  = 0
    n_sideways     = 0

    for sig, close, ma, bucket in zip(signals, close_vals, ma_vals, buckets):
        above_ma = (not pd.isna(ma)) and (close > ma)

        if bucket in ("BULL", "RECOVERY"):
            if sig == "SELL":
                new_signals.append("HOLD")
                n_blocked_sell += 1
            elif above_ma:
                new_signals.append("BUY")
                if sig != "BUY":
                    n_forced_buy += 1
            else:
                new_signals.append("HOLD")

        elif bucket == "SIDEWAYS":
            new_signals.append(sig)
            n_sideways += 1

        else:   # STRESS
            if sig == "BUY" and not above_ma:
                new_signals.append("HOLD")
                n_blocked_buy += 1
            else:
                new_signals.append(sig)

    df["signal"] = new_signals

    counts = pd.Series(new_signals).value_counts()
    bucket_counts = df["regime_bucket"].value_counts().to_dict()

    print(
        f"[regime_adaptive]  Regime rows — "
        + "  ".join(f"{b}: {bucket_counts.get(b, 0):,}"
                    for b in ("BULL", "RECOVERY", "SIDEWAYS", "STRESS"))
    )
    print(
        f"[regime_adaptive]  Overrides — "
        f"forced BUY (bull/recovery above MA): {n_forced_buy:,}  |  "
        f"SELL blocked (bull/recovery): {n_blocked_sell:,}  |  "
        f"BUY blocked (stress below MA): {n_blocked_buy:,}  |  "
        f"sideways unchanged: {n_sideways:,}"
    )
    print(
        f"[regime_adaptive]  Post-adaptive — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Trend Dominance Override ---
# ===========================================================================
def apply_trend_dominance_override(
    df: pd.DataFrame,
    ma_window: int = TREND_MA_WINDOW,
    slope_lookback: int = 5,
    regime_col: str = "market_regime",
) -> pd.DataFrame:
    """Force BUY during confirmed uptrends, regardless of upstream signals."""
    df = df.copy()

    ma_col = "ma50"
    if ma_col not in df.columns:
        if "trend_ma50" in df.columns:
            df[ma_col] = df["trend_ma50"]
        else:
            df[ma_col] = (
                df["close"]
                .rolling(window=ma_window, min_periods=ma_window)
                .mean()
            )

    ma_series  = df[ma_col]
    ma_shifted = ma_series.shift(slope_lookback)

    trend_up = (
        (df["close"] > ma_series)
        & (ma_series > ma_shifted)
        & ma_series.notna()
        & ma_shifted.notna()
    )

    stress_mask = pd.Series(False, index=df.index)
    resolved_col = None
    if regime_col and regime_col in df.columns:
        resolved_col = regime_col
    elif "regime_label" in df.columns:
        resolved_col = "regime_label"

    if resolved_col is not None:
        _STRESS_STRINGS = {"stress", "crash", "bear", "bearish"}
        def _is_stress(val) -> bool:
            if isinstance(val, str):
                return val.strip().lower() in _STRESS_STRINGS
            if isinstance(val, (int, float)) and not pd.isna(val):
                return int(val) == 3
            return False
        stress_mask = df[resolved_col].apply(_is_stress)
    else:
        print(
            "[trend_override]  Regime column not found — "
            "stress guard disabled; override applies to all uptrend rows."
        )

    override_mask = trend_up & ~stress_mask
    n_uptrend    = int(trend_up.sum())
    n_stress     = int(stress_mask.sum())
    n_overridden = int((override_mask & (df["signal"] != "BUY")).sum())

    df.loc[override_mask, "signal"] = "BUY"
    df["trend_dominant"] = override_mask

    counts = df["signal"].value_counts()
    print(
        f"[trend_override]  Rows in uptrend (close>MA & MA rising): {n_uptrend:,}  |  "
        f"Stress rows excluded: {n_stress:,}"
    )
    print(
        f"[trend_override]  Signals overridden → BUY: {n_overridden:,}  |  "
        f"Post-override — BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Trend Hold Filter ---
# Prevents premature exits during strong upward momentum by converting
# SELL signals to HOLD when the market is in a short-term uptrend.
#
# Uptrend condition (both must hold):
#   close > MA20              — price is above the 20-day moving average
#   MA20  > MA20.shift(3)     — MA20 itself is rising over the past 3 days
#
# When trend_hold is TRUE and signal == SELL:
#   signal → HOLD  (stay long; do not exit the trend early)
#
# Only SELL signals are affected.  BUY and HOLD signals pass through
# unchanged.  Rows where MA20 has not yet warmed up (NaN) are left
# unfiltered to avoid suppressing signals at the start of the test set.
# ===========================================================================
def apply_trend_hold(
    df: pd.DataFrame,
    ma_window: int = TREND_HOLD_MA_WINDOW,
    slope_lookback: int = TREND_HOLD_SLOPE_LOOKBACK,
) -> pd.DataFrame:
    """Block premature SELL exits during confirmed short-term uptrends.

    Computes a 20-day simple moving average of *close* and checks whether
    both the price is above the MA and the MA is rising.  When both are
    true the strategy is in a strong momentum state and any SELL signal
    is converted to HOLD to avoid exiting the trend prematurely.

    Formula
    -------
        ma20        = close.rolling(20).mean()
        trend_hold  = (close > ma20) AND (ma20 > ma20.shift(3))

        if trend_hold AND signal == SELL:
            signal = HOLD

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *signal* and *close* columns, sorted chronologically.
    ma_window : int
        Rolling window for the short-term MA.  Default 20.
    slope_lookback : int
        Lookback used to assess whether the MA is rising.  Default 3.

    Returns
    -------
    pd.DataFrame
        DataFrame with premature SELL exits converted to HOLD during
        uptrend conditions, plus a *ma20* helper column (dropped before
        final save).
    """
    df = df.copy()

    # ── Compute 20-day MA ────────────────────────────────────────────────────
    df["ma20"] = df["close"].rolling(window=ma_window, min_periods=ma_window).mean()

    ma20         = df["ma20"]
    ma20_shifted = ma20.shift(slope_lookback)

    # ── trend_hold condition: close above MA20 AND MA20 slope positive ───────
    trend_hold_mask = (
        (df["close"] > ma20)
        & (ma20 > ma20_shifted)
        & ma20.notna()
        & ma20_shifted.notna()
    )

    # ── Apply: SELL during trend_hold → HOLD ────────────────────────────────
    sell_in_trend  = trend_hold_mask & (df["signal"] == "SELL")
    n_in_trend     = int(trend_hold_mask.sum())
    n_sell_blocked = int(sell_in_trend.sum())

    df.loc[sell_in_trend, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[trend_hold]  rows_in_trend_hold: {n_in_trend:,}  |  "
        f"sell_signals_blocked: {n_sell_blocked:,}"
    )
    print(
        f"[trend_hold]  Post-trend-hold — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Strong Trend Persistence ---
# Prevents the strategy from exiting confirmed bull trends by blocking
# SELL signals when both MA20 is above MA50 (short-term trend leading
# the long-term trend) AND price is above MA50 (price above key support).
#
# strong_trend = (close > MA50) AND (MA20 > MA50)
#
# If strong_trend is TRUE and signal == SELL:
#   signal → HOLD
#
# Only SELL signals are affected.  BUY and HOLD pass through unchanged.
# Rows where either MA has not yet warmed up are left unfiltered.
# Runs after apply_trend_hold so both momentum layers are applied before
# regime scaling.
# ===========================================================================
def apply_strong_trend_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Block SELL signals during confirmed strong bull trends.

    A strong trend is defined by two simultaneously true conditions:
      1. close > MA50  — price is above the 50-day moving average
      2. MA20  > MA50  — the short-term MA has crossed above the long-term MA
                         (classic golden-cross signal of trend strength)

    When both hold, the strategy is inside a structural uptrend and any
    SELL signal is converted to HOLD to prevent premature exits.

    The MA20 column is reused from ``apply_trend_hold`` if already present
    (``df["ma20"]``); otherwise it is computed here.  Similarly, MA50 is
    reused from ``apply_trend_filter`` (``df["trend_ma50"]``) or
    ``apply_regime_adaptive_strategy`` (``df["ma50"]``) if present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *signal* and *close* columns, sorted chronologically.

    Returns
    -------
    pd.DataFrame
        DataFrame with SELL signals inside strong trends converted to HOLD.
        No new columns are added; existing helper columns are reused.
    """
    df = df.copy()

    # ── Resolve or compute MA50 ──────────────────────────────────────────────
    if "ma50" in df.columns:
        ma50 = df["ma50"]
    elif "trend_ma50" in df.columns:
        ma50 = df["trend_ma50"]
    else:
        ma50 = df["close"].rolling(window=50, min_periods=50).mean()
        df["ma50"] = ma50

    # ── Resolve or compute MA20 ──────────────────────────────────────────────
    if "ma20" in df.columns:
        ma20 = df["ma20"]
    else:
        ma20 = df["close"].rolling(window=20, min_periods=20).mean()
        df["ma20"] = ma20

    # ── strong_trend mask: both MAs must be valid (no NaN) ───────────────────
    strong_trend_mask = (
        (df["close"] > ma50)
        & (ma20 > ma50)
        & ma50.notna()
        & ma20.notna()
    )

    # ── Apply: SELL during strong_trend → HOLD ───────────────────────────────
    sell_in_strong_trend = strong_trend_mask & (df["signal"] == "SELL")
    n_in_strong_trend    = int(strong_trend_mask.sum())
    n_sell_blocked       = int(sell_in_strong_trend.sum())

    df.loc[sell_in_strong_trend, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[strong_trend]  rows_in_strong_trend: {n_in_strong_trend:,}  |  "
        f"sell_signals_blocked: {n_sell_blocked:,}"
    )
    print(
        f"[strong_trend]  Post-strong-trend — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# --- Position Carry Logic ---
# ===========================================================================
def apply_position_carry(df: pd.DataFrame) -> pd.DataFrame:
    """Pass all signals through unchanged; track position state for diagnostics."""
    df = df.copy()

    state     = "FLAT"
    positions = []

    for sig in df["signal"]:
        if sig == "BUY":
            state = "LONG"
        elif sig == "SELL":
            state = "FLAT"
        positions.append(state)

    df["carry_position"] = positions

    counts = df["signal"].value_counts()
    print(
        f"[position_carry]  buy_signals: {counts.get('BUY', 0):,}  |  "
        f"sell_signals: {counts.get('SELL', 0):,}  |  "
        f"hold_signals: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 7. apply_regime_filter
# ===========================================================================
def apply_regime_filter(
    df: pd.DataFrame,
    rules: dict[str, dict] = REGIME_RULES,
) -> pd.DataFrame:
    df = df.copy()
    regime_scale = np.ones(len(df))

    for idx, row in df.iterrows():
        rule = rules.get(row["market_regime"], DEFAULT_REGIME_RULE)
        regime_scale[idx]  = rule["scale"]

        if row["signal"] not in rule["allowed"]:
            df.at[idx, "signal"] = "HOLD"

    df["regime_scale"] = regime_scale

    counts = df["signal"].value_counts()
    print(
        f"[apply_regime_filter]   Post-regime signals — "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 8. compute_position_sizes  (v13: confidence-floor position scaling)
# ===========================================================================
def compute_position_sizes(
    df: pd.DataFrame,
    risk_target:    float = RISK_TARGET,
    target_vol:     float = TARGET_VOL,
    vol_adjust_min: float = VOL_ADJUST_MIN,
    vol_adjust_max: float = VOL_ADJUST_MAX,
    pos_min:        float = POSITION_MIN,
    pos_max:        float = POSITION_MAX,
) -> pd.DataFrame:
    """Size positions using inverse-volatility scaling with a confidence floor.

    Sizing formula (two-stage)
    --------------------------
    Stage 1 — Base size (inverse short-term volatility):
        base_size = risk_target / volatility_7
        scaled    = base_size * regime_scale

    Stage 2 — Confidence-floor position scalar:
        position_scalar = 0.6 + (0.8 * confidence)   # range [0.6, 1.4]
        position_scalar = min(position_scalar, 1.5)   # hard cap at 1.5

        final_size      = scaled * vol_adjustment * position_scalar
        position_size   = clip(final_size, pos_min, pos_max)

    The confidence-floor formula replaces the previous direct multiplication
    (``base_size * confidence``) which drove average position sizes to ~0.26.
    The floor of 0.6 ensures meaningful capital participation even when
    entropy is at its maximum; the ceiling of 1.5 prevents over-leverage
    in very low-entropy (high-confidence) rows.

    HOLD signals always receive position_size = 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *signal*, *volatility_7*, *volatility_30*,
        *regime_scale*, and optionally *confidence* columns.
    risk_target : float
        Numerator for inverse short-term vol sizing.  Default 0.02.
    target_vol : float
        Desired daily portfolio volatility for vol-targeting.  Default 0.02.
    vol_adjust_min, vol_adjust_max : float
        Clamp bounds on the 30-day vol adjustment.  Default 0.3 / 1.5.
    pos_min, pos_max : float
        Final position size clipping bounds.  Default 0.0 / 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with *position_size* column added.
    """
    df = df.copy()

    # ── Confidence: default to 1.0 (full confidence) if column absent ────────
    confidence = (
        df["confidence"]
        if "confidence" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    # ── Stage 1: inverse short-term vol base size ────────────────────────────
    vol7 = df["volatility_7"].replace(0, np.nan).fillna(df["volatility_7"].median())
    base_size   = risk_target / vol7
    scaled_size = base_size * df["regime_scale"]

    # ── Stage 2: 30-day volatility targeting adjustment ──────────────────────
    vol30 = df["volatility_30"].replace(0, np.nan).fillna(df["volatility_30"].median())
    vol_adjustment = (target_vol / vol30).clip(lower=vol_adjust_min, upper=vol_adjust_max)

    # ── Stage 3: confidence-floor position scalar ────────────────────────────
    # Minimum scalar = 0.6  (confidence = 0 → 0.6 + 0.0 = 0.6)
    # Maximum scalar = 1.4  (confidence = 1 → 0.6 + 0.8 = 1.4, capped at 1.5)
    position_scalar = (0.6 + (0.8 * confidence)).clip(upper=1.5)

    # ── Combine all factors ───────────────────────────────────────────────────
    final_size = scaled_size * vol_adjustment * position_scalar
    clipped    = final_size.clip(lower=pos_min, upper=pos_max)

    # Zero out HOLD rows explicitly
    df["position_size"] = np.where(df["signal"] == "HOLD", 0.0, clipped)

    # ── Existing diagnostics ─────────────────────────────────────────────────
    non_hold   = df.loc[df["signal"] != "HOLD", "position_size"]
    active_adj = vol_adjustment[df["signal"] != "HOLD"]
    print(
        f"[compute_position_sizes]  Avg position size (non-HOLD): "
        f"{non_hold.mean():.4f}  |  "
        f"Overall avg: {df['position_size'].mean():.4f}"
    )
    print(
        f"[compute_position_sizes]  Vol adjustment (non-HOLD) — "
        f"avg: {active_adj.mean():.4f}  |  "
        f"min: {active_adj.min():.4f}  |  "
        f"max: {active_adj.max():.4f}  "
        f"[clamp: {vol_adjust_min}–{vol_adjust_max}]"
    )

    # ── New position-scaling diagnostics ─────────────────────────────────────
    active_scalar = position_scalar[df["signal"] != "HOLD"]
    print(
        f"[position_scaling]  avg_position_size: {non_hold.mean():.4f}  |  "
        f"min_position_size: {non_hold.min():.4f}  |  "
        f"max_position_size: {non_hold.max():.4f}"
    )
    print(
        f"[position_scaling]  confidence scalar (non-HOLD) — "
        f"avg: {active_scalar.mean():.4f}  |  "
        f"min: {active_scalar.min():.4f}  |  "
        f"max: {active_scalar.max():.4f}  "
        f"[formula: 0.6 + (0.8 * confidence), cap 1.5]"
    )
    return df


# ===========================================================================
# 9. compute_strategy_returns
# ===========================================================================
def compute_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    direction             = df["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(float)
    df["strategy_return"] = direction * df["position_size"] * df["target_return"]

    cumulative = df["strategy_return"].sum()
    print(
        f"[compute_strategy_returns]  "
        f"Cumulative strategy return: {cumulative:+.4f}  "
        f"({cumulative * 100:+.2f} %)"
    )
    return df


# ===========================================================================
# 17. strategy_pipeline  (v12: strong trend persistence added)
# ===========================================================================
def strategy_pipeline(
    input_path:  Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the complete v12 strategy engine pipeline end-to-end.

    Steps
    -----
    1.  Load predictions dataset.
    2.  Compute entropy confidence scores.
    3.  Generate base signals from prediction sign.
    4.  Apply trend filter (50-day MA; confirm signal direction).
    5.  Apply bull regime protection (SELL → HOLD in bull regimes).
    6.  Apply regime adaptive strategy (regime-specific signal overrides).
    7.  Apply trend dominance override (force BUY during confirmed uptrends).
    8.  Apply trend hold filter (block SELL during MA20 momentum).
    9.  Apply strong trend persistence (block SELL when MA20 > MA50 & close > MA50).
    10. Apply regime allowed-signal filter + attach regime_scale.
    11. Compute inverse-volatility + vol-targeted position sizes.
    12. Apply position carry (track state; pass signals through).
    13. Compute per-row strategy returns.
    14. Print diagnostics and save output.

    Parameters
    ----------
    input_path  : Path or str
    output_path : Path or str

    Returns
    -------
    pd.DataFrame
        Signals dataset with all strategy columns attached.
    """
    df = load_predictions(input_path)
    df = compute_entropy_confidence(df)
    df = generate_base_signal(df)
    df = apply_trend_filter(df)
    df = apply_bull_regime_protection(df)
    df = apply_regime_adaptive_strategy(df)
    df = apply_trend_dominance_override(df)
    df = apply_trend_hold(df)
    df = apply_strong_trend_persistence(df)             # ← v12: new step
    df = apply_regime_filter(df)
    df = compute_position_sizes(df)
    df = apply_position_carry(df)
    df.loc[df["signal"] == "HOLD", "position_size"] = 0.0
    df = compute_strategy_returns(df)

    df.drop(
        columns=["regime_scale", "hold_day", "pred_percentile",
                 "persistence_pending", "trend_ma50", "position_state",
                 "regime_bucket", "ma50", "ma20", "trend_dominant",
                 "carry_position", "confidence"],
        inplace=True, errors="ignore",
    )

    _print_diagnostics(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[strategy_pipeline]  Saved signals dataset -> '{output_path}'  "
          f"({len(df):,} rows).")
    return df


# ===========================================================================
# Private helpers
# ===========================================================================
def _log_percentile_threshold_summary(
    df: pd.DataFrame,
    regime_thresholds: dict[str, dict[str, float]],
) -> None:
    print("[generate_base_signal]  Percentile threshold config:")
    regime_counts = df.groupby("regime_label")["signal"].value_counts().unstack(fill_value=0)
    for regime, thresh in regime_thresholds.items():
        if regime not in regime_counts.index:
            continue
        rc = regime_counts.loc[regime]
        hold_window = f"({thresh['sell_pct']:.0f}-{thresh['buy_pct']:.0f})"
        print(
            f"    {regime:<12}  buy>=pct{thresh['buy_pct']:.0f}  "
            f"sell<=pct{thresh['sell_pct']:.0f}  hold{hold_window}  "
            f"->  BUY:{rc.get('BUY', 0):>4}  SELL:{rc.get('SELL', 0):>4}  "
            f"HOLD:{rc.get('HOLD', 0):>4}"
        )


def _print_diagnostics(df: pd.DataFrame) -> None:
    signal_counts = df["signal"].value_counts()
    total         = len(df)
    avg_pos       = df["position_size"].mean()
    avg_pos_nhold = df.loc[df["signal"] != "HOLD", "position_size"].mean()
    strat_ret     = df["strategy_return"]

    sharpe = (
        strat_ret.mean() / strat_ret.std() * np.sqrt(252)
        if strat_ret.std() > 0 else np.nan
    )
    win_rate = (strat_ret > 0).sum() / max((strat_ret != 0).sum(), 1) * 100

    print("\n── Strategy Engine Diagnostics ─────────────────────────────────")
    print(f"  Total rows           : {total:,}")
    print(f"  Date range           : {df['date'].min().date()}  →  "
          f"{df['date'].max().date()}")

    print(f"\n  Signal distribution  :")
    for sig in ["BUY", "SELL", "HOLD"]:
        count = signal_counts.get(sig, 0)
        pct   = count / total * 100
        bar   = "█" * int(pct / 2)
        print(f"    {sig:<6}  {count:>5,}  ({pct:5.1f} %)  {bar}")

    print(f"\n  Position sizing      :")
    print(f"    Avg size (all rows)      : {avg_pos:.4f}")
    print(f"    Avg size (non-HOLD rows) : {avg_pos_nhold:.4f}")
    print(f"    Min / Max                : "
          f"{df['position_size'].min():.4f} / {df['position_size'].max():.4f}")

    print(f"\n  Strategy performance :")
    print(f"    Cumulative return    : {strat_ret.sum():+.4f}  "
          f"({strat_ret.sum() * 100:+.2f} %)")
    print(f"    Daily mean return    : {strat_ret.mean():+.6f}")
    print(f"    Daily std            : {strat_ret.std():.6f}")
    print(f"    Annualised Sharpe    : {sharpe:.3f}")
    print(f"    Win rate (non-zero)  : {win_rate:.1f} %")
    print("─────────────────────────────────────────────────────────────────\n")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """CLI entry point: run the v12 strategy engine pipeline."""
    print("=" * 60)
    print("  Strategy Engine Pipeline v12 — project_ari")
    print("=" * 60)
    try:
        df = strategy_pipeline()
        print(f"Pipeline finished successfully.  "
              f"Output: {len(df):,} rows.\n")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
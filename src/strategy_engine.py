"""
strategy_engine.py
------------------
Adaptive dual-logic trading strategy engine for Project ARI.

v14 — Dual-Logic Adaptive Switch  (Techkriti '26 final)
---------------------------------------------------------
Extends v13 Regime-Gated Execution with four targeted upgrades:

  1. Dual-Logic Signal Generation
     ─────────────────────────────
     BEAR / SIDEWAYS  →  LightGBM branch
         Use predicted_return percentile rank.  Confirmed alpha sources
         (IC +0.11 and +0.08 from the Post-WFO Signal Audit).

     BULL / RECOVERY  →  Trend-Following branch
         Ignore LightGBM entirely (IC −0.08 to −0.12 in these regimes).
         Generate signal from price structure only:
             BUY   if  close > MA50  AND  MA50 > MA50.shift(5)
             HOLD  otherwise
         No SELL signals in trend-following mode — macro direction is up,
         shorting into a bull trend is structurally wrong and adds
         unnecessary round-trip cost.

     The Entropy Kill-Switch (Gate 1) still applies to ALL rows first.
     High-entropy regime boundaries are unreliable regardless of signal
     source; both branches are gated identically at entropy >= 0.50.

  2. Dynamic Volatility Targeting  (replaces multi-stage static scaling)
     ──────────────────────────────────────────────────────────────────
     Single-formula position sizing:

         position_size = clip(TARGET_VOL / volatility_7
                              × regime_scale × heatmap_scalar,  0.0,  1.0)

     where TARGET_VOL = 0.02 (2 % daily vol target).
     Removes the v13 Stage 2 (vol30 adj) and Stage 3 (confidence scalar).
     Those were additional multipliers with no clear audit basis.

     Regime scales (v14):
         BEAR      1.20   (strongest LGB alpha)
         SIDEWAYS  1.10   (second LGB alpha)
         BULL      0.80   (trend-follow, moderate)
         RECOVERY  0.60   (trend-follow, conservative)

  3. Outlier Capping  (new preprocessing step)
     ──────────────────────────────────────────
     winsorize_features() caps volume_zscore_30 at ±3.0 σ before any
     feature reaches the signal generator.  Volume spikes corrupt both
     the HMM entropy estimate and the LGB percentile ranking.

  4. OOP Modular Architecture
     ─────────────────────────
     All logic is wrapped in AdaptiveTradingSystem with four method stages:
         preprocess()          load + winsorize
         detect_regimes()      entropy confidence + regime label preview
         generate_signals()    entropy gate + dual-logic regime router
         apply_risk_filters()  trend filters + sizing + returns
     The class exposes a fluent run() interface and delegates to the
     existing standalone functions for full backward compatibility.

Retained from v13
-----------------
• Entropy Kill-Switch (Gate 1, entropy >= 0.50 -> HOLD on ALL rows)
• 50-day MA trend filter
• Trend hold (block SELL vs MA20 momentum)
• Strong trend persistence (golden-cross block)
• Trend dominance override (SIDEWAYS rows only in v14)
• Minimum holding period (5 days)
• Position carry tracking
• Heatmap Vol-Entropy 50% reduction scalar (Stage 4)
• All suppression tracking columns + diagnostics

Replaced / removed vs v13
--------------------------
• apply_regime_gate     -> apply_dual_logic_regime_router
• Position sizing Stage 2 (vol30 adj) and Stage 3 (confidence scalar)
  -> replaced by pure TARGET_VOL / vol7 formula

Project structure expected
--------------------------
    project_ari/
    +-- data/
    |   +-- processed/
    |       +-- predictions.csv       <- input
    |       +-- trading_signals.csv   <- output
    +-- src/
        +-- strategy_engine.py

Usage
-----
    # OOP interface (recommended)
    from src.strategy_engine import AdaptiveTradingSystem
    df = AdaptiveTradingSystem().run()

    # Functional interface (backward-compatible)
    from src.strategy_engine import strategy_pipeline
    df = strategy_pipeline()

    # CLI
    python src/strategy_engine.py
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
# Configuration
# ---------------------------------------------------------------------------

# -- Signal generation -------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.002   # |predicted_return| floor for LGB signals
MIN_HOLDING_DAYS:     int   = 5
TREND_MA_WINDOW:      int   = 50      # MA window for trend filter + trend-follow

TREND_HOLD_MA_WINDOW:        int = 20
TREND_HOLD_SLOPE_LOOKBACK:   int = 3
TREND_FOLLOW_SLOPE_LOOKBACK: int = 5  # MA50 lookback for BULL/RECOVERY branch

# -- Entropy Kill-Switch (Gate 1) --------------------------------------------
# Applies to ALL rows regardless of regime; IC = -0.2548 above this threshold.
ENTROPY_KILL_THRESHOLD: float = 0.50

# -- Regime ID -> semantic label mapping -------------------------------------
# Derived from HMM emission statistics in regime_detection.py.
REGIME_ID_TO_LABEL: dict[int, str] = {
    0: "RECOVERY",
    1: "BULL",
    2: "SIDEWAYS",
    3: "BEAR",
}

# -- Dual-Logic Regime Configuration -----------------------------------------
# mode  = "LGB"          -> use LightGBM predicted_return percentile signals
# mode  = "TREND_FOLLOW" -> use close>MA50 and MA50>MA50.shift(5); no SELL
# scale = regime-specific position-size multiplier in vol targeting formula
REGIME_CONFIG: dict[str, dict] = {
    "BEAR"    : {"mode": "LGB",          "scale": 1.20},  # IC +0.1131
    "SIDEWAYS": {"mode": "LGB",          "scale": 1.10},  # IC +0.0839
    "BULL"    : {"mode": "TREND_FOLLOW", "scale": 0.80},  # IC -0.0785 -> ignore LGB
    "RECOVERY": {"mode": "TREND_FOLLOW", "scale": 0.60},  # IC -0.1182 -> ignore LGB
}
DEFAULT_REGIME_CONFIG: dict = {"mode": "LGB", "scale": 1.00}

TREND_FOLLOW_REGIMES: frozenset[str] = frozenset({"BULL", "RECOVERY"})
LGB_ALPHA_REGIMES:    frozenset[str] = frozenset({"BEAR", "SIDEWAYS"})

# -- Dynamic Volatility Targeting --------------------------------------------
TARGET_VOL:   float = 0.02    # 2% daily vol target (Req 2 formula numerator)
POSITION_MIN: float = 0.0
POSITION_MAX: float = 1.0     # hard cap prevents leverage > 100%

# -- Heatmap Vol-Entropy Scalar (retained from v13) --------------------------
# Trigger: vol7 >= Q75 AND entropy > 0.25 -> 50% size reduction.
# Audit: MAE = 0.067 in this bucket.
VOL_ENTROPY_ENTROPY_GATE: float = 0.25
VOL_ENTROPY_SIZE_PENALTY: float = 0.50

# -- Winsorisation (Req 3) ---------------------------------------------------
WINSORIZE_CLIP: float     = 3.0
WINSORIZE_COLS: list[str] = ["volume_zscore_30"]


# ===========================================================================
# 0. winsorize_features  (v14 NEW - Req 3)
# ===========================================================================
def winsorize_features(
    df:       pd.DataFrame,
    cols:     list[str] = WINSORIZE_COLS,
    clip_val: float     = WINSORIZE_CLIP,
) -> pd.DataFrame:
    """Cap extreme feature values at +/- clip_val standard deviations.

    Background
    ----------
    volume_zscore_30 is a rolling 30-day z-score of traded volume.  During
    exchange-specific events (liquidation cascades, listing spikes) it can
    reach +/- 10-20 sigma.  These outliers affect two downstream components:

    1. HMM regime assignment: a single row with |zscore| > 3 shifts the
       Gaussian emission means for the volume state, making the HMM declare
       a transient regime change that affects surrounding rows.

    2. LGB percentile ranking: an extreme volume row receives the maximum or
       minimum predicted_return rank in its fold, generating a spurious BUY
       or SELL signal with no causal price signal behind it.

    Winsorising at +/- 3 sigma retains all distributional information for
    normal variance (~99.7% of observations) while nullifying tail distortion.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Columns to winsorise.  Default: ["volume_zscore_30"].
    clip_val : float
        Symmetric cap level.  Default: 3.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped values in the specified columns.
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            print(
                f"[winsorize_features]  WARNING: column '{col}' not found -- skipped."
            )
            continue

        pre_min = float(df[col].min())
        pre_max = float(df[col].max())
        n_upper = int((df[col] >  clip_val).sum())
        n_lower = int((df[col] < -clip_val).sum())

        df[col] = df[col].clip(lower=-clip_val, upper=clip_val)

        print(
            f"[winsorize_features]  '{col}'  clip=+/-{clip_val}  |  "
            f"before: [{pre_min:.3f}, {pre_max:.3f}]  "
            f"after:  [{df[col].min():.3f}, {df[col].max():.3f}]  |  "
            f"upper capped: {n_upper:,}  lower capped: {n_lower:,}"
        )

    return df


# ===========================================================================
# 1. load_predictions
# ===========================================================================
def load_predictions(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the WFO predictions dataset.

    Parameters
    ----------
    filepath : Path or str

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError, ValueError
    """
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
        "regime_entropy", "volatility_7", "volatility_30", "target_return",
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

    print(f"[load_predictions]  Loaded {len(df):,} OOS rows from '{filepath}'.")
    return df


# ===========================================================================
# 2. compute_entropy_confidence
# ===========================================================================
def compute_entropy_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise regime entropy into a per-row confidence score [0, 1].

    confidence = 1 - normalised_entropy

    Retained for diagnostics; no longer drives position sizing in v14
    (the confidence scalar was removed from compute_position_sizes).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  with *confidence* column added.
    """
    df = df.copy()

    e       = df["regime_entropy"]
    e_min   = e.min()
    e_max   = e.max()
    e_range = e_max - e_min

    normalised = (e - e_min) / e_range if e_range > 0 else pd.Series(0.0, index=df.index)
    df["confidence"] = 1.0 - normalised

    print(
        f"[compute_entropy_confidence]  "
        f"avg entropy: {float(e.mean()):.4f}  |  "
        f"range: [{float(e_min):.4f}, {float(e_max):.4f}]  |  "
        f"avg confidence: {float(df['confidence'].mean()):.4f}"
    )
    return df


# ===========================================================================
# 3. generate_base_signal  (LGB branch; BULL/RECOVERY rows overwritten later)
# ===========================================================================
def generate_base_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Generate raw LightGBM signals from predicted return percentile ranks.

    Called for ALL rows as a first pass.  apply_dual_logic_regime_router
    subsequently overwrites BULL and RECOVERY rows with trend-following logic.

    Signal rules
    ------------
    BUY  : pred_percentile >= 0.80
    SELL : pred_percentile <= 0.20
    HOLD : middle 60%, or |predicted_return| < CONFIDENCE_THRESHOLD

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  with *signal* and *pred_percentile* columns added.
    """
    if "predicted_return" not in df.columns:
        raise ValueError("Column 'predicted_return' not found in DataFrame.")

    df = df.copy()
    df["pred_percentile"] = df["predicted_return"].rank(method="average", pct=True)

    conditions = [df["pred_percentile"] >= 0.80, df["pred_percentile"] <= 0.20]
    df["signal"] = np.select(conditions, ["BUY", "SELL"], default="HOLD")

    weak_mask = df["predicted_return"].abs() < CONFIDENCE_THRESHOLD
    n_weak    = int((weak_mask & (df["signal"] != "HOLD")).sum())
    df.loc[weak_mask, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[generate_base_signal]  LGB initial signals -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}  "
        f"| confidence filter suppressed: {n_weak:,}"
    )
    return df


# ===========================================================================
# 4. apply_entropy_kill_switch  (Gate 1; applies to ALL rows)
# ===========================================================================
def apply_entropy_kill_switch(
    df:                pd.DataFrame,
    entropy_threshold: float = ENTROPY_KILL_THRESHOLD,
) -> pd.DataFrame:
    """Hard Gate 1 -- force HOLD on any row with regime_entropy >= threshold.

    Applied universally before the regime router.  Both the LGB branch and
    the trend-following branch are gated: high entropy means the HMM is
    broadly spread across regime hypotheses, making both signal sources
    unreliable regardless of their internal logic.

    Audit basis: IC = -0.2548 for entropy >= 0.50 (n=56, 5.3% of OOS rows).

    Parameters
    ----------
    df : pd.DataFrame
    entropy_threshold : float

    Returns
    -------
    pd.DataFrame  with *_suppressed_entropy* boolean column added.
    """
    df = df.copy()

    kill_mask   = df["regime_entropy"] >= entropy_threshold
    active_mask = df["signal"] != "HOLD"
    suppressed  = kill_mask & active_mask

    n_kill       = int(kill_mask.sum())
    n_suppressed = int(suppressed.sum())

    df.loc[suppressed, "signal"] = "HOLD"
    df["_suppressed_entropy"] = suppressed

    counts = df["signal"].value_counts()
    print(
        f"[entropy_kill_switch]  Threshold: entropy >= {entropy_threshold}  |  "
        f"Rows above threshold: {n_kill:,}  |  "
        f"Active signals suppressed: {n_suppressed:,}\n"
        f"[entropy_kill_switch]  Post-gate -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 5. apply_dual_logic_regime_router  (v14 NEW; replaces apply_regime_gate)
# ===========================================================================
def apply_dual_logic_regime_router(
    df:             pd.DataFrame,
    config:         dict[str, dict] = REGIME_CONFIG,
    id_to_label:    dict[int, str]  = REGIME_ID_TO_LABEL,
    ma_window:      int             = TREND_MA_WINDOW,
    slope_lookback: int             = TREND_FOLLOW_SLOPE_LOOKBACK,
) -> pd.DataFrame:
    """Dual-Logic Adaptive Switch: route each row to its correct signal branch.

    Branching logic
    ---------------
    BEAR / SIDEWAYS  ->  LGB branch
        Signal from generate_base_signal() is kept unchanged.
        These are confirmed alpha sources (IC +0.11 / +0.08).

    BULL / RECOVERY  ->  Trend-Following branch
        LGB predictions have negative IC here (-0.08 / -0.12).
        Signal is overridden with pure price-structure logic:

            MA50   = close.rolling(50).mean()
            slope  = MA50 > MA50.shift(5)      <- 5-day slope check
            signal = "BUY"  if  (close > MA50) AND slope
                   = "HOLD" otherwise

        No SELL signals are generated in trend-following mode.
        In BULL and RECOVERY the macro direction is up; shorting creates
        unnecessary risk and round-trip costs.

    Entropy Kill-Switch interaction
    --------------------------------
    Rows already forced to HOLD by apply_entropy_kill_switch
    (_suppressed_entropy = True) are left completely untouched.
    The entropy gate is final and supersedes both signal branches.

    Regime scale assignment
    -----------------------
    Every row receives regime_scale used by compute_position_sizes:
        BEAR 1.20  SIDEWAYS 1.10  BULL 0.80  RECOVERY 0.60

    Signal attribution
    ------------------
    *signal_source* column written for diagnostics:
        "LGB"          -- predicted_return percentile signal
        "TREND_FOLLOW" -- MA50 cross + slope signal
        "ENTROPY_KILL" -- killed by entropy gate before routing
        "BASE_HOLD"    -- middle-band HOLD from generate_base_signal

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *market_regime*, *signal*, *close*, *_suppressed_entropy*.
    config : dict
    id_to_label : dict
    ma_window : int
    slope_lookback : int

    Returns
    -------
    pd.DataFrame  with *regime_label*, *regime_scale*, *signal_source* added.
    """
    df = df.copy()

    # -- Resolve integer regime ID -> semantic label -------------------------
    def _resolve(val) -> str:
        if isinstance(val, str):
            return val.strip().upper()
        if isinstance(val, (int, float)) and not pd.isna(val):
            return id_to_label.get(int(val), "REGIME_" + str(int(val)))
        return "UNKNOWN"

    df["regime_label"] = df["market_regime"].apply(_resolve)

    # -- Attach regime scale to every row ------------------------------------
    df["regime_scale"] = df["regime_label"].apply(
        lambda lbl: config.get(lbl, DEFAULT_REGIME_CONFIG)["scale"]
    )

    # -- Compute MA50 and its slope (trend-follow branch) --------------------
    if "ma50" in df.columns:
        ma50 = df["ma50"]
    elif "trend_ma50" in df.columns:
        ma50 = df["trend_ma50"]
        df["ma50"] = ma50
    else:
        df["ma50"] = df["close"].rolling(window=ma_window, min_periods=ma_window).mean()
        ma50 = df["ma50"]

    ma50_shifted = ma50.shift(slope_lookback)

    # Trend-following condition: price above MA50 AND MA50 slope is positive
    tf_condition = (
        (df["close"] > ma50)
        & (ma50 > ma50_shifted)
        & ma50.notna()
        & ma50_shifted.notna()
    )

    # -- Build signal_source from current state -----------------------------
    entropy_killed = df.get("_suppressed_entropy", pd.Series(False, index=df.index))

    signal_source = pd.Series("BASE_HOLD", index=df.index)
    signal_source[entropy_killed]                                   = "ENTROPY_KILL"
    signal_source[~entropy_killed & (df["signal"].isin(["BUY", "SELL"]))] = "LGB"

    # -- Apply trend-following override to BULL/RECOVERY rows ---------------
    # Only rows that:  (a) are in a trend-follow regime, (b) not entropy-killed
    tf_regime_mask = df["regime_label"].isin(TREND_FOLLOW_REGIMES) & ~entropy_killed

    n_tf_rows  = int(tf_regime_mask.sum())
    n_tf_buy   = int((tf_regime_mask & tf_condition).sum())
    n_tf_hold  = int((tf_regime_mask & ~tf_condition).sum())
    n_lgb_rows = int((~tf_regime_mask & ~entropy_killed).sum())
    n_warmup   = int((tf_regime_mask & ma50.isna()).sum())

    df.loc[tf_regime_mask & tf_condition,  "signal"] = "BUY"
    df.loc[tf_regime_mask & ~tf_condition, "signal"] = "HOLD"

    signal_source[tf_regime_mask] = "TREND_FOLLOW"
    df["signal_source"] = signal_source

    # _suppressed_regime tag kept for backtester compatibility (all False in v14)
    df["_suppressed_regime"] = pd.Series(False, index=df.index)

    counts = df["signal"].value_counts()
    print(f"\n[dual_logic_router]  Regime routing:")
    for label in sorted(config):
        lbl_config = config[label]
        cnt = int((df["regime_label"] == label).sum())
        print(
            "    " + label + " " * (10 - len(label)) +
            "mode=" + lbl_config["mode"] + " " * (14 - len(lbl_config["mode"])) +
            "scale=" + str(lbl_config["scale"]) +
            "  rows=" + f"{cnt:,}"
        )
    print(
        f"[dual_logic_router]  LGB rows: {n_lgb_rows:,}  |  "
        f"Trend-follow rows: {n_tf_rows:,}  (MA50 warm-up: {n_warmup:,})\n"
        f"[dual_logic_router]  Trend-follow -- "
        f"BUY: {n_tf_buy:,}  HOLD: {n_tf_hold:,}  (no SELL in trend-follow mode)\n"
        f"[dual_logic_router]  Post-router -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 6. apply_trend_filter
# ===========================================================================
def apply_trend_filter(
    df:        pd.DataFrame,
    ma_window: int = TREND_MA_WINDOW,
) -> pd.DataFrame:
    """Confirm signals against the 50-day MA.

    BUY  confirmed only when close > MA50.
    SELL confirmed only when close < MA50.
    Disagreements converted to HOLD.

    In v14, BULL/RECOVERY BUY signals trivially satisfy this filter
    (they were generated with close>MA50 as a precondition).  BEAR/SIDEWAYS
    SELL signals are the primary beneficiary of this filter.

    Parameters
    ----------
    df : pd.DataFrame
    ma_window : int

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "trend_ma50" not in df.columns:
        df["trend_ma50"] = df["close"].rolling(
            window=ma_window, min_periods=ma_window
        ).mean()

    ma       = df["trend_ma50"]
    ma_valid = ma.notna()
    above_ma = df["close"] > ma
    below_ma = df["close"] < ma

    buy_confirmed  = (df["signal"] == "BUY")  & above_ma & ma_valid
    sell_confirmed = (df["signal"] == "SELL") & below_ma & ma_valid
    disagreement   = ma_valid & ~buy_confirmed & ~sell_confirmed & (df["signal"] != "HOLD")

    n_buy_kept   = int(buy_confirmed.sum())
    n_sell_kept  = int(sell_confirmed.sum())
    n_suppressed = int(disagreement.sum())
    n_warmup     = int((~ma_valid).sum())

    df.loc[disagreement, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[trend_filter]  MA{ma_window}  |  "
        f"Warm-up: {n_warmup:,}  |  "
        f"BUY confirmed: {n_buy_kept:,}  |  "
        f"SELL confirmed: {n_sell_kept:,}  |  "
        f"Suppressed: {n_suppressed:,}\n"
        f"[trend_filter]  Post-filter -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 7. apply_trend_dominance_override
# ===========================================================================
def apply_trend_dominance_override(
    df:             pd.DataFrame,
    ma_window:      int = TREND_MA_WINDOW,
    slope_lookback: int = 5,
    regime_col:     str = "market_regime",
) -> pd.DataFrame:
    """Force BUY during confirmed uptrends -- SIDEWAYS rows only in v14.

    BULL/RECOVERY rows already source their signal from the trend-following
    branch and are excluded.  BEAR rows are excluded to protect SELL alpha.
    Only SIDEWAYS rows where the LGB signal disagrees with an uptrend are
    upgraded to BUY.

    Uptrend condition: close > MA50  AND  MA50 > MA50.shift(slope_lookback)

    Parameters
    ----------
    df : pd.DataFrame
    ma_window : int
    slope_lookback : int
    regime_col : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    ma_col = "ma50"
    if ma_col not in df.columns:
        df[ma_col] = (
            df.get("trend_ma50") or
            df["close"].rolling(window=ma_window, min_periods=ma_window).mean()
        )

    ma_series  = df[ma_col]
    ma_shifted = ma_series.shift(slope_lookback)

    trend_up = (
        (df["close"] > ma_series)
        & (ma_series > ma_shifted)
        & ma_series.notna()
        & ma_shifted.notna()
    )

    # Only override SIDEWAYS rows; exclude BEAR (protect SELL) and BULL/RECOVERY
    sideways_mask = pd.Series(False, index=df.index)
    resolved_col = regime_col if regime_col in df.columns else (
        "regime_label" if "regime_label" in df.columns else None
    )
    if resolved_col is not None:
        def _is_sideways(val) -> bool:
            if isinstance(val, str):
                return val.strip().upper() in {"SIDEWAYS", "NEUTRAL", "RANGING"}
            if isinstance(val, (int, float)) and not pd.isna(val):
                return int(val) == 2
            return False
        sideways_mask = df[resolved_col].apply(_is_sideways)

    override_mask = trend_up & sideways_mask
    n_overridden  = int((override_mask & (df["signal"] != "BUY")).sum())

    df.loc[override_mask, "signal"] = "BUY"
    df["trend_dominant"] = override_mask

    counts = df["signal"].value_counts()
    print(
        f"[trend_override]  Uptrend rows: {int(trend_up.sum()):,}  |  "
        f"SIDEWAYS eligible: {int(sideways_mask.sum()):,}  |  "
        f"Overridden -> BUY: {n_overridden:,}\n"
        f"[trend_override]  Post-override -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 8. apply_trend_hold
# ===========================================================================
def apply_trend_hold(
    df:             pd.DataFrame,
    ma_window:      int = TREND_HOLD_MA_WINDOW,
    slope_lookback: int = TREND_HOLD_SLOPE_LOOKBACK,
) -> pd.DataFrame:
    """Block premature SELL exits during short-term uptrends.

    Converts SELL -> HOLD when: close > MA20  AND  MA20 > MA20.shift(3)

    Parameters
    ----------
    df : pd.DataFrame
    ma_window : int
    slope_lookback : int

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["ma20"] = df["close"].rolling(window=ma_window, min_periods=ma_window).mean()

    ma20         = df["ma20"]
    ma20_shifted = ma20.shift(slope_lookback)

    trend_hold_mask = (
        (df["close"] > ma20)
        & (ma20 > ma20_shifted)
        & ma20.notna()
        & ma20_shifted.notna()
    )

    sell_in_trend  = trend_hold_mask & (df["signal"] == "SELL")
    n_sell_blocked = int(sell_in_trend.sum())
    df.loc[sell_in_trend, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[trend_hold]  Rows in trend: {int(trend_hold_mask.sum()):,}  |  "
        f"SELL blocked: {n_sell_blocked:,}\n"
        f"[trend_hold]  Post -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 9. apply_strong_trend_persistence
# ===========================================================================
def apply_strong_trend_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Block SELL signals during golden-cross confirmed strong bull trends.

    Condition: close > MA50  AND  MA20 > MA50
    Only SELL is affected.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    ma50 = df["ma50"] if "ma50" in df.columns else (
        df["trend_ma50"] if "trend_ma50" in df.columns else
        df["close"].rolling(50, min_periods=50).mean()
    )
    if "ma50" not in df.columns:
        df["ma50"] = ma50

    ma20 = df["ma20"] if "ma20" in df.columns else (
        df["close"].rolling(20, min_periods=20).mean()
    )
    if "ma20" not in df.columns:
        df["ma20"] = ma20

    strong_trend = (
        (df["close"] > ma50)
        & (ma20 > ma50)
        & ma50.notna()
        & ma20.notna()
    )

    sell_in_strong = strong_trend & (df["signal"] == "SELL")
    n_blocked      = int(sell_in_strong.sum())
    df.loc[sell_in_strong, "signal"] = "HOLD"

    counts = df["signal"].value_counts()
    print(
        f"[strong_trend]  Golden-cross rows: {int(strong_trend.sum()):,}  |  "
        f"SELL blocked: {n_blocked:,}\n"
        f"[strong_trend]  Post -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 10. apply_holding_period
# ===========================================================================
def apply_holding_period(
    df:       pd.DataFrame,
    min_days: int = MIN_HOLDING_DAYS,
) -> pd.DataFrame:
    """Enforce minimum holding period of min_days on active signals.

    Once a non-HOLD signal fires it is maintained for at least min_days
    rows regardless of subsequent signals.

    Parameters
    ----------
    df : pd.DataFrame
    min_days : int

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    signals       = df["signal"].tolist()
    filtered      = []
    hold_days_col = []

    days_held     = 0
    active_signal = "HOLD"

    for raw_sig in signals:
        if active_signal != "HOLD":
            if days_held < min_days:
                filtered.append(active_signal)
                hold_days_col.append(days_held)
                days_held += 1
            else:
                active_signal = raw_sig if raw_sig != "HOLD" else "HOLD"
                days_held     = 1 if active_signal != "HOLD" else 0
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

    counts       = pd.Series(filtered).value_counts()
    n_suppressed = sum(1 for o, f in zip(signals, filtered) if o != f)
    print(
        f"[holding_period]  Min hold: {min_days}d  |  "
        f"Suppressed: {n_suppressed:,}  |  "
        f"Post -- "
        f"BUY: {counts.get('BUY', 0):,}  "
        f"SELL: {counts.get('SELL', 0):,}  "
        f"HOLD: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 11. apply_position_carry
# ===========================================================================
def apply_position_carry(df: pd.DataFrame) -> pd.DataFrame:
    """Track position state (LONG / FLAT) for diagnostics."""
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
        f"[position_carry]  "
        f"buy: {counts.get('BUY', 0):,}  |  "
        f"sell: {counts.get('SELL', 0):,}  |  "
        f"hold: {counts.get('HOLD', 0):,}"
    )
    return df


# ===========================================================================
# 12. compute_position_sizes  (v14: pure dynamic vol targeting  - Req 2)
# ===========================================================================
def compute_position_sizes(
    df:         pd.DataFrame,
    target_vol: float = TARGET_VOL,
    pos_min:    float = POSITION_MIN,
    pos_max:    float = POSITION_MAX,
) -> pd.DataFrame:
    """Size positions using pure dynamic volatility targeting.

    Formula (v14)
    -------------
        base_size      = TARGET_VOL / volatility_7
        scaled_size    = base_size * regime_scale
        heatmap_scalar = 0.50  if (vol7 >= Q75) AND (entropy > 0.25)
                       = 1.00  otherwise
        raw_size       = scaled_size * heatmap_scalar
        position_size  = clip(raw_size, 0.0, 1.0)
        HOLD rows      -> position_size = 0.0  (forced)

    Design
    ------
    The v13 formula applied four multiplicative stages including a 30-day
    vol adjustment and a confidence scalar.  These are removed in v14 in
    favour of the single transparent formula above.

    Mathematical properties:
      - When vol7 rises, position_size falls proportionally (inverse-vol sizing).
      - With TARGET_VOL=0.02 and regime_scale=1.20 (BEAR), position_size
        peaks at 0.024/vol7, capped at 1.0.
      - At the historical BTC daily vol of ~0.04, BEAR position_size = 0.60.
      - If vol7 spikes to 0.08 (crash), position_size = 0.30 automatically,
        providing natural drawdown protection against the 30% penalty.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *signal*, *volatility_7*, *regime_scale*, *regime_entropy*.
    target_vol : float
        Daily volatility target.  Default 0.02.
    pos_min, pos_max : float
        Final clipping bounds.  Default 0.0 / 1.0.

    Returns
    -------
    pd.DataFrame  with *position_size* and *_heatmap_scalar_fired* added.
    """
    df = df.copy()

    # -- Base size: TARGET_VOL / vol7 ----------------------------------------
    vol7 = df["volatility_7"].replace(0, np.nan).fillna(df["volatility_7"].median())
    base_size   = target_vol / vol7
    scaled_size = base_size * df["regime_scale"]

    # -- Heatmap Vol-Entropy Scalar (tail-risk guard, retained from v13) ------
    vol7_q75 = float(df["volatility_7"].quantile(0.75))
    heatmap_trigger = (
        (df["volatility_7"] >= vol7_q75)
        & (df["regime_entropy"] > VOL_ENTROPY_ENTROPY_GATE)
    )
    heatmap_scalar = np.where(heatmap_trigger, VOL_ENTROPY_SIZE_PENALTY, 1.0)
    df["_heatmap_scalar_fired"] = heatmap_trigger
    n_heatmap = int(heatmap_trigger.sum())

    # -- Combine and clip ----------------------------------------------------
    raw_size = scaled_size * heatmap_scalar
    clipped  = raw_size.clip(lower=pos_min, upper=pos_max)

    df["position_size"] = np.where(df["signal"] == "HOLD", 0.0, clipped)

    non_hold = df.loc[df["signal"] != "HOLD", "position_size"]
    print(
        f"[position_sizing]  Formula: {target_vol}/vol7 * regime_scale * heatmap  |  "
        f"vol7_Q75: {vol7_q75:.5f}  |  "
        f"Heatmap scalar fired: {n_heatmap:,} rows (x{VOL_ENTROPY_SIZE_PENALTY})\n"
        f"[position_sizing]  Avg non-HOLD: {non_hold.mean():.4f}  |  "
        f"Overall avg: {df['position_size'].mean():.4f}  |  "
        f"Range: [{df['position_size'].min():.4f}, {df['position_size'].max():.4f}]"
    )
    return df


# ===========================================================================
# 13. compute_strategy_returns
# ===========================================================================
def compute_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row strategy returns.

    strategy_return = direction * position_size * target_return
    direction: BUY->+1, SELL->-1, HOLD->0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    direction             = df["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(float)
    df["strategy_return"] = direction * df["position_size"] * df["target_return"]

    cumulative = df["strategy_return"].sum()
    print(
        f"[strategy_returns]  Cumulative: {cumulative:+.4f} "
        f"({cumulative * 100:+.2f} %)"
    )
    return df


# ===========================================================================
# Private helpers
# ===========================================================================
def _print_diagnostics(df: pd.DataFrame) -> None:
    """Print full strategy diagnostics with v14 dual-logic attribution."""
    signal_counts = df["signal"].value_counts()
    total         = len(df)
    strat_ret     = df["strategy_return"]
    non_hold_pos  = df.loc[df["signal"] != "HOLD", "position_size"]

    sharpe = (
        strat_ret.mean() / strat_ret.std() * np.sqrt(252)
        if strat_ret.std() > 0 else float("nan")
    )
    win_rate = (strat_ret > 0).sum() / max((strat_ret != 0).sum(), 1) * 100

    n_entropy = int(df["_suppressed_entropy"].sum()) if "_suppressed_entropy" in df.columns else 0
    n_heatmap = int(df["_heatmap_scalar_fired"].sum()) if "_heatmap_scalar_fired" in df.columns else 0

    print("\n== Strategy Engine Diagnostics -- v14 Dual-Logic Adaptive ===========")
    print(f"  Total rows      : {total:,}")
    print(f"  Date range      : {df['date'].min().date()}  ->  {df['date'].max().date()}")

    print(f"\n  Signal distribution:")
    for sig in ["BUY", "SELL", "HOLD"]:
        count = signal_counts.get(sig, 0)
        pct   = count / total * 100
        bar   = chr(9608) * int(pct / 2)
        print(f"    {sig:<6}  {count:>5,}  ({pct:5.1f} %)  {bar}")

    if "signal_source" in df.columns:
        print(f"\n  Signal source (surviving non-HOLD):")
        active     = df[df["signal"] != "HOLD"]
        src_counts = active["signal_source"].value_counts()
        for src, cnt in src_counts.items():
            pct = 100 * cnt / max(len(active), 1)
            print(f"    {src:<16}  {cnt:>4,}  ({pct:.1f} %)")

    if "regime_label" in df.columns:
        print(f"\n  Regime label distribution:")
        for label in sorted(REGIME_CONFIG):
            lbl_mode  = REGIME_CONFIG[label]["mode"]
            lbl_scale = REGIME_CONFIG[label]["scale"]
            cnt       = int((df["regime_label"] == label).sum())
            pct       = 100 * cnt / total
            active_n  = int(
                ((df["regime_label"] == label) & (df["signal"] != "HOLD")).sum()
            )
            print(
                f"    {label:<10}  {cnt:>4,} rows ({pct:4.1f} %)  "
                f"mode={lbl_mode:<14}  scale={lbl_scale:.2f}  "
                f"active_signals={active_n:,}"
            )

    print(f"\n  Risk filters:")
    print(f"    1. Entropy Kill-Switch  (entropy >= {ENTROPY_KILL_THRESHOLD}) : "
          f"{n_entropy:>4,} suppressed")
    print(f"    2. Heatmap Vol-Entropy  (Q75 vol + ent>{VOL_ENTROPY_ENTROPY_GATE}) : "
          f"{n_heatmap:>4,} positions halved")

    print(f"\n  Position sizing:")
    print(f"    Avg (all rows)      : {df['position_size'].mean():.4f}")
    if len(non_hold_pos) > 0:
        print(f"    Avg (non-HOLD rows) : {non_hold_pos.mean():.4f}")
    print(f"    Range               : [{df['position_size'].min():.4f}, "
          f"{df['position_size'].max():.4f}]")

    print(f"\n  Strategy performance:")
    print(f"    Cumulative return : {strat_ret.sum():+.4f}  "
          f"({strat_ret.sum() * 100:+.2f} %)")
    print(f"    Daily mean        : {strat_ret.mean():+.6f}")
    print(f"    Daily std         : {strat_ret.std():.6f}")
    print(f"    Annualised Sharpe : {sharpe:.3f}")
    print(f"    Win rate          : {win_rate:.1f} %")
    print("======================================================================\n")


# ===========================================================================
# AdaptiveTradingSystem  (v14 NEW - Req 4)
# ===========================================================================
class AdaptiveTradingSystem:
    """OOP wrapper for the Project ARI v14 strategy pipeline.

    Wraps all standalone pipeline functions into four semantically meaningful
    method stages mirroring the competition system architecture:

        preprocess()          data loading + feature winsorisation
        detect_regimes()      entropy confidence + regime label preview
        generate_signals()    entropy gate + dual-logic regime router
        apply_risk_filters()  trend filters + holding period + position sizing

    Attributes
    ----------
    df : pd.DataFrame or None
        The working DataFrame.  None until preprocess() is called.
        Accessible at any stage for intermediate inspection.

    Usage
    -----
        # Standard
        df = AdaptiveTradingSystem().run()

        # Stepwise inspection
        sys = AdaptiveTradingSystem()
        sys.preprocess()
        print(sys.df["volume_zscore_30"].describe())  # after winsorisation
        sys.detect_regimes()
        sys.generate_signals()
        print(sys.df["signal_source"].value_counts())
        sys.apply_risk_filters()
        df = sys.save()

        # Config override
        df = AdaptiveTradingSystem(config={"TARGET_VOL": 0.015}).run()

    Parameters
    ----------
    input_path : Path or str or None
    output_path : Path or str or None
    config : dict or None
        Overrides for module-level constants.  Recognised keys:
        TARGET_VOL, POSITION_MIN, POSITION_MAX, MIN_HOLDING_DAYS,
        CONFIDENCE_THRESHOLD, ENTROPY_KILL_THRESHOLD, TREND_MA_WINDOW,
        TREND_HOLD_MA_WINDOW, TREND_HOLD_SLOPE_LOOKBACK,
        TREND_FOLLOW_SLOPE_LOOKBACK, VOL_ENTROPY_ENTROPY_GATE,
        VOL_ENTROPY_SIZE_PENALTY, WINSORIZE_CLIP.
    """

    _VALID_CONFIG_KEYS: frozenset[str] = frozenset({
        "TARGET_VOL", "POSITION_MIN", "POSITION_MAX",
        "MIN_HOLDING_DAYS", "CONFIDENCE_THRESHOLD",
        "ENTROPY_KILL_THRESHOLD", "TREND_MA_WINDOW",
        "TREND_HOLD_MA_WINDOW", "TREND_HOLD_SLOPE_LOOKBACK",
        "TREND_FOLLOW_SLOPE_LOOKBACK",
        "VOL_ENTROPY_ENTROPY_GATE", "VOL_ENTROPY_SIZE_PENALTY",
        "WINSORIZE_CLIP",
    })

    def __init__(
        self,
        input_path:  Path | str | None = None,
        output_path: Path | str | None = None,
        config:      dict | None        = None,
    ) -> None:
        self.input_path  = Path(input_path)  if input_path  else INPUT_PATH
        self.output_path = Path(output_path) if output_path else OUTPUT_PATH
        self.df: pd.DataFrame | None = None

        self._config: dict = {
            "TARGET_VOL"                 : TARGET_VOL,
            "POSITION_MIN"               : POSITION_MIN,
            "POSITION_MAX"               : POSITION_MAX,
            "MIN_HOLDING_DAYS"           : MIN_HOLDING_DAYS,
            "CONFIDENCE_THRESHOLD"       : CONFIDENCE_THRESHOLD,
            "ENTROPY_KILL_THRESHOLD"     : ENTROPY_KILL_THRESHOLD,
            "TREND_MA_WINDOW"            : TREND_MA_WINDOW,
            "TREND_HOLD_MA_WINDOW"       : TREND_HOLD_MA_WINDOW,
            "TREND_HOLD_SLOPE_LOOKBACK"  : TREND_HOLD_SLOPE_LOOKBACK,
            "TREND_FOLLOW_SLOPE_LOOKBACK": TREND_FOLLOW_SLOPE_LOOKBACK,
            "VOL_ENTROPY_ENTROPY_GATE"   : VOL_ENTROPY_ENTROPY_GATE,
            "VOL_ENTROPY_SIZE_PENALTY"   : VOL_ENTROPY_SIZE_PENALTY,
            "WINSORIZE_CLIP"             : WINSORIZE_CLIP,
        }
        if config:
            unknown = set(config) - self._VALID_CONFIG_KEYS
            if unknown:
                print(f"[AdaptiveTradingSystem]  WARNING: unknown config keys "
                      f"ignored: {sorted(unknown)}")
            for key, val in config.items():
                if key in self._VALID_CONFIG_KEYS:
                    self._config[key] = val
                    print(f"[AdaptiveTradingSystem]  Config override: {key} = {val}")

    # -- Stage 1 -------------------------------------------------------------
    def preprocess(self) -> "AdaptiveTradingSystem":
        """Load predictions.csv and apply winsorisation to volume_zscore_30.

        Operations
        ----------
        1. load_predictions()     -- validates schema, drops NaN-pred rows
        2. winsorize_features()   -- caps volume_zscore_30 at +/- WINSORIZE_CLIP sigma

        Returns
        -------
        AdaptiveTradingSystem  (self, for method chaining)
        """
        self._assert_stage("preprocess", requires_df=False)
        _sep("Stage 1: Preprocess")
        self.df = load_predictions(self.input_path)
        self.df = winsorize_features(
            self.df,
            cols     = WINSORIZE_COLS,
            clip_val = self._config["WINSORIZE_CLIP"],
        )
        return self

    # -- Stage 2 -------------------------------------------------------------
    def detect_regimes(self) -> "AdaptiveTradingSystem":
        """Compute entropy confidence and preview regime label distribution.

        Operations
        ----------
        1. compute_entropy_confidence()  -- normalise entropy to [0, 1]
        2. Preview: map market_regime IDs to semantic labels and report
           which signal branch each regime will use (LGB or TREND_FOLLOW).

        Returns
        -------
        AdaptiveTradingSystem
        """
        self._assert_stage("detect_regimes")
        _sep("Stage 2: Detect Regimes")
        self.df = compute_entropy_confidence(self.df)

        def _preview(val) -> str:
            if isinstance(val, (int, float)) and not pd.isna(val):
                return REGIME_ID_TO_LABEL.get(int(val), "REGIME_" + str(int(val)))
            return str(val)

        preview = self.df["market_regime"].apply(_preview).value_counts()
        print("[detect_regimes]  Regime distribution preview:")
        for label, cnt in preview.items():
            pct  = 100 * cnt / len(self.df)
            mode = REGIME_CONFIG.get(label, DEFAULT_REGIME_CONFIG)["mode"]
            print(f"    {label:<10}  {cnt:>4,} rows ({pct:4.1f} %)  -> {mode} branch")

        return self

    # -- Stage 3 -------------------------------------------------------------
    def generate_signals(self) -> "AdaptiveTradingSystem":
        """Apply Entropy Kill-Switch and Dual-Logic Regime Router.

        Operations
        ----------
        1. generate_base_signal()            -- LGB percentile signals (all rows)
        2. apply_entropy_kill_switch()        -- entropy >= 0.50 -> HOLD (all rows)
        3. apply_dual_logic_regime_router()   -- BEAR/SIDEWAYS: keep LGB signal
                                                BULL/RECOVERY: trend-follow override

        Returns
        -------
        AdaptiveTradingSystem
        """
        self._assert_stage("generate_signals")
        _sep("Stage 3: Generate Signals (Dual-Logic)")
        self.df = generate_base_signal(self.df)
        self.df = apply_entropy_kill_switch(
            self.df,
            entropy_threshold=self._config["ENTROPY_KILL_THRESHOLD"],
        )
        self.df = apply_dual_logic_regime_router(
            self.df,
            config         = REGIME_CONFIG,
            id_to_label    = REGIME_ID_TO_LABEL,
            ma_window      = self._config["TREND_MA_WINDOW"],
            slope_lookback = self._config["TREND_FOLLOW_SLOPE_LOOKBACK"],
        )
        return self

    # -- Stage 4 -------------------------------------------------------------
    def apply_risk_filters(self) -> "AdaptiveTradingSystem":
        """Apply all trend filters, holding period, and position sizing.

        Operations
        ----------
        1. apply_trend_filter()              -- 50-day MA confirmation
        2. apply_trend_dominance_override()  -- force BUY in SIDEWAYS uptrends
        3. apply_trend_hold()                -- block SELL vs MA20 momentum
        4. apply_strong_trend_persistence()  -- block SELL at golden cross
        5. apply_holding_period()            -- minimum 5-day hold
        6. compute_position_sizes()          -- pure vol targeting formula
        7. apply_position_carry()            -- LONG/FLAT state tracking
        8. Safety net: HOLD -> position_size = 0
        9. compute_strategy_returns()

        Returns
        -------
        AdaptiveTradingSystem
        """
        self._assert_stage("apply_risk_filters")
        _sep("Stage 4: Risk Filters & Position Sizing")

        self.df = apply_trend_filter(self.df, ma_window=self._config["TREND_MA_WINDOW"])
        self.df = apply_trend_dominance_override(
            self.df, ma_window=self._config["TREND_MA_WINDOW"]
        )
        self.df = apply_trend_hold(
            self.df,
            ma_window      = self._config["TREND_HOLD_MA_WINDOW"],
            slope_lookback = self._config["TREND_HOLD_SLOPE_LOOKBACK"],
        )
        self.df = apply_strong_trend_persistence(self.df)
        self.df = apply_holding_period(
            self.df, min_days=self._config["MIN_HOLDING_DAYS"]
        )
        self.df = compute_position_sizes(
            self.df,
            target_vol = self._config["TARGET_VOL"],
            pos_min    = self._config["POSITION_MIN"],
            pos_max    = self._config["POSITION_MAX"],
        )
        self.df = apply_position_carry(self.df)
        self.df.loc[self.df["signal"] == "HOLD", "position_size"] = 0.0
        self.df = compute_strategy_returns(self.df)
        return self

    # -- Output --------------------------------------------------------------
    def save(self) -> pd.DataFrame:
        """Persist signals to trading_signals.csv and return the DataFrame.

        Drops internal helper and diagnostic columns before saving so the
        downstream backtester receives only production-relevant columns.

        Returns
        -------
        pd.DataFrame
        """
        self._assert_stage("save")
        _print_diagnostics(self.df)

        drop_cols = [
            "regime_scale", "hold_day", "pred_percentile",
            "trend_ma50", "carry_position", "confidence",
            "ma50", "ma20", "trend_dominant",
            "_suppressed_entropy", "_suppressed_regime", "_heatmap_scalar_fired",
        ]
        out_df = self.df.drop(columns=drop_cols, errors="ignore")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"[AdaptiveTradingSystem]  Saved -> '{self.output_path}'  "
              f"({len(out_df):,} rows).")
        return out_df

    # -- Orchestrator --------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Run the full pipeline: preprocess -> detect_regimes ->
        generate_signals -> apply_risk_filters -> save.

        Returns
        -------
        pd.DataFrame
        """
        print("=" * 68)
        print("  AdaptiveTradingSystem -- Project ARI v14  (Techkriti 26)")
        print("=" * 68)
        return (
            self
            .preprocess()
            .detect_regimes()
            .generate_signals()
            .apply_risk_filters()
            .save()
        )

    # -- Internal helpers ----------------------------------------------------
    def _assert_stage(self, caller: str, requires_df: bool = True) -> None:
        if requires_df and self.df is None:
            raise RuntimeError(
                f"[AdaptiveTradingSystem.{caller}]  "
                "preprocess() must be called before this method."
            )


# ===========================================================================
# Module-level helpers
# ===========================================================================
def _sep(title: str) -> None:
    """Print a stage separator."""
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


# ===========================================================================
# Functional pipeline (backward-compatible)
# ===========================================================================
def strategy_pipeline(
    input_path:  Path | str   = INPUT_PATH,
    output_path: Path | str   = OUTPUT_PATH,
    config:      dict | None  = None,
) -> pd.DataFrame:
    """Backward-compatible functional entry point.

    Delegates entirely to AdaptiveTradingSystem.run().

    Parameters
    ----------
    input_path, output_path : Path or str
    config : dict or None

    Returns
    -------
    pd.DataFrame
    """
    return AdaptiveTradingSystem(
        input_path  = input_path,
        output_path = output_path,
        config      = config,
    ).run()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """CLI entry point."""
    try:
        df = strategy_pipeline()
        print(f"Pipeline finished successfully.  Output: {len(df):,} rows.\n")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
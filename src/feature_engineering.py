"""
feature_engineering.py
-----------------------
Feature engineering pipeline for BTC OHLCV data.
Produces a rich feature set suitable for training algorithmic trading models.

Project structure expected:
    project_ari/
    ├── data/
    │   └── processed/
    │       ├── clean_btc_data.csv   ← input
    │       └── features_btc.csv    ← output
    └── src/
        └── feature_engineering.py

Usage:
    Run directly:   python src/feature_engineering.py
    Import:         from src.feature_engineering import feature_pipeline
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "clean_btc_data.csv"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "features_btc.csv"

# Rolling window sizes
SHORT_WINDOW  = 7
LONG_WINDOW   = 30
RSI_PERIOD    = 14
ZSCORE_WINDOW = 30   # window for rolling z-score normalisation of derived features


# ---------------------------------------------------------------------------
# 1. load_clean_data
# ---------------------------------------------------------------------------
def load_clean_data(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the cleaned BTC dataset produced by data_pipeline.py.

    Parameters
    ----------
    filepath : Path or str
        Path to the cleaned CSV file.
        Defaults to ``data/processed/clean_btc_data.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with *date* parsed as ``datetime64[ns]``, sorted
        chronologically, and index reset to a clean RangeIndex.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *filepath*.
    ValueError
        If required OHLCV columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found: {filepath}\n"
            "Run data_pipeline.py first to generate clean_btc_data.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    required = {"date", "open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required column(s): {missing}")

    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[load_clean_data]  Loaded {len(df):,} rows from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 2. create_price_features
# ---------------------------------------------------------------------------
def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all price-derived features and append them to the DataFrame.

    Design principles
    -----------------
    * Every feature at row *t* uses **only** data from rows ≤ t-1 (no
      lookahead).  Rolling windows never include the current row's future.
    * Raw price-level quantities (momentum, VWAP deviation) are normalised
      into rolling z-scores so the LightGBM model sees each value relative
      to the recent 30-day distribution rather than the global 3-year one.
      This is critical for regime-robustness: a +$1,000 move meant something
      very different in 2020 vs. 2021.

    Features added
    --------------
    log_return              : Daily log return  log(close_t / close_{t-1})
    volatility_7            : 7-day rolling std of log returns
    volatility_30           : 30-day rolling std of log returns
    volatility_7_zscore     : Rolling 30-day z-score of volatility_7
    volatility_30_zscore    : Rolling 30-day z-score of volatility_30
    momentum_7_adj          : Risk-adjusted 7-day momentum
                              (close - close[t-7]) / (volatility_7 × close[t-7])
    momentum_30_adj         : Risk-adjusted 30-day momentum
                              (close - close[t-30]) / (volatility_30 × close[t-30])
    rsi_14                  : 14-period RSI (Wilder smoothing; bounded [0, 100])
    vwap_dev_30             : Rolling 30-day VWAP deviation  (close - vwap_30) / vwap_30
    vwap_dev_30_zscore      : Rolling 30-day z-score of vwap_dev_30

    Note: the raw ``vwap`` intermediate column is **not** retained in the
    output; only the deviation and its z-score are meaningful as ML features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *close* and *volume* columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended (copy).
    """
    df    = df.copy()
    close = df["close"]
    vol   = df["volume"]

    # ── Log Returns ────────────────────────────────────────────────────────
    # Uses close[t-1] only → no lookahead.
    df["log_return"] = np.log(close / close.shift(1))

    # ── Rolling Volatility ─────────────────────────────────────────────────
    # std over the previous N log-returns.  rolling(N) at position t uses
    # rows [t-N+1 … t] — the current row's log_return is t, which itself
    # only depends on close[t-1].  No lookahead.
    df["volatility_7"]  = df["log_return"].rolling(SHORT_WINDOW).std()
    df["volatility_30"] = df["log_return"].rolling(LONG_WINDOW).std()

    # ── Rolling Z-Score of Volatility ─────────────────────────────────────
    # Normalises each volatility value against the trailing 30-day
    # distribution of that same volatility series.  At time t, the z-score
    # uses volatility values from [t-29 … t] — all of which are computed
    # from log-returns ending no later than t.  No lookahead.
    df["volatility_7_zscore"]  = _rolling_zscore(df["volatility_7"],  ZSCORE_WINDOW)
    df["volatility_30_zscore"] = _rolling_zscore(df["volatility_30"], ZSCORE_WINDOW)

    # ── Risk-Adjusted Momentum ─────────────────────────────────────────────
    # Raw: close[t] - close[t-N]  (price-level, non-stationary, regime-dependent)
    # Adjusted: divide by (rolling_vol × close[t-N]) to make it unitless and
    # comparable across calm vs. volatile regimes.
    #
    # Formula: (close[t] - close[t-N]) / (vol_N[t] × close[t-N])
    #   ≡  pct_change(N) / vol_N[t]
    #
    # Denominator clamp: replace zero/NaN vol with NaN so the division
    # silently propagates NaN rather than inf; NaN rows are dropped in
    # feature_pipeline's dropna() step.
    vol7_safe  = df["volatility_7"].replace(0, np.nan)
    vol30_safe = df["volatility_30"].replace(0, np.nan)

    raw_mom_7  = close - close.shift(SHORT_WINDOW)
    raw_mom_30 = close - close.shift(LONG_WINDOW)
    ref_close_7  = close.shift(SHORT_WINDOW).replace(0, np.nan)
    ref_close_30 = close.shift(LONG_WINDOW).replace(0, np.nan)

    df["momentum_7_adj"]  = raw_mom_7  / (vol7_safe  * ref_close_7)
    df["momentum_30_adj"] = raw_mom_30 / (vol30_safe * ref_close_30)

    # ── RSI (14-period, Wilder's smoothed method) ──────────────────────────
    # RSI is already bounded in [0, 100]; no z-score needed.
    df["rsi_14"] = _rsi(close, period=RSI_PERIOD)

    # ── Rolling VWAP Deviation ─────────────────────────────────────────────
    # Cumulative VWAP drifts indefinitely and is non-stationary over 3 years.
    # A 30-day rolling VWAP anchors the reference price to recent activity,
    # making the deviation stationary and regime-comparable.
    df["_vwap_30"]      = _vwap_rolling(close, vol, LONG_WINDOW)
    df["vwap_dev_30"]   = (close - df["_vwap_30"]) / df["_vwap_30"]
    df.drop(columns=["_vwap_30"], inplace=True)   # intermediate only

    # Z-score of the deviation to further normalise across regimes.
    df["vwap_dev_30_zscore"] = _rolling_zscore(df["vwap_dev_30"], ZSCORE_WINDOW)

    price_features = [
        "log_return",
        "volatility_7", "volatility_30",
        "volatility_7_zscore", "volatility_30_zscore",
        "momentum_7_adj", "momentum_30_adj",
        "rsi_14",
        "vwap_dev_30", "vwap_dev_30_zscore",
    ]
    print(f"[create_price_features]   Added {len(price_features)} price feature(s): "
          f"{price_features}")
    return df


# ---------------------------------------------------------------------------
# 3. create_volume_features
# ---------------------------------------------------------------------------
def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-derived features and append them to the DataFrame.

    Features added
    --------------
    volume_zscore_30        : (volume - rolling_mean_30) / rolling_std_30
                              Identifies abnormal trading activity relative to
                              the trailing 30-day distribution.
    volume_price_imbalance  : log_return × volume_zscore_30
                              Interaction term capturing high-conviction
                              directional moves: a large return on abnormally
                              high volume is a much stronger signal than the
                              same return on thin volume.

    Lookahead note
    --------------
    ``volume_zscore_30`` uses ``rolling(30)`` which at row *t* includes
    volume[t].  This is safe because volume[t] is contemporaneous with
    close[t] — it is fully observed before the *next-day* target is realised.
    The interaction term inherits the same guarantee.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *volume* and *log_return* columns.
        Must be called **after** :func:`create_price_features`.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new volume feature columns appended.
    """
    df = df.copy()
    volume = df["volume"]

    roll_mean = volume.rolling(LONG_WINDOW).mean()
    roll_std  = volume.rolling(LONG_WINDOW).std()

    df["volume_zscore_30"] = (volume - roll_mean) / roll_std

    # ── Feature Interaction: volume × price direction ──────────────────────
    # Both operands are available at time t without lookahead.
    df["volume_price_imbalance"] = df["log_return"] * df["volume_zscore_30"]

    volume_features = ["volume_zscore_30", "volume_price_imbalance"]
    print(f"[create_volume_features]  Added {len(volume_features)} volume feature(s): "
          f"{volume_features}")
    return df


# ---------------------------------------------------------------------------
# 4. create_target_variable
# ---------------------------------------------------------------------------
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Append the supervised-learning target to the DataFrame.

    Target
    ------
    target_return : Next-day log return  log(close_{t+1} / close_t)
                    The final row will be NaN (no future close available)
                    and will be dropped downstream.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a *close* column.

    Returns
    -------
    pd.DataFrame
        DataFrame with *target_return* column appended.
    """
    df = df.copy()
    df["target_return"] = np.log(df["close"].shift(-1) / df["close"])

    print("[create_target_variable]  Added target column: ['target_return']  "
          "(next-day log return).")
    return df


# ---------------------------------------------------------------------------
# 5. feature_pipeline
# ---------------------------------------------------------------------------
def feature_pipeline(
    input_path:  Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline end-to-end.

    Steps
    -----
    1. Load cleaned OHLCV data     (:func:`load_clean_data`).
    2. Engineer price features     (:func:`create_price_features`).
    3. Engineer volume features    (:func:`create_volume_features`).
    4. Append target variable      (:func:`create_target_variable`).
    5. Drop warm-up / NaN rows.
       Warm-up depth analysis (all windows are causal):
       • volatility_30             needs 30 rows of log_return
       • volatility_*_zscore       needs 30 rows of volatility  (≤ 59 total)
       • momentum_30_adj           needs 30 rows of close + 30 rows of vol30
       • rsi_14                    needs 14 rows
       • vwap_dev_30               needs 30 rows
       • vwap_dev_30_zscore        needs 30 rows of vwap_dev_30 (≤ 59 total)
       • volume_zscore_30          needs 30 rows
       • volume_price_imbalance    inherits from above
       • target_return             loses 1 trailing row (shift(-1))
       The single global ``dropna()`` correctly handles all of the above
       because NaN propagation from each rolling window is preserved until
       the final drop.
    6. Print diagnostics and save to *output_path*.

    Parameters
    ----------
    input_path  : Path or str
        Location of the cleaned CSV.
    output_path : Path or str
        Destination for the feature CSV.

    Returns
    -------
    pd.DataFrame
        Fully engineered, NaN-free feature DataFrame.
    """
    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_clean_data(input_path)
    rows_raw = len(df)

    # ── 2–4. Feature creation ──────────────────────────────────────────────
    df = create_price_features(df)
    df = create_volume_features(df)
    df = create_target_variable(df)

    # ── 5. Drop NaN rows ───────────────────────────────────────────────────
    # Global dropna() covers all rolling warm-up rows (up to ~59 leading rows
    # for features that chain two 30-day windows) plus the 1 trailing row
    # lost to target_return's shift(-1).
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    rows_dropped = rows_raw - len(df)

    # ── 6. Diagnostics ─────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns
                    if c not in {"date", "open", "high", "low", "close", "volume"}]

    print("\n── Feature Engineering Diagnostics ─────────────────────────────")
    print(f"  Input rows          : {rows_raw:,}")
    print(f"  Rows dropped (NaN)  : {rows_dropped:,}  "
          f"(rolling warm-up ≤59 rows + 1 trailing target row)")
    print(f"  Output shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Features created    : {len(feature_cols)}")
    print(f"  Feature columns     : {feature_cols}")
    print(f"  Date range          : {df['date'].min().date()}  →  "
          f"{df['date'].max().date()}")
    print("\n  First 3 rows (feature columns only):")
    print(df[["date"] + feature_cols].head(3).to_string(index=False))
    print("─────────────────────────────────────────────────────────────────\n")

    # ── Save ───────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[feature_pipeline]  Saved feature dataset → '{output_path}'  "
          f"({len(df):,} rows).")

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothed moving average (EWM).

    Parameters
    ----------
    series : pd.Series
        Close-price series.
    period : int
        Look-back period (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100].
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing: equivalent to EMA with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)   # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a causal rolling z-score for a pandas Series.

    At each position *t*, the mean and standard deviation are computed
    over the window ``[t - window + 1, t]`` — i.e. entirely from past
    and current values.  No future data is touched.

    Z = (x_t - μ_{t-window+1:t}) / σ_{t-window+1:t}

    Where σ is the sample standard deviation (ddof=1).  If σ = 0 for
    a given window (all values identical), the result is 0 rather than
    NaN or inf, because the deviation from a constant mean is zero.

    Parameters
    ----------
    series : pd.Series
        Input time series (must already be causally computed).
    window : int
        Look-back window length.

    Returns
    -------
    pd.Series
        Rolling z-scores, NaN for the first ``window - 1`` rows
        (insufficient history).
    """
    roll   = series.rolling(window)
    mean_  = roll.mean()
    std_   = roll.std()                     # ddof=1 by default
    # Where std is zero, set z-score to 0 (no deviation from flat series)
    zscore = (series - mean_) / std_.replace(0, np.nan)
    return zscore.fillna(0) if std_.eq(0).any() else zscore


def _vwap_rolling(
    price:  pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Compute a causal rolling Volume-Weighted Average Price.

    Unlike cumulative VWAP, the rolling variant anchors the reference
    price to a recent window, making ``vwap_deviation`` stationary and
    comparable across different price regimes.

    At position *t*, the rolling VWAP is:
        VWAP_t = Σ_{i=t-window+1}^{t}  (price_i × volume_i)
                 ─────────────────────────────────────────────
                 Σ_{i=t-window+1}^{t}  volume_i

    All data in the sum is at or before *t* → no lookahead.

    Parameters
    ----------
    price  : pd.Series   Proxy price (close).
    volume : pd.Series   Traded volume.
    window : int         Look-back window length.

    Returns
    -------
    pd.Series
        Rolling VWAP, NaN for the first ``window - 1`` rows.
    """
    pv_sum  = (price * volume).rolling(window).sum()
    vol_sum = volume.rolling(window).sum()
    return pv_sum / vol_sum.replace(0, np.nan)


def _vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate cumulative Volume-Weighted Average Price.

    .. deprecated::
        Cumulative VWAP is retained only for backward compatibility.
        Use :func:`_vwap_rolling` for any new ML feature work, as
        cumulative VWAP is non-stationary over multi-year datasets.

    VWAP = Σ(price × volume) / Σ(volume)

    Parameters
    ----------
    price  : pd.Series  Proxy price (close).
    volume : pd.Series  Traded volume.

    Returns
    -------
    pd.Series
        Cumulative VWAP aligned to *price* index.
    """
    cum_pv  = (price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / cum_vol


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: execute the feature engineering pipeline."""
    print("=" * 60)
    print("  Feature Engineering Pipeline — project_ari")
    print("=" * 60)
    try:
        df = feature_pipeline()
        print(f"\nPipeline finished successfully.")
        print(f"Final feature dataset: "
              f"{df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
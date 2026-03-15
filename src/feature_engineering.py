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
SHORT_WINDOW = 7
LONG_WINDOW  = 30
RSI_PERIOD   = 14


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

    Features added
    --------------
    log_return        : Daily log return  log(close_t / close_{t-1})
    volatility_7      : 7-day rolling std of log returns
    volatility_30     : 30-day rolling std of log returns
    momentum_7        : close - close.shift(7)
    momentum_30       : close - close.shift(30)
    rsi_14            : 14-period Relative Strength Index (Wilder smoothing)
    vwap              : Cumulative VWAP  Σ(close × volume) / Σ(volume)
    vwap_deviation    : (close - vwap) / vwap

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *close* and *volume* columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended (in-place copy).
    """
    df = df.copy()
    close  = df["close"]
    volume = df["volume"]

    # ── Log Returns ────────────────────────────────────────────────────────
    df["log_return"] = np.log(close / close.shift(1))

    # ── Rolling Volatility (std of log returns) ────────────────────────────
    df["volatility_7"]  = df["log_return"].rolling(SHORT_WINDOW).std()
    df["volatility_30"] = df["log_return"].rolling(LONG_WINDOW).std()

    # ── Momentum ───────────────────────────────────────────────────────────
    df["momentum_7"]  = close - close.shift(SHORT_WINDOW)
    df["momentum_30"] = close - close.shift(LONG_WINDOW)

    # ── RSI (14-period, Wilder's smoothed method) ──────────────────────────
    df["rsi_14"] = _rsi(close, period=RSI_PERIOD)

    # ── VWAP (cumulative, using close as price proxy) ──────────────────────
    df["vwap"] = _vwap(close, volume)

    # ── VWAP Deviation ─────────────────────────────────────────────────────
    df["vwap_deviation"] = (close - df["vwap"]) / df["vwap"]

    price_features = [
        "log_return", "volatility_7", "volatility_30",
        "momentum_7", "momentum_30", "rsi_14",
        "vwap", "vwap_deviation",
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
    volume_zscore_30 : (volume - rolling_mean_30) / rolling_std_30
                       Identifies abnormal trading activity relative to the
                       trailing 30-day distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a *volume* column.

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

    volume_features = ["volume_zscore_30"]
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
    1. Load cleaned OHLCV data  (:func:`load_clean_data`).
    2. Engineer price features  (:func:`create_price_features`).
    3. Engineer volume features (:func:`create_volume_features`).
    4. Append target variable   (:func:`create_target_variable`).
    5. Drop rows containing NaN values introduced by rolling windows
       or the forward-shifted target.
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
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    rows_dropped = rows_raw - len(df)

    # ── 6. Diagnostics ─────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns
                    if c not in {"date", "open", "high", "low", "close", "volume"}]

    print("\n── Feature Engineering Diagnostics ─────────────────────────────")
    print(f"  Input rows          : {rows_raw:,}")
    print(f"  Rows dropped (NaN)  : {rows_dropped:,}  "
          f"(rolling window warm-up + target shift)")
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


def _vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate cumulative Volume-Weighted Average Price.

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
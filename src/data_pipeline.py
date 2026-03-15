"""
data_pipeline.py
----------------
Modular data pipeline for loading, validating, cleaning, and saving
BTC OHLCV market data.

Project structure expected:
    project_ari/
    ├── data/
    │   ├── raw/btc_dataset.csv
    │   └── processed/
    └── src/
        └── data_pipeline.py

Usage:
    Run directly:   python src/data_pipeline.py
    Import:         from src.data_pipeline import clean_pipeline
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent          # project_ari/src/
_PROJECT_ROOT = _SRC_DIR.parent                     # project_ari/
RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "btc_dataset.csv"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "clean_btc_data.csv"

REQUIRED_COLUMNS: list[str] = ["date", "open", "high", "low", "close", "volume"]
MAX_INTERPOLATION_GAP: int = 3   # rows; gaps larger than this use forward-fill

# Columns that must be strictly positive; any row with a zero or negative
# value in any of these is considered a price/volume glitch.
_POSITIVE_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# 1. load_data
# ---------------------------------------------------------------------------
def load_data(filepath: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load raw BTC OHLCV data from a CSV file.

    Parameters
    ----------
    filepath : Path or str
        Location of the raw CSV file.
        Defaults to ``data/raw/btc_dataset.csv`` relative to the project root.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with the date column parsed as ``datetime64[ns]``
        and rows sorted chronologically.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *filepath*.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            "Make sure btc_dataset.csv is placed in data/raw/."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])

    # Normalise column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Sort chronologically and reset index
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[load_data]  Loaded {len(df):,} rows from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 2. validate_data
# ---------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the DataFrame against expected schema and remove duplicates.

    Checks performed
    ----------------
    * All required columns (date, open, high, low, close, volume) are present.
    * Duplicate rows are detected and removed.
    * Diagnostic summary (shape, date range, missing-value counts) is printed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with duplicates removed.

    Raises
    ------
    ValueError
        If one or more required columns are absent.
    """
    # --- Column presence check -------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # --- Duplicate removal -----------------------------------------------
    n_dupes = df.duplicated().sum()
    if n_dupes:
        print(f"[validate_data]  Removing {n_dupes:,} duplicate row(s).")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[validate_data]  No duplicate rows found.")

    # --- Diagnostics ------------------------------------------------------
    print("\n── Dataset Diagnostics ─────────────────────────────────────────")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range   : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"  Columns      : {list(df.columns)}")

    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing:
        print(f"\n  Missing values ({total_missing:,} total):")
        for col, cnt in missing_counts[missing_counts > 0].items():
            pct = cnt / len(df) * 100
            print(f"    {col:<10} {cnt:>6,}  ({pct:.2f} %)")
    else:
        print("  Missing values : none detected")
    print("────────────────────────────────────────────────────────────────\n")

    return df


# ---------------------------------------------------------------------------
# 3. remove_price_glitches
# ---------------------------------------------------------------------------
def remove_price_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing invalid (zero or negative) price or volume data.

    A row is considered a price glitch if any of the core OHLCV fields —
    open, high, low, close, or volume — is less than or equal to zero.
    Such values are physically impossible in a live market and indicate
    data-feed errors, missing-data fill artefacts, or exchange outages
    that would corrupt downstream feature engineering and signal generation.

    Rows are dropped entirely rather than imputed because a zero price
    carries no recoverable information; linear interpolation across a
    zero would produce artifically low values that distort returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from :func:`validate_data`, containing all required columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with glitch rows removed and the index reset.
    """
    df = df.copy()
    n_before = len(df)

    # Build a mask that is True for any row containing a non-positive value
    # in any of the five price/volume columns.
    glitch_mask = (
        (df["close"]  <= 0) |
        (df["volume"] <= 0) |
        (df["open"]   <= 0) |
        (df["high"]   <= 0) |
        (df["low"]    <= 0)
    )

    # Per-column breakdown for the diagnostic (only columns that have glitches)
    col_counts = {
        col: int((df[col] <= 0).sum())
        for col in _POSITIVE_COLUMNS
        if (df[col] <= 0).sum() > 0
    }

    df = df[~glitch_mask].reset_index(drop=True)

    rows_removed   = n_before - len(df)
    rows_remaining = len(df)

    print(
        f"[remove_price_glitches]  rows_removed: {rows_removed:,}  |  "
        f"rows_remaining: {rows_remaining:,}"
    )
    if col_counts:
        detail = "  ".join(f"{col}: {cnt:,}" for col, cnt in col_counts.items())
        print(f"[remove_price_glitches]  Glitch breakdown — {detail}")
    else:
        print("[remove_price_glitches]  No price or volume glitches detected.")

    return df


# ---------------------------------------------------------------------------
# 4. handle_missing_values
# ---------------------------------------------------------------------------
def handle_missing_values(
    df: pd.DataFrame,
    max_gap: int = MAX_INTERPOLATION_GAP,
) -> pd.DataFrame:
    """Impute missing values using a gap-size-aware strategy.

    Strategy
    --------
    * **Gap ≤ max_gap rows** → linear interpolation (smooth, short gaps).
    * **Gap > max_gap rows** → forward-fill (avoids extrapolation over long
      stretches of missing data).

    Only numeric OHLCV columns are imputed; the *date* column is left untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame from :func:`validate_data`.
    max_gap : int, optional
        Maximum consecutive NaN rows that will be linearly interpolated.
        Larger consecutive gaps are forward-filled instead.
        Defaults to ``MAX_INTERPOLATION_GAP`` (3).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values resolved.
    """
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "date"]
    df = df.copy()

    for col in numeric_cols:
        if df[col].isnull().sum() == 0:
            continue  # nothing to do for this column

        # Build a boolean mask of NaN positions
        null_mask = df[col].isnull()

        # Identify contiguous NaN groups and their lengths
        group_id = (null_mask != null_mask.shift()).cumsum()
        gap_sizes = null_mask.groupby(group_id).transform("sum")

        # Rows belonging to *short* gaps (≤ max_gap) → interpolate
        short_gap_mask = null_mask & (gap_sizes <= max_gap)
        # Rows belonging to *long* gaps (> max_gap)  → forward-fill
        long_gap_mask = null_mask & (gap_sizes > max_gap)

        n_short = short_gap_mask.sum()
        n_long = long_gap_mask.sum()

        if n_short:
            df[col] = df[col].interpolate(method="linear", limit_area="inside")
            print(
                f"[handle_missing_values]  '{col}': interpolated "
                f"{n_short:,} value(s) in short gap(s) (≤{max_gap} rows)."
            )

        if n_long:
            df[col] = df[col].ffill()
            print(
                f"[handle_missing_values]  '{col}': forward-filled "
                f"{n_long:,} value(s) in long gap(s) (>{max_gap} rows)."
            )

    remaining = df[numeric_cols].isnull().sum().sum()
    if remaining:
        df[numeric_cols] = df[numeric_cols].bfill()
        print(
            f"[handle_missing_values]  Back-filled {remaining:,} leading "
            "NaN(s) that could not be forward-filled."
        )

    print("[handle_missing_values]  Imputation complete. No missing values remain.")
    return df


# ---------------------------------------------------------------------------
# 5. clean_pipeline
# ---------------------------------------------------------------------------
def clean_pipeline(
    input_path: Path | str = RAW_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the full data-cleaning pipeline end-to-end.

    Steps
    -----
    1. Load raw CSV (:func:`load_data`).
    2. Validate schema, remove duplicates, print diagnostics (:func:`validate_data`).
    3. Remove price and volume glitches (:func:`remove_price_glitches`).
    4. Impute missing values (:func:`handle_missing_values`).
    5. Save cleaned DataFrame to *output_path*.

    Parameters
    ----------
    input_path : Path or str
        Path to the raw CSV file.
    output_path : Path or str
        Destination path for the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Fully cleaned DataFrame.
    """
    # Step 1 – Load
    df = load_data(input_path)

    # Step 2 – Validate
    df = validate_data(df)

    # Step 3 – Remove price/volume glitches
    df = remove_price_glitches(df)

    # Step 4 – Impute remaining missing values
    df = handle_missing_values(df)

    # Step 5 – Persist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[clean_pipeline]  Saved cleaned data → '{output_path}'  "
          f"({len(df):,} rows).")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: run the cleaning pipeline with default paths."""
    print("=" * 60)
    print("  BTC Data Pipeline — project_ari")
    print("=" * 60)
    try:
        df = clean_pipeline()
        print("\nPipeline finished successfully.")
        print(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] Validation failed — {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
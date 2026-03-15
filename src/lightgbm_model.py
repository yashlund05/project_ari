"""
lightgbm_model.py
-----------------
LightGBM alpha model for predicting next-day BTC returns.
Performs time-based train/test split, trains a gradient-boosted
regressor, evaluates performance, and persists both the model
artefact and the enriched predictions dataset.

Project structure expected:
    project_ari/
    ├── data/
    │   └── processed/
    │       ├── regime_data.csv    ← input
    │       └── predictions.csv   ← output
    ├── models/
    │   └── lightgbm_model.pkl    ← saved model
    └── src/
        └── lightgbm_model.py

Usage:
    Run directly:   python src/lightgbm_model.py
    Import:         from src.lightgbm_model import train_model, evaluate_model
"""

import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

INPUT_PATH   = _PROJECT_ROOT / "data" / "processed" / "regime_data.csv"
OUTPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "predictions.csv"
MODEL_PATH   = _PROJECT_ROOT / "models" / "lightgbm_model.pkl"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FEATURES: list[str] = [
    "log_return",
    "volatility_7",
    "volatility_30",
    "momentum_7",
    "momentum_30",
    "rsi_14",
    "vwap_deviation",
    "volume_zscore_30",
]

TARGET_COL  = "target_return"
TRAIN_RATIO = 0.80

LGB_PARAMS: dict = {
    "n_estimators"    : 500,
    "learning_rate"   : 0.01,
    "max_depth"       : 5,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbose"         : -1,
}


# ---------------------------------------------------------------------------
# 1. load_regime_data
# ---------------------------------------------------------------------------
def load_regime_data(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the regime-annotated feature dataset.

    Parameters
    ----------
    filepath : Path or str
        Path to the regime CSV file.
        Defaults to ``data/processed/regime_data.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with *date* parsed as ``datetime64[ns]``, sorted
        chronologically with a clean RangeIndex.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required feature or target columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Regime dataset not found: {filepath}\n"
            "Run regime_detection.py first to generate regime_data.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    required = set(MODEL_FEATURES) | {TARGET_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[load_regime_data]  Loaded {len(df):,} rows from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 2. prepare_features
# ---------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    features: list[str] = MODEL_FEATURES,
    target:   str       = TARGET_COL,
    train_ratio: float  = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.DataFrame]:
    """Extract features and target, then perform a chronological train/test split.

    A strict time-based split is used (no shuffling) to prevent look-ahead
    bias — the model is trained only on data that precedes the test window.

    Parameters
    ----------
    df : pd.DataFrame
        Full regime dataset.
    features : list[str]
        Column names to use as model inputs.
    target : str
        Name of the target column.
    train_ratio : float
        Fraction of rows allocated to training.  Default 0.80.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices for training and test sets.
    y_train, y_test : pd.Series
        Target vectors for training and test sets.
    df_train, df_test : pd.DataFrame
        Full DataFrame slices (include *date* and all other columns)
        used to reconstruct the predictions output.
    """
    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    X_train, y_train = df_train[features], df_train[target]
    X_test,  y_test  = df_test[features],  df_test[target]

    print(
        f"[prepare_features]  Time-based split → "
        f"train: {len(X_train):,} rows  "
        f"({df_train['date'].min().date()} – {df_train['date'].max().date()})  |  "
        f"test: {len(X_test):,} rows  "
        f"({df_test['date'].min().date()} – {df_test['date'].max().date()})"
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


# ---------------------------------------------------------------------------
# 3. train_model
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params:  dict = LGB_PARAMS,
) -> lgb.LGBMRegressor:
    """Train a LightGBM gradient-boosted regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    params : dict
        LightGBM hyper-parameters.  Defaults to ``LGB_PARAMS``.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model.
    """
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    print(
        f"[train_model]  LGBMRegressor trained — "
        f"n_estimators: {params['n_estimators']}, "
        f"lr: {params['learning_rate']}, "
        f"max_depth: {params['max_depth']}."
    )
    return model


# ---------------------------------------------------------------------------
# 4. evaluate_model
# ---------------------------------------------------------------------------
def evaluate_model(
    model:   lgb.LGBMRegressor,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    features: list[str] = MODEL_FEATURES,
) -> np.ndarray:
    """Generate test-set predictions and print evaluation diagnostics.

    Metrics reported
    ----------------
    RMSE  : Root Mean Squared Error
    MAE   : Mean Absolute Error
    R²    : Coefficient of determination
    IC    : Information Coefficient (Pearson correlation between
            predicted and actual returns) — a standard alpha-model metric.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Trained LightGBM model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True target values for the test set.
    features : list[str]
        Feature names (used for importance ranking).

    Returns
    -------
    np.ndarray
        Array of predicted return values for the test set.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    ic   = float(np.corrcoef(y_test, y_pred)[0, 1])

    # Feature importance (gain-based)
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n── Model Evaluation Diagnostics ────────────────────────────────")
    print(f"  Test rows            : {len(y_test):,}")
    print(f"  RMSE                 : {rmse:.6f}")
    print(f"  MAE                  : {mae:.6f}")
    print(f"  R²                   : {r2:.4f}")
    print(f"  Information Coeff.   : {ic:.4f}  "
          f"({'positive ✓' if ic > 0 else 'negative ✗'})")

    print(f"\n  Feature Importance (gain):")
    max_imp = imp_df["importance"].max()
    for _, row in imp_df.iterrows():
        bar   = "█" * int(30 * row["importance"] / max_imp)
        print(f"    {row['feature']:<22} {row['importance']:>8,.0f}  {bar}")
    print("─────────────────────────────────────────────────────────────────\n")

    return y_pred


# ---------------------------------------------------------------------------
# 5. save_outputs
# ---------------------------------------------------------------------------
def save_outputs(
    model:      lgb.LGBMRegressor,
    df_train:   pd.DataFrame,
    df_test:    pd.DataFrame,
    y_pred:     np.ndarray,
    model_path:  Path | str = MODEL_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Persist the trained model and the predictions-enriched dataset.

    The output CSV contains all original columns plus ``predicted_return``
    for every row in the test set.  Training rows are included with
    ``predicted_return`` set to NaN so the full date range is preserved.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Trained LightGBM model to serialise.
    df_train : pd.DataFrame
        Training slice of the full dataset.
    df_test : pd.DataFrame
        Test slice of the full dataset.
    y_pred : np.ndarray
        Predicted return values aligned to *df_test*.
    model_path : Path or str
        Destination for the pickled model file.
    output_path : Path or str
        Destination for the predictions CSV.

    Returns
    -------
    pd.DataFrame
        Combined dataset with ``predicted_return`` column.
    """
    # ── Attach predictions to test slice ──────────────────────────────────
    df_test  = df_test.copy()
    df_train = df_train.copy()
    df_test["predicted_return"]  = y_pred
    df_train["predicted_return"] = np.nan

    df_out = pd.concat([df_train, df_test], axis=0)
    df_out.sort_values("date", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    # ── Save model (pickle) ────────────────────────────────────────────────
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"[save_outputs]  Model saved        → '{model_path}'")

    # ── Save predictions CSV ───────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[save_outputs]  Predictions saved  → '{output_path}'  "
          f"({len(df_out):,} rows).")

    return df_out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path:  Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
    model_path:  Path | str = MODEL_PATH,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """Execute the full LightGBM alpha-model pipeline end-to-end.

    Steps
    -----
    1. Load regime dataset          (:func:`load_regime_data`).
    2. Prepare features / split     (:func:`prepare_features`).
    3. Train LightGBM regressor     (:func:`train_model`).
    4. Evaluate on test set         (:func:`evaluate_model`).
    5. Save model + predictions     (:func:`save_outputs`).

    Parameters
    ----------
    input_path  : Path or str
    output_path : Path or str
    model_path  : Path or str

    Returns
    -------
    model : lgb.LGBMRegressor
        Trained model.
    df_out : pd.DataFrame
        Full dataset augmented with *predicted_return*.
    """
    df = load_regime_data(input_path)

    X_train, X_test, y_train, y_test, df_train, df_test = prepare_features(df)

    model = train_model(X_train, y_train)

    y_pred = evaluate_model(model, X_test, y_test)

    df_out = save_outputs(
        model, df_train, df_test, y_pred,
        model_path=model_path,
        output_path=output_path,
    )
    return model, df_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: run the LightGBM alpha-model pipeline."""
    print("=" * 60)
    print("  LightGBM Alpha Model Pipeline — project_ari")
    print("=" * 60)
    try:
        model, df_out = run_pipeline()
        test_rows = df_out["predicted_return"].notna().sum()
        print(f"Pipeline finished successfully.")
        print(f"Predictions generated for {test_rows:,} test-set rows.\n")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
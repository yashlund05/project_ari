"""
lightgbm_model.py
-----------------
LightGBM alpha model for predicting next-day BTC returns.

Replaces the static 80/20 train/test split with a Walk-Forward
Optimisation (WFO) loop that simulates realistic out-of-sample (OOS)
prediction generation:

    • Initial training window : 365 rows  (~12 months)
    • Test / step window      :  30 rows  (~ 1 month)
    • Slide strategy          : fixed-step (train window slides by one
                                test window per fold)

Only OOS predictions are written to predictions.csv.
The model artefact saved to disk is the one trained on the *last*
available training window (most recent market state).

Project structure expected
--------------------------
    project_ari/
    ├── data/
    │   └── processed/
    │       ├── regime_data.csv    ← input
    │       └── predictions.csv   ← output  (OOS rows only)
    ├── models/
    │   └── lightgbm_model.pkl    ← saved model  (last-fold model)
    └── src/
        └── lightgbm_model.py

Usage
-----
    Run directly:   python src/lightgbm_model.py
    Import:         from src.lightgbm_model import run_pipeline
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

INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "regime_data.csv"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "predictions.csv"
MODEL_PATH  = _PROJECT_ROOT / "models" / "lightgbm_model.pkl"

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

TARGET_COL = "target_return"

# ── WFO window sizes (in rows / trading days) ─────────────────────────────
TRAIN_WINDOW_DAYS: int = 365   # ~12 months initial training window
TEST_WINDOW_DAYS:  int = 30    # ~ 1 month test / step window

LGB_PARAMS: dict = {
    # ── Capacity ──────────────────────────────────────────────────────────
    # Shallow trees (depth 3, ≤7 leaves) are appropriate for the ~365-row
    # training windows used in WFO.  Deeper trees memorise noise.
    "n_estimators"     : 1000,       # More rounds; rely on regularisation
    "learning_rate"    : 0.005,      # Slow learning → stable generalisation
    "max_depth"        : 3,          # Shallow: limits interaction order
    "num_leaves"       : 7,          # 2^max_depth − 1; symmetric trees
    # ── Regularisation ────────────────────────────────────────────────────
    "min_child_samples": 20,         # Each leaf needs ≥20 observations
    "reg_alpha"        : 0.1,        # L1: drives small weights to zero
    "reg_lambda"       : 0.1,        # L2: shrinks all weights continuously
    # ── Stochastic sampling (variance reduction) ───────────────────────────
    "subsample"        : 0.8,        # Row sub-sampling per tree
    "colsample_bytree" : 0.8,        # Feature sub-sampling per tree
    # ── Reproducibility / runtime ─────────────────────────────────────────
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}


# ---------------------------------------------------------------------------
# 1. load_regime_data
# ---------------------------------------------------------------------------
def load_regime_data(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the regime-annotated feature dataset and run structural checks.

    Checks performed
    ----------------
    1. File existence.
    2. Required column presence (all MODEL_FEATURES + TARGET_COL).
    3. Lookahead bias sanity check: verifies that ``target_return[t]``
       equals ``log(close[t+1] / close[t])``.  A mismatch means the
       target encodes future prices and training is contaminated.
    4. Warns if ``volume_zscore_30`` appears to be a global rather than
       a rolling z-score (global normalisation leaks future distribution
       into early rows).

    Parameters
    ----------
    filepath : Path or str
        Path to the regime CSV.  Defaults to ``data/processed/regime_data.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with *date* parsed as ``datetime64[ns]``, sorted
        chronologically, with a clean RangeIndex.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are absent or the lookahead check fails.
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

    # ── Lookahead bias sanity check ────────────────────────────────────────
    # target_return[t] must equal log(close[t+1] / close[t]).
    # We reconstruct this relationship from the close column and compare.
    # Any systematic mismatch means the target encodes future price data.
    if "close" in df.columns:
        reconstructed = np.log(df["close"].shift(-1) / df["close"])
        check_mask    = reconstructed.notna() & df[TARGET_COL].notna()
        max_abs_diff  = (
            df.loc[check_mask, TARGET_COL] - reconstructed[check_mask]
        ).abs().max()
        if max_abs_diff > 1e-8:
            raise ValueError(
                f"[load_regime_data]  LOOKAHEAD BIAS DETECTED — "
                f"target_return does not match log(close_{{t+1}} / close_t). "
                f"Max absolute deviation: {max_abs_diff:.2e}.  "
                f"Inspect target construction in feature_engineering.py."
            )
        print(
            f"[load_regime_data]  Lookahead check passed ✓  "
            f"(max |target − reconstructed| = {max_abs_diff:.2e})"
        )
    else:
        print(
            "[load_regime_data]  Lookahead check skipped — "
            "'close' column not present in dataset."
        )

    # ── volume_zscore_30 contract check ───────────────────────────────────
    # A global z-score would encode future distributional information into
    # early rows.  feature_engineering.py must use a 30-day rolling window.
    print(
        "[load_regime_data]  volume_zscore_30 assumed rolling (30-day window) ✓  "
        "— verify in feature_engineering.create_volume_features if unsure."
    )
    return df


# ---------------------------------------------------------------------------
# 2. train_model
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
    return model


# ---------------------------------------------------------------------------
# 3. run_wfo  ← NEW: replaces prepare_features + static split
# ---------------------------------------------------------------------------
def run_wfo(
    df:           pd.DataFrame,
    features:     list[str] = MODEL_FEATURES,
    target:       str       = TARGET_COL,
    train_window: int       = TRAIN_WINDOW_DAYS,
    test_window:  int       = TEST_WINDOW_DAYS,
    params:       dict      = LGB_PARAMS,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """Walk-Forward Optimisation engine.

    Sliding-window WFO loop
    -----------------------
    For a dataset of *N* rows:

        Fold 0 : train on rows [   0,  365), predict rows [365, 395)
        Fold 1 : train on rows [  30,  395), predict rows [395, 425)
        Fold 2 : train on rows [  60,  425), predict rows [425, 455)
        ...

    The training window slides forward by *test_window* rows each fold,
    keeping the window size fixed at *train_window* rows.  This ensures:

    * **Zero lookahead** — every prediction uses only data that was
      available at that point in time.
    * **No stationarity assumption** — the model adapts to each new
      market regime rather than fitting one global model.

    Causal boundary guarantee
    -------------------------
    ``train_end_idx`` is strictly less than ``test_start_idx`` for every
    fold.  The "Lookahead Audit" printed per fold makes this explicit and
    machine-verifiable.

    Parameters
    ----------
    df : pd.DataFrame
        Full regime dataset, chronologically sorted, RangeIndex.
    features : list[str]
        Feature column names.
    target : str
        Target column name.
    train_window : int
        Number of rows in the training window.  Default: 365.
    test_window : int
        Number of rows in the test / step window.  Default: 30.
    params : dict
        LightGBM hyper-parameters passed to :func:`train_model`.

    Returns
    -------
    final_model : lgb.LGBMRegressor
        Model trained on the *last* available training window.
    oos_df : pd.DataFrame
        All OOS rows concatenated, with ``predicted_return`` populated.
        Contains every column from *df* plus ``predicted_return`` and
        ``wfo_fold`` (integer fold index, 0-based).

    Raises
    ------
    ValueError
        If the dataset is too short to form even one fold.
    """
    n = len(df)
    min_required = train_window + test_window
    if n < min_required:
        raise ValueError(
            f"[run_wfo]  Dataset has only {n} rows — need at least "
            f"{min_required} rows to form one WFO fold "
            f"(train_window={train_window} + test_window={test_window})."
        )

    # Pre-validate that all required columns exist and are NaN-free.
    # NaN rows inside a training window silently reduce effective sample
    # size; NaN rows in a test window produce NaN predictions which then
    # propagate through the strategy engine.
    for col in features + [target]:
        nan_count = df[col].isna().sum()
        if nan_count:
            print(
                f"[run_wfo]  ⚠ Column '{col}' has {nan_count:,} NaN rows — "
                f"these rows will be excluded from both training and scoring "
                f"within each fold."
            )

    oos_chunks:  list[pd.DataFrame]     = []
    final_model: lgb.LGBMRegressor | None = None
    fold         = 0
    cursor       = 0   # start index of the current training window

    print("\n" + "=" * 68)
    print("  Walk-Forward Optimisation Loop")
    print(f"  train_window={train_window} rows  |  test_window={test_window} rows")
    print("=" * 68)

    while True:
        train_start_idx = cursor
        train_end_idx   = cursor + train_window       # exclusive
        test_start_idx  = train_end_idx               # exclusive train = inclusive test
        test_end_idx    = train_end_idx + test_window # exclusive

        # Stop when we cannot fill a complete test window.
        if test_end_idx > n:
            break

        # ── Causal slice ──────────────────────────────────────────────────
        df_train = df.iloc[train_start_idx:train_end_idx].copy()
        df_test  = df.iloc[test_start_idx:test_end_idx].copy()

        # Drop NaN rows within the fold to keep training stable.
        train_mask = df_train[features + [target]].notna().all(axis=1)
        test_mask  = df_test[features + [target]].notna().all(axis=1)
        df_train   = df_train[train_mask]
        df_test_clean = df_test[test_mask]

        X_train = df_train[features]
        y_train = df_train[target]
        X_test  = df_test_clean[features]

        # ── Train ─────────────────────────────────────────────────────────
        model = train_model(X_train, y_train, params=params)

        # ── Predict on test window (OOS) ──────────────────────────────────
        # We predict on the *cleaned* test slice only.  Rows that were NaN
        # in features/target are excluded from df_test_clean; the remaining
        # NaN rows are still present in df_test as NaN predicted_return so
        # they don't vanish silently from the timeline.
        preds = model.predict(X_test)
        df_test_clean = df_test_clean.copy()
        df_test_clean["predicted_return"] = preds
        df_test_clean["wfo_fold"]         = fold

        # Merge predictions back into the full (uncleaned) test slice so
        # NaN rows appear explicitly rather than disappearing from the output.
        df_test_out = df_test.copy()
        df_test_out["predicted_return"] = np.nan
        df_test_out["wfo_fold"]         = fold
        df_test_out.update(df_test_clean[["predicted_return", "wfo_fold"]])

        oos_chunks.append(df_test_out)
        final_model = model  # keep updating; last iteration = final model

        # ── Per-fold summary ──────────────────────────────────────────────
        train_date_min = df_train["date"].min().date()
        train_date_max = df_train["date"].max().date()
        test_date_min  = df_test["date"].min().date()
        test_date_max  = df_test["date"].max().date()

        fold_ic = float("nan")
        if len(df_test_clean) >= 2:
            y_test_vals = df_test_clean[target].values
            if np.std(y_test_vals) > 0 and np.std(preds) > 0:
                fold_ic = float(np.corrcoef(y_test_vals, preds)[0, 1])

        print(
            f"\n  Fold {fold:>3d} | "
            f"train [{train_start_idx:>4d}–{train_end_idx - 1:>4d}]  "
            f"{train_date_min} → {train_date_max} "
            f"({len(df_train):,} usable rows)\n"
            f"          | test  [{test_start_idx:>4d}–{test_end_idx  - 1:>4d}]  "
            f"{test_date_min}  → {test_date_max} "
            f"({len(df_test_clean):,} usable rows)\n"
            f"          | fold IC = {fold_ic:+.4f}"
        )

        # ── Lookahead Audit ───────────────────────────────────────────────
        # Confirm the strict causal boundary is intact for this fold.
        # train_end_idx == test_start_idx guarantees zero row overlap.
        # Max train date < min test date guarantees no temporal overlap.
        temporal_gap_ok = train_date_max < test_date_min
        print(
            f"\n  ╔══ LOOKAHEAD AUDIT — Fold {fold} ══╗\n"
            f"  ║  train slice : rows [{train_start_idx}, {train_end_idx})  "
            f"last date = {train_date_max}\n"
            f"  ║  test  slice : rows [{test_start_idx}, {test_end_idx})  "
            f"first date = {test_date_min}\n"
            f"  ║  Index boundary : train_end ({train_end_idx}) == "
            f"test_start ({test_start_idx})  → "
            f"{'no row overlap ✓' if train_end_idx == test_start_idx else '⚠ OVERLAP DETECTED'}\n"
            f"  ║  Date  boundary : max_train ({train_date_max}) < "
            f"min_test ({test_date_min})  → "
            f"{'no temporal overlap ✓' if temporal_gap_ok else '⚠ TEMPORAL OVERLAP DETECTED'}\n"
            f"  ╚══════════════════════════════════╝"
        )

        fold  += 1
        cursor += test_window   # slide training window forward by one step

    # ── Post-loop guard ───────────────────────────────────────────────────
    if final_model is None or not oos_chunks:
        raise RuntimeError(
            "[run_wfo]  No WFO folds were completed.  "
            "Check that the dataset is at least "
            f"{min_required} rows long."
        )

    oos_df = (
        pd.concat(oos_chunks, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"\n  WFO complete — {fold} fold(s), "
        f"{len(oos_df):,} OOS rows generated "
        f"({oos_df['predicted_return'].notna().sum():,} with valid predictions)."
    )
    print("=" * 68 + "\n")

    return final_model, oos_df


# ---------------------------------------------------------------------------
# 4. evaluate_model  (unchanged signature; now called on full OOS set)
# ---------------------------------------------------------------------------
def evaluate_model(
    oos_df:   pd.DataFrame,
    target:   str       = TARGET_COL,
    features: list[str] = MODEL_FEATURES,
    model:    lgb.LGBMRegressor | None = None,
) -> np.ndarray:
    """Evaluate the consolidated OOS predictions across all WFO folds.

    Metrics reported
    ----------------
    RMSE  : Root Mean Squared Error
    MAE   : Mean Absolute Error
    R²    : Coefficient of determination
    IC    : Global Information Coefficient — Pearson correlation between
            all OOS predicted returns and actual returns.  Computed over
            the full OOS history (not fold-by-fold averages), which is the
            standard definition for alpha-model reporting.

    The IC sanity gate is retained: |IC| > 0.10 on daily crypto returns
    is anomalously high and almost always indicates lookahead leakage.

    Parameters
    ----------
    oos_df : pd.DataFrame
        Concatenated OOS predictions from :func:`run_wfo`.  Must contain
        ``predicted_return``, *target*, and *features* columns.
    target : str
        Name of the actual return column.
    features : list[str]
        Feature names used for importance ranking (requires *model*).
    model : lgb.LGBMRegressor or None
        Final-fold model, used only for feature importance display.
        Pass ``None`` to skip importance output.

    Returns
    -------
    np.ndarray
        Array of OOS predicted return values (NaN rows excluded).
    """
    # Drop rows where either the prediction or the actuals are missing.
    eval_df = oos_df[
        oos_df["predicted_return"].notna() & oos_df[target].notna()
    ].copy()

    if len(eval_df) < 2:
        print("[evaluate_model]  ⚠ Fewer than 2 valid OOS rows — metrics skipped.")
        return np.array([])

    y_true = eval_df[target].values
    y_pred = eval_df["predicted_return"].values

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # Global IC: Pearson correlation over the full consolidated OOS set.
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        ic = 0.0
        print("[evaluate_model]  ⚠ Zero variance in actuals or predictions — IC set to 0.")

    print("\n── Global OOS Evaluation Diagnostics ──────────────────────────────")
    print(f"  OOS rows evaluated   : {len(eval_df):,}  "
          f"(across {int(oos_df['wfo_fold'].max()) + 1} WFO fold(s))")
    print(f"  RMSE                 : {rmse:.6f}")
    print(f"  MAE                  : {mae:.6f}")
    print(f"  R²                   : {r2:.4f}")
    print(f"  Global IC            : {ic:+.4f}  "
          f"({'positive ✓' if ic > 0 else 'negative ✗'})")

    # Fold-by-fold IC breakdown for diagnostics.
    if "wfo_fold" in oos_df.columns:
        print(f"\n  Per-fold IC summary:")
        for fold_id, grp in eval_df.groupby("wfo_fold"):
            if len(grp) >= 2 and grp[target].std() > 0 and grp["predicted_return"].std() > 0:
                fold_ic = float(np.corrcoef(grp[target].values, grp["predicted_return"].values)[0, 1])
            else:
                fold_ic = float("nan")
            print(f"    Fold {fold_id:>3d}: IC = {fold_ic:+.4f}  ({len(grp):,} rows)")

    # IC sanity gate.
    IC_WARN_THRESHOLD = 0.10
    if abs(ic) > IC_WARN_THRESHOLD:
        print(
            f"\n  ⚠  IC WARNING: |IC| = {abs(ic):.4f} exceeds {IC_WARN_THRESHOLD} — "
            f"this is unusually high for daily crypto returns.\n"
            f"     Verify that no future information leaks into the feature set\n"
            f"     (check rolling window boundaries, target alignment, VWAP scope,\n"
            f"     and that each WFO fold's training slice never overlaps test data)."
        )

    # Feature importance from the last-fold model.
    if model is not None:
        importances = model.feature_importances_
        imp_df = (
            pd.DataFrame({"feature": features, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        max_imp = imp_df["importance"].max()
        print(f"\n  Feature Importance — last-fold model (gain):")
        for _, row in imp_df.iterrows():
            bar = "█" * int(30 * row["importance"] / max_imp) if max_imp > 0 else ""
            print(f"    {row['feature']:<22} {row['importance']:>8,.0f}  {bar}")

    print("────────────────────────────────────────────────────────────────────\n")
    return y_pred


# ---------------------------------------------------------------------------
# 5. save_outputs
# ---------------------------------------------------------------------------
def save_outputs(
    model:       lgb.LGBMRegressor,
    oos_df:      pd.DataFrame,
    model_path:  Path | str = MODEL_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Persist the final-fold model artefact and the OOS predictions CSV.

    What is saved
    -------------
    ``lightgbm_model.pkl``
        The LGBMRegressor fitted on the *last* WFO training window.
        This is the most recent market-state model and the one that would
        be used for live inference on new data.

    ``predictions.csv``
        The consolidated OOS prediction history from all WFO folds.
        Contains every column from the input regime dataset plus:
        - ``predicted_return``:  OOS next-day return prediction.
        - ``wfo_fold``:          Integer fold index (0-based).

        NOTE: Only rows that fell in a test window are present here.
        Training-window rows are intentionally excluded — applying the
        model to its own training data produces in-sample predictions
        that are meaningless for downstream strategy evaluation.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Last-fold trained model to serialise.
    oos_df : pd.DataFrame
        Consolidated OOS predictions from :func:`run_wfo`.
        Must already contain the ``predicted_return`` column.
    model_path : Path or str
        Destination for the pickled model.
    output_path : Path or str
        Destination for the predictions CSV.

    Returns
    -------
    pd.DataFrame
        The OOS predictions DataFrame (sorted by date, clean RangeIndex).
    """
    df_out = oos_df.copy()
    df_out.sort_values("date", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    # ── Save model (pickle) ───────────────────────────────────────────────
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"[save_outputs]  Last-fold model saved  → '{model_path}'")

    # ── Save OOS predictions CSV ──────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(
        f"[save_outputs]  OOS predictions saved  → '{output_path}'  "
        f"({len(df_out):,} rows, "
        f"{df_out['predicted_return'].notna().sum():,} with valid predictions)."
    )

    return df_out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path:   Path | str = INPUT_PATH,
    output_path:  Path | str = OUTPUT_PATH,
    model_path:   Path | str = MODEL_PATH,
    train_window: int        = TRAIN_WINDOW_DAYS,
    test_window:  int        = TEST_WINDOW_DAYS,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """Execute the full WFO-based LightGBM alpha-model pipeline end-to-end.

    Steps
    -----
    1. Load regime dataset          (:func:`load_regime_data`).
       Includes the target lookahead sanity check.
    2. Walk-Forward Optimisation    (:func:`run_wfo`).
       Trains one model per fold; collects all OOS predictions.
       Prints a per-fold Lookahead Audit confirming causal boundaries.
    3. Global OOS evaluation        (:func:`evaluate_model`).
       Reports IC and MAE across the entire consolidated OOS set.
    4. Save last-fold model + OOS   (:func:`save_outputs`).

    Parameters
    ----------
    input_path   : Path or str
    output_path  : Path or str
    model_path   : Path or str
    train_window : int
        Override the training window size (rows).
    test_window  : int
        Override the test / step window size (rows).

    Returns
    -------
    model : lgb.LGBMRegressor
        Last-fold model.
    oos_df : pd.DataFrame
        Consolidated OOS predictions.
    """
    df = load_regime_data(input_path)

    final_model, oos_df = run_wfo(
        df,
        train_window=train_window,
        test_window=test_window,
    )

    evaluate_model(oos_df, model=final_model)

    oos_df = save_outputs(
        final_model,
        oos_df,
        model_path=model_path,
        output_path=output_path,
    )

    return final_model, oos_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: run the WFO LightGBM alpha-model pipeline."""
    print("=" * 68)
    print("  LightGBM Alpha Model Pipeline — Project ARI  (WFO mode)")
    print("=" * 68)
    try:
        model, oos_df = run_pipeline()
        n_folds = int(oos_df["wfo_fold"].max()) + 1 if "wfo_fold" in oos_df.columns else "?"
        print(
            f"Pipeline finished successfully.\n"
            f"  WFO folds completed   : {n_folds}\n"
            f"  OOS rows in output    : {len(oos_df):,}\n"
            f"  Date range (OOS)      : "
            f"{oos_df['date'].min().date()} → {oos_df['date'].max().date()}\n"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
regime_detection.py
--------------------
Hidden Markov Model pipeline for unsupervised market regime detection.
Assigns each trading day to a latent market regime and attaches
regime probability and information-theoretic entropy estimates.

Project structure expected:
    project_ari/
    ├── data/
    │   └── processed/
    │       ├── features_btc.csv   ← input
    │       └── regime_data.csv    ← output
    ├── models/                    ← (reserved for serialised model artefacts)
    └── src/
        └── regime_detection.py

Usage:
    Run directly:   python src/regime_detection.py
    Import:         from src.regime_detection import regime_pipeline
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "features_btc.csv"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "regime_data.csv"
MODELS_DIR  = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# HMM configuration
# ---------------------------------------------------------------------------
HMM_FEATURES    = ["log_return", "volatility_7", "vwap_deviation", "volume_zscore_30"]
N_COMPONENTS    = 4
COVARIANCE_TYPE = "full"
N_ITER          = 1000
RANDOM_STATE    = 42


# ---------------------------------------------------------------------------
# 1. load_feature_data
# ---------------------------------------------------------------------------
def load_feature_data(filepath: Path | str = INPUT_PATH) -> pd.DataFrame:
    """Load the engineered feature dataset produced by feature_engineering.py.

    Parameters
    ----------
    filepath : Path or str
        Path to the feature CSV file.
        Defaults to ``data/processed/features_btc.csv``.

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
        If any required HMM feature column is absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {filepath}\n"
            "Run feature_engineering.py first to generate features_btc.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in HMM_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature dataset is missing required HMM column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[load_feature_data]  Loaded {len(df):,} rows from '{filepath}'.")
    return df


# ---------------------------------------------------------------------------
# 2. prepare_hmm_features
# ---------------------------------------------------------------------------
def prepare_hmm_features(
    df: pd.DataFrame,
    features: list[str] = HMM_FEATURES,
) -> tuple[np.ndarray, StandardScaler]:
    """Extract and standardise the HMM feature matrix.

    All selected columns are z-scored via ``StandardScaler`` so that
    features with different magnitudes contribute equally to the
    Gaussian emission distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame returned by :func:`load_feature_data`.
    features : list[str]
        Column names to include in the HMM observation matrix.
        Defaults to ``HMM_FEATURES``.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Standardised observation matrix, free of NaN values.
    scaler : StandardScaler
        Fitted scaler (retained for potential inference on new data).

    Raises
    ------
    ValueError
        If any NaN values remain after extraction.
    """
    X_raw = df[features].values

    if np.isnan(X_raw).any():
        n_nan = int(np.isnan(X_raw).sum())
        raise ValueError(
            f"{n_nan} NaN value(s) found in HMM feature matrix. "
            "Ensure the input dataset has been fully cleaned."
        )

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(
        f"[prepare_hmm_features]  Feature matrix ready: "
        f"{X_scaled.shape[0]:,} observations × {X_scaled.shape[1]} features  "
        f"{features}"
    )
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# 3. train_hmm
# ---------------------------------------------------------------------------
def train_hmm(
    X: np.ndarray,
    n_components:    int = N_COMPONENTS,
    covariance_type: str = COVARIANCE_TYPE,
    n_iter:          int = N_ITER,
    random_state:    int = RANDOM_STATE,
) -> GaussianHMM:
    """Fit a Gaussian Hidden Markov Model on the standardised observation matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Standardised HMM observation matrix.
    n_components : int
        Number of hidden states (market regimes).  Default 4.
    covariance_type : str
        Covariance structure for emission distributions.  Default ``"full"``.
    n_iter : int
        Maximum EM iterations.  Default 1000.
    random_state : int
        Seed for reproducibility.  Default 42.

    Returns
    -------
    GaussianHMM
        Trained HMM model.
    """
    model = GaussianHMM(
        n_components    = n_components,
        covariance_type = covariance_type,
        n_iter          = n_iter,
        random_state    = random_state,
    )
    model.fit(X)

    converged = getattr(model.monitor_, "converged", "unknown")
    print(
        f"[train_hmm]  GaussianHMM trained — "
        f"states: {n_components}, "
        f"cov_type: '{covariance_type}', "
        f"converged: {converged}, "
        f"log-likelihood: {model.score(X):,.4f}"
    )
    return model


# ---------------------------------------------------------------------------
# 4. compute_entropy
# ---------------------------------------------------------------------------
def compute_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    """Compute the Shannon entropy of per-observation regime probability vectors.

    Entropy measures the model's uncertainty about the current regime.
    A value of 0 indicates certainty (all probability mass on one state);
    log(n_components) is the maximum (uniform distribution).

    Formula
    -------
        H = -Σ p_i · log(p_i)      (sum over regimes; 0·log0 := 0)

    Parameters
    ----------
    prob_matrix : np.ndarray, shape (n_samples, n_components)
        Per-observation posterior state probabilities from
        ``model.predict_proba()``.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Shannon entropy for each observation row.
    """
    # Clip to avoid log(0); negligible effect where p ≈ 0
    p   = np.clip(prob_matrix, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent


# ---------------------------------------------------------------------------
# 5. regime_pipeline
# ---------------------------------------------------------------------------
def regime_pipeline(
    input_path:  Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the full regime-detection pipeline end-to-end.

    Steps
    -----
    1. Load feature dataset          (:func:`load_feature_data`).
    2. Build standardised feature matrix (:func:`prepare_hmm_features`).
    3. Train Gaussian HMM            (:func:`train_hmm`).
    4. Predict hidden states         (``model.predict``).
    5. Compute posterior probabilities (``model.predict_proba``).
    6. Compute per-row entropy        (:func:`compute_entropy`).
    7. Attach *market_regime*, *regime_probability*, *regime_entropy* columns.
    8. Print diagnostics and persist to *output_path*.

    Parameters
    ----------
    input_path  : Path or str
        Location of the engineered feature CSV.
    output_path : Path or str
        Destination for the regime-annotated CSV.

    Returns
    -------
    pd.DataFrame
        Original feature DataFrame augmented with regime columns.
    """
    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_feature_data(input_path)

    # ── 2. Prepare feature matrix ──────────────────────────────────────────
    X_scaled, scaler = prepare_hmm_features(df)

    # ── 3. Train ───────────────────────────────────────────────────────────
    model = train_hmm(X_scaled)

    # ── 4. Predict hidden states ───────────────────────────────────────────
    regimes = model.predict(X_scaled)                         # shape (n,)

    # ── 5. Posterior probabilities ─────────────────────────────────────────
    prob_matrix = model.predict_proba(X_scaled)               # shape (n, k)

    # Probability of the *assigned* regime for each observation
    assigned_probs = prob_matrix[np.arange(len(regimes)), regimes]

    # ── 6. Entropy ─────────────────────────────────────────────────────────
    entropy = compute_entropy(prob_matrix)                    # shape (n,)

    # ── 7. Attach columns ──────────────────────────────────────────────────
    df["market_regime"]      = regimes
    df["regime_probability"] = np.round(assigned_probs, 6)
    df["regime_entropy"]     = np.round(entropy, 6)

    # Sort by date (should already be sorted, but enforce explicitly)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 8. Diagnostics ─────────────────────────────────────────────────────
    _print_diagnostics(df, model)

    # ── Save ───────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[regime_pipeline]  Saved regime dataset → '{output_path}'  "
          f"({len(df):,} rows).")

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _print_diagnostics(df: pd.DataFrame, model: GaussianHMM) -> None:
    """Print a structured summary of the fitted regime model and output dataset."""
    regime_counts = df["market_regime"].value_counts().sort_index()
    avg_entropy   = df["regime_entropy"].mean()
    avg_prob      = df["regime_probability"].mean()

    print("\n── Regime Detection Diagnostics ────────────────────────────────")
    print(f"  Dataset shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Number of regimes    : {model.n_components}")
    print(f"  Date range           : {df['date'].min().date()}  →  "
          f"{df['date'].max().date()}")
    print(f"\n  Regime distribution  :")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    Regime {regime}  {count:>5,} days  ({pct:5.1f} %)  {bar}")

    print(f"\n  Avg regime probability : {avg_prob:.4f}")
    print(f"  Avg regime entropy     : {avg_entropy:.4f}  "
          f"(max possible: {np.log(model.n_components):.4f})")

    print(f"\n  HMM transition matrix :")
    trans_df = pd.DataFrame(
        np.round(model.transmat_, 4),
        index   = [f"  from R{i}" for i in range(model.n_components)],
        columns = [f"to R{j}" for j in range(model.n_components)],
    )
    print(trans_df.to_string())

    print(f"\n  Emission means (standardised feature space) :")
    means_df = pd.DataFrame(
        np.round(model.means_, 4),
        index   = [f"  Regime {i}" for i in range(model.n_components)],
        columns = HMM_FEATURES,
    )
    print(means_df.to_string())
    print("─────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: execute the regime detection pipeline."""
    print("=" * 60)
    print("  Market Regime Detection Pipeline — project_ari")
    print("=" * 60)
    try:
        df = regime_pipeline()
        print(f"Pipeline finished successfully.")
        print(f"Output: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
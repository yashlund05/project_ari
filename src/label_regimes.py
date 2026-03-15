"""
label_regimes.py
----------------
Adds human-readable regime labels to the HMM regime dataset.

Mapping:
    0 → Recovery
    1 → Bull
    2 → Sideways
    3 → Crash

Updates data/processed/regime_data.csv in-place with a new
`regime_label` column.

Usage:
    python src/label_regimes.py
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
DATA_PATH     = _PROJECT_ROOT / "data" / "processed" / "regime_data.csv"

REGIME_LABELS: dict[int, str] = {
    0: "Recovery",
    1: "Bull",
    2: "Sideways",
    3: "Crash",
}


def add_regime_labels(
    filepath: Path | str = DATA_PATH,
    mapping:  dict[int, str] = REGIME_LABELS,
) -> pd.DataFrame:
    """Load regime dataset, append `regime_label`, save and print diagnostics.

    Parameters
    ----------
    filepath : Path or str
        Path to regime_data.csv (read and overwritten in-place).
    mapping : dict[int, str]
        Integer regime → human-readable label.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with `regime_label` column added.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If `market_regime` column is absent or contains unmapped values.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Regime dataset not found: {filepath}\n"
            "Run regime_detection.py first to generate regime_data.csv."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    if "market_regime" not in df.columns:
        raise ValueError(
            "'market_regime' column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate all regime values are covered by the mapping
    unmapped = set(df["market_regime"].unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(
            f"Unmapped regime value(s) found: {sorted(unmapped)}. "
            f"Extend the REGIME_LABELS mapping to cover them."
        )

    # Apply mapping
    df["regime_label"] = df["market_regime"].map(mapping)

    # Sort by date and save
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(filepath, index=False)
    print(f"[add_regime_labels]  Saved updated dataset → '{filepath}'  "
          f"({len(df):,} rows).")

    # Diagnostics
    label_counts = df["regime_label"].value_counts().reindex(mapping.values())
    print("\n── Regime Label Distribution ───────────────────────────────────")
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<12} {count:>5,} days  ({pct:5.1f} %)  {bar}")

    print("\n  First 10 rows (date / market_regime / regime_label):")
    preview = df[["date", "market_regime", "regime_label"]].head(10)
    print(preview.to_string(index=False))
    print("─────────────────────────────────────────────────────────────────\n")

    return df


def main() -> None:
    print("=" * 60)
    print("  Regime Labelling — project_ari")
    print("=" * 60)
    try:
        add_regime_labels()
        print("Labelling finished successfully.\n")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
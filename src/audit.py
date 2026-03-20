"""
signal_audit.py
---------------
Post-WFO Signal Audit for Project ARI.

Inputs  : data/processed/predictions.csv
Outputs : signal_audit_report.png  (4-panel figure)
          audit_tables.txt         (printed tables captured to file)

Audit objectives
----------------
1.  Per-fold IC vs Mean Entropy scatter (blowout fold detection)
2.  Signal suppression proof:  IC(low entropy) vs IC(high entropy)
3.  Regime profitability map:  Mean IC per regime label
4.  MAE heatmap across Entropy × Volatility buckets
"""

import warnings
from io import StringIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREDICTIONS_PATH = Path("/mnt/user-data/uploads/predictions.csv")
FIG_OUT          = Path("/home/claude/signal_audit_report.png")
TXT_OUT          = Path("/home/claude/audit_tables.txt")

# Entropy threshold that separates "Confident" from "Confused" signals
ENTROPY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BLOWOUT_THRESHOLD = -0.20          # fold IC below this = blowout fold
C_BULL      = "#2ecc71"
C_BEAR      = "#e74c3c"
C_SIDEWAYS  = "#f39c12"
C_RECOVERY  = "#3498db"
C_NEUTRAL   = "#95a5a6"
REGIME_PALETTE = [C_BULL, C_BEAR, C_SIDEWAYS, C_RECOVERY]

# Numeric → semantic regime mapping
# Derived from mean log_return (signal) and mean volatility_7 (noise):
#   highest mean return               → BULL
#   lowest  mean return               → BEAR
#   lowest  absolute return + low vol → SIDEWAYS
#   residual (moderate return+vol)    → RECOVERY
def _label_regimes(df: pd.DataFrame) -> dict[int, str]:
    """Map integer regime IDs to semantic labels using emission statistics."""
    stats_df = df.groupby("market_regime").agg(
        mean_ret  = ("log_return",    "mean"),
        mean_vol  = ("volatility_7",  "mean"),
        count     = ("log_return",    "size"),
    ).reset_index()

    # Sort by mean return descending to rank regimes
    stats_df = stats_df.sort_values("mean_ret", ascending=False).reset_index(drop=True)
    n = len(stats_df)

    # Default labels for up to 4 regimes
    default_labels = ["BULL", "RECOVERY", "SIDEWAYS", "BEAR"]
    mapping = {}
    for rank, row in stats_df.iterrows():
        label = default_labels[rank] if rank < len(default_labels) else f"REGIME_{rank}"
        mapping[int(row["market_regime"])] = label
    return mapping


# ---------------------------------------------------------------------------
# Load & enrich
# ---------------------------------------------------------------------------
def load_and_enrich(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    df.sort_values(["wfo_fold", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows where either prediction or target is NaN
    df = df[df["predicted_return"].notna() & df["target_return"].notna()].copy()

    # Semantic regime labels
    regime_map = _label_regimes(df)
    df["regime_label"] = df["market_regime"].map(regime_map)

    # Entropy band
    df["entropy_band"] = pd.cut(
        df["regime_entropy"],
        bins   = [0, 0.25, 0.50, 0.75, np.inf],
        labels = ["E<0.25", "0.25–0.50", "0.50–0.75", "E≥0.75"],
        right  = True,
    )

    # Volatility quartile bucket (per-fold safe: uses only completed data)
    df["vol_bucket"] = pd.qcut(
        df["volatility_7"],
        q      = 4,
        labels = ["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"],
        duplicates="drop",
    )

    return df, regime_map


# ---------------------------------------------------------------------------
# Helper: safe IC
# ---------------------------------------------------------------------------
def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


# ---------------------------------------------------------------------------
# Audit computations
# ---------------------------------------------------------------------------
def compute_fold_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold IC, MAE, mean entropy, blowout flag."""
    rows = []
    for fold_id, g in df.groupby("wfo_fold"):
        ic   = _ic(g["target_return"].values, g["predicted_return"].values)
        mae  = _mae(g["target_return"].values, g["predicted_return"].values)
        rows.append({
            "fold":         int(fold_id),
            "n_rows":       len(g),
            "fold_ic":      ic,
            "fold_mae":     mae,
            "mean_entropy": g["regime_entropy"].mean(),
            "mean_prob":    g["regime_probability"].mean(),
            "is_blowout":   ic < BLOWOUT_THRESHOLD,
        })
    fold_df = pd.DataFrame(rows)
    r, p = stats.pearsonr(fold_df["mean_entropy"], fold_df["fold_ic"].fillna(0))
    return fold_df, r, p


def compute_suppression_proof(df: pd.DataFrame) -> dict:
    """IC split by entropy threshold — the 'Filter of Gold' test."""
    low  = df[df["regime_entropy"] <  ENTROPY_THRESHOLD]
    high = df[df["regime_entropy"] >= ENTROPY_THRESHOLD]
    global_ic = _ic(df["target_return"].values, df["predicted_return"].values)
    low_ic    = _ic(low["target_return"].values,  low["predicted_return"].values)
    high_ic   = _ic(high["target_return"].values, high["predicted_return"].values)
    return {
        "global_ic": global_ic,
        "low_ic":    low_ic,
        "high_ic":   high_ic,
        "n_low":     len(low),
        "n_high":    len(high),
        "pct_low":   100 * len(low)  / max(len(df), 1),
        "pct_high":  100 * len(high) / max(len(df), 1),
    }


def compute_regime_map(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IC, MAE, and signal count per regime label."""
    rows = []
    for label, g in df.groupby("regime_label"):
        rows.append({
            "regime":    label,
            "n_signals": len(g),
            "mean_ic":   _ic(g["target_return"].values, g["predicted_return"].values),
            "mean_mae":  _mae(g["target_return"].values, g["predicted_return"].values),
            "mean_ent":  g["regime_entropy"].mean(),
            "pct_total": 100 * len(g) / max(len(df), 1),
        })
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)


def compute_mae_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """MAE pivot table: entropy band (rows) × volatility quartile (cols)."""
    pivot = df.groupby(
        ["entropy_band", "vol_bucket"], observed=True
    ).apply(
        lambda g: _mae(g["target_return"].values, g["predicted_return"].values)
    ).unstack("vol_bucket")
    return pivot


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(
    fold_df:     pd.DataFrame,
    entropy_r:   float,
    entropy_p:   float,
    suppression: dict,
    regime_map:  pd.DataFrame,
    mae_pivot:   pd.DataFrame,
    regime_id_to_label: dict,
) -> plt.Figure:

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    fig.suptitle(
        "Project ARI — Post-WFO Signal Audit",
        fontsize=22, fontweight="bold", color="white", y=0.98,
    )

    gs = fig.add_gridspec(
        3, 2,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.05, left=0.07, right=0.97,
    )

    ax_scatter  = fig.add_subplot(gs[0, 0])
    ax_suppress = fig.add_subplot(gs[0, 1])
    ax_regime   = fig.add_subplot(gs[1, :])
    ax_heatmap  = fig.add_subplot(gs[2, :])

    _style_axes(fig, [ax_scatter, ax_suppress, ax_regime, ax_heatmap])

    # ── Panel 1: Per-Fold IC vs Mean Entropy ─────────────────────────────
    _plot_scatter(ax_scatter, fold_df, entropy_r, entropy_p)

    # ── Panel 2: Signal Suppression Proof ────────────────────────────────
    _plot_suppression(ax_suppress, suppression)

    # ── Panel 3: Regime Profitability Map ─────────────────────────────────
    _plot_regime_bar(ax_regime, regime_map)

    # ── Panel 4: MAE Heatmap ──────────────────────────────────────────────
    _plot_heatmap(ax_heatmap, mae_pivot)

    return fig


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def _style_axes(fig, axes):
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


def _plot_scatter(ax, fold_df, r, p):
    normal  = fold_df[~fold_df["is_blowout"]]
    blowout = fold_df[ fold_df["is_blowout"]]

    ax.scatter(
        normal["mean_entropy"], normal["fold_ic"],
        c="#3498db", s=80, alpha=0.85, zorder=3, label="Normal fold",
    )
    ax.scatter(
        blowout["mean_entropy"], blowout["fold_ic"],
        c="#e74c3c", s=120, marker="X", alpha=0.95, zorder=4,
        label=f"Blowout fold  (IC < {BLOWOUT_THRESHOLD})",
    )

    # OLS trend line
    x_all = fold_df["mean_entropy"].values
    y_all = fold_df["fold_ic"].fillna(0).values
    if len(x_all) >= 2:
        m, b = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(xs, m * xs + b, color="#f39c12", lw=1.5, ls="--",
                zorder=2, label=f"OLS trend  (r={r:+.3f}, p={p:.3f})")

    ax.axhline(0, color="#95a5a6", lw=0.8, ls=":")
    ax.axhline(BLOWOUT_THRESHOLD, color="#e74c3c", lw=0.8, ls=":",
               label=f"Blowout threshold ({BLOWOUT_THRESHOLD})")
    ax.axvline(ENTROPY_THRESHOLD, color="#2ecc71", lw=0.8, ls=":",
               label=f"Entropy cut ({ENTROPY_THRESHOLD})")

    ax.set_xlabel("Mean Fold Entropy", fontsize=11)
    ax.set_ylabel("Per-Fold IC", fontsize=11)
    ax.set_title("① Per-Fold IC vs Mean Entropy", fontsize=13, fontweight="bold", pad=10)
    leg = ax.legend(fontsize=8.5, facecolor="#21262d", edgecolor="#30363d",
                    labelcolor="white", loc="upper right")
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # Annotate blowout folds with fold number
    for _, row in blowout.iterrows():
        ax.annotate(
            f"F{int(row['fold'])}",
            xy     = (row["mean_entropy"], row["fold_ic"]),
            xytext = (5, 4), textcoords="offset points",
            fontsize=7.5, color="#e74c3c",
        )


def _plot_suppression(ax, s):
    labels = [
        f"Global IC\n(n={len_str(s['n_low']+s['n_high'])})",
        f"Low Entropy\n(E<{ENTROPY_THRESHOLD}, n={len_str(s['n_low'])}, {s['pct_low']:.0f}%)",
        f"High Entropy\n(E≥{ENTROPY_THRESHOLD}, n={len_str(s['n_high'])}, {s['pct_high']:.0f}%)",
    ]
    values = [s["global_ic"], s["low_ic"], s["high_ic"]]
    colors = [
        "#f39c12" if v < 0 else "#2ecc71" if v > 0 else C_NEUTRAL
        for v in values
    ]

    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3,
                  edgecolor="#30363d", linewidth=0.8)

    for bar, val in zip(bars, values):
        ypos = val + 0.002 if val >= 0 else val - 0.006
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=11, fontweight="bold",
            color="#2ecc71" if val > 0 else "#e74c3c",
        )

    ax.axhline(0, color="#95a5a6", lw=1.0)
    ax.set_ylabel("Information Coefficient (IC)", fontsize=11)
    ax.set_title("② Signal Suppression Proof  —  \"Filter of Gold\"",
                 fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    # zero line label
    ax.text(2.35, 0.001, "IC = 0", color="#95a5a6", fontsize=8.5, va="bottom")


def _plot_regime_bar(ax, regime_map):
    regime_colors = {
        "BULL": C_BULL, "RECOVERY": C_RECOVERY,
        "SIDEWAYS": C_SIDEWAYS, "BEAR": C_BEAR,
    }
    colors = [
        regime_colors.get(r, C_NEUTRAL)
        for r in regime_map["regime"]
    ]
    x = np.arange(len(regime_map))
    bars = ax.bar(x, regime_map["mean_ic"], color=colors,
                  width=0.55, zorder=3, edgecolor="#30363d", linewidth=0.8)

    # Overlay bar: count as subtle annotation
    for i, (bar, row) in enumerate(zip(bars, regime_map.itertuples())):
        yoff = 0.003 if row.mean_ic >= 0 else -0.005
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.mean_ic + yoff,
            f"{row.mean_ic:+.4f}\nn={row.n_signals:,} ({row.pct_total:.0f}%)",
            ha="center",
            va="bottom" if row.mean_ic >= 0 else "top",
            fontsize=9.5, fontweight="bold",
            color="#2ecc71" if row.mean_ic > 0 else "#e74c3c",
        )

    ax.axhline(0, color="#95a5a6", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r}\n(MAE={row.mean_mae:.5f}  |  Avg Ent={row.mean_ent:.3f})"
         for r, row in zip(regime_map["regime"], regime_map.itertuples())],
        fontsize=10,
    )
    ax.set_ylabel("Mean IC", fontsize=11)
    ax.set_title("③ Regime Profitability Map  —  Per-Regime IC Breakdown",
                 fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))


def _plot_heatmap(ax, pivot):
    # Replace NaN with 0 for display; track which cells are empty
    data   = pivot.values.astype(float)
    n_rows, n_cols = data.shape

    import matplotlib.colors as mcolors
    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")   # low MAE = green
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.astype(str), fontsize=10, color="white")
    ax.set_yticks(range(n_rows))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=10, color="white")

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                txt = "N/A"
                color = "#95a5a6"
            else:
                txt = f"{v:.5f}"
                # White text on dark cells, dark on light
                brightness = norm(v)
                color = "white" if brightness > 0.55 else "#0d1117"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.set_label("MAE", color="white", fontsize=10)

    ax.set_xlabel("Volatility Quartile (volatility_7)", fontsize=11)
    ax.set_ylabel("Entropy Band", fontsize=11)
    ax.set_title(
        "④ MAE Heatmap  —  Entropy Band × Volatility Quartile  "
        "(Red = High Error, Green = Low Error)",
        fontsize=13, fontweight="bold", pad=10,
    )


def len_str(n):
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def build_text_report(
    fold_df:     pd.DataFrame,
    r:           float,
    p:           float,
    suppression: dict,
    regime_map:  pd.DataFrame,
    mae_pivot:   pd.DataFrame,
    regime_id_to_label: dict,
) -> str:
    buf = StringIO()

    def pr(s=""): buf.write(s + "\n")

    pr("=" * 72)
    pr("  PROJECT ARI — POST-WFO SIGNAL AUDIT REPORT")
    pr("=" * 72)

    # ── 1. Fold stats ──────────────────────────────────────────────────────
    pr("\n── 1. Per-Fold IC & Entropy Summary ──────────────────────────────")
    pr(f"  {'Fold':>4}  {'n':>5}  {'IC':>8}  {'MAE':>9}  {'Mean Ent':>9}  {'Blowout':>7}")
    pr("  " + "-" * 55)
    for _, row in fold_df.iterrows():
        flag = "⚠ YES" if row["is_blowout"] else "no"
        pr(
            f"  {int(row['fold']):>4}  {int(row['n_rows']):>5}  "
            f"{row['fold_ic']:>+8.4f}  {row['fold_mae']:>9.5f}  "
            f"{row['mean_entropy']:>9.4f}  {flag:>7}"
        )
    pr(f"\n  Pearson r (entropy vs IC) : {r:+.4f}   p-value: {p:.4f}")
    n_blowout = fold_df["is_blowout"].sum()
    pr(f"  Blowout folds (IC<{BLOWOUT_THRESHOLD})  : {n_blowout}  "
       f"({100*n_blowout/len(fold_df):.0f}%)")

    # ── 2. Suppression proof ───────────────────────────────────────────────
    s = suppression
    pr("\n── 2. Signal Suppression Proof  (\"Filter of Gold\") ───────────────")
    pr(f"  Global IC              : {s['global_ic']:+.6f}  (n={s['n_low']+s['n_high']:,})")
    pr(f"  Low Entropy  (E<{ENTROPY_THRESHOLD})   : {s['low_ic']:+.6f}  "
       f"(n={s['n_low']:,}, {s['pct_low']:.1f}%)")
    pr(f"  High Entropy (E≥{ENTROPY_THRESHOLD})   : {s['high_ic']:+.6f}  "
       f"(n={s['n_high']:,}, {s['pct_high']:.1f}%)")
    delta = s["low_ic"] - s["high_ic"]
    pr(f"  IC improvement (low−high): {delta:+.6f}")

    if s["low_ic"] > 0:
        pr(f"\n  ✓ FILTER OF GOLD CONFIRMED: Low-entropy IC is POSITIVE ({s['low_ic']:+.4f})")
        pr(f"    while Global IC is negative ({s['global_ic']:+.4f}).")
        pr(f"    The model is correct when confident — apply Double-Lock filter.")
    else:
        pr(f"\n  ✗ Filter of Gold NOT confirmed: Low-entropy IC ({s['low_ic']:+.4f}) ≤ 0.")
        pr(f"    Entropy alone may not isolate positive-alpha periods.")

    # ── 3. Regime map ──────────────────────────────────────────────────────
    pr("\n── 3. Regime Profitability Map ────────────────────────────────────")
    pr(f"  {'Regime':<10}  {'n':>5}  {'%Total':>7}  {'Mean IC':>9}  "
       f"{'Mean MAE':>9}  {'Avg Ent':>8}  {'Verdict'}")
    pr("  " + "-" * 68)
    for _, row in regime_map.iterrows():
        verdict = (
            "✓ Alpha source"       if row["mean_ic"] > 0.05  else
            "⚠ Weak alpha"         if row["mean_ic"] > 0     else
            "✗ Scale down"         if row["mean_ic"] > -0.05 else
            "✗✗ Suppress / exit"
        )
        pr(
            f"  {row['regime']:<10}  {int(row['n_signals']):>5}  "
            f"{row['pct_total']:>6.1f}%  {row['mean_ic']:>+9.4f}  "
            f"{row['mean_mae']:>9.5f}  {row['mean_ent']:>8.4f}  {verdict}"
        )

    # ── 4. MAE heatmap table ───────────────────────────────────────────────
    pr("\n── 4. MAE Heatmap  (Entropy Band × Volatility Quartile) ──────────")
    pr(mae_pivot.to_string(float_format=lambda x: f"{x:.5f}"))

    # ── Regime ID mapping ──────────────────────────────────────────────────
    pr("\n── Regime ID → Semantic Label Mapping ────────────────────────────")
    for rid, label in sorted(regime_id_to_label.items()):
        pr(f"  Regime {rid}  →  {label}")

    pr("\n" + "=" * 72)
    pr("  END OF AUDIT REPORT")
    pr("=" * 72)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[audit]  Loading predictions...")
    df, regime_id_to_label = load_and_enrich(PREDICTIONS_PATH)
    print(f"[audit]  {len(df):,} OOS rows loaded.  "
          f"Regime mapping: {regime_id_to_label}")

    fold_df, r, p = compute_fold_stats(df)
    suppression   = compute_suppression_proof(df)
    regime_map    = compute_regime_map(df)
    mae_pivot     = compute_mae_heatmap(df)

    # ── Text report ────────────────────────────────────────────────────────
    report = build_text_report(
        fold_df, r, p, suppression, regime_map, mae_pivot, regime_id_to_label,
    )
    print(report)
    TXT_OUT.write_text(report)

    # ── Figure ─────────────────────────────────────────────────────────────
    print("[audit]  Building figure...")
    fig = build_figure(
        fold_df, r, p, suppression, regime_map, mae_pivot, regime_id_to_label,
    )
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"[audit]  Figure saved → {FIG_OUT}")
    print(f"[audit]  Report saved → {TXT_OUT}")


if __name__ == "__main__":
    main()
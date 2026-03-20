import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_competition_plots(equity_path, drawdown_path, output_dir):
    """Generates the three mandatory plots for Techkriti '26 submission."""
    # Load Data
    eq_df = pd.read_csv(equity_path, parse_dates=['date'])
    dd_df = pd.read_csv(drawdown_path, parse_dates=['date'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Style
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # 1. Equity Curve vs Benchmark (Task 4)
    plt.figure()
    # Calculate Buy & Hold for visual comparison
    bnh_equity = (1 + eq_df['close'].pct_change().fillna(0)).cumprod()
    
    plt.plot(eq_df['date'], eq_df['equity'], label='Project ARI (Net)', color='#00ff99', linewidth=2)
    plt.plot(eq_df['date'], bnh_equity, label='BTC Buy & Hold', color='#ff9900', linestyle='--', alpha=0.7)
    
    plt.title("Equity Growth: Project ARI vs. BTC Benchmark", fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value (Normalized to 1.0)")
    plt.legend()
    plt.savefig(output_dir / "equity_curve.png")
    plt.close()

    # 2. Drawdown Plot (Risk Management - Task 5)
    plt.figure()
    plt.fill_between(dd_df['date'], dd_df['drawdown'] * 100, 0, color='#ff4d4d', alpha=0.3)
    plt.plot(dd_df['date'], dd_df['drawdown'] * 100, color='#ff4d4d', linewidth=1)
    
    # Highlight the -30% Penalty Line
    plt.axhline(y=-30, color='white', linestyle=':', label='Penalty Threshold (-30%)')
    
    plt.title("Drawdown Analysis (Risk Control)", fontsize=14, fontweight='bold')
    plt.ylabel("Drawdown %")
    plt.ylim(-40, 5) # Ensure visibility of the penalty zone
    plt.legend()
    plt.savefig(output_dir / "drawdown_curve.png")
    plt.close()

    # 3. Regime-Logic Heatmap (Adaptive Strategy - Task 3)
    plt.figure()
    # Create a simple color map for regimes
    regime_map = {'BEAR': 0, 'SIDEWAYS': 1, 'BULL': 2, 'RECOVERY': 3}
    regime_colors = ['#e74c3c', '#95a5a6', '#2ecc71', '#3498db'] # Red, Gray, Green, Blue
    
    # We plot the close price and shade the background by regime
    plt.plot(eq_df['date'], eq_df['close'], color='white', alpha=0.5, linewidth=1)
    for regime, color in zip(regime_map.keys(), regime_colors):
        mask = eq_df['regime_label'] == regime
        plt.fill_between(eq_df['date'], eq_df['close'].min(), eq_df['close'].max(), 
                         where=mask, color=color, alpha=0.2, label=f"Regime: {regime}")
    
    plt.yscale('log')
    plt.title("Adaptive Logic: Regime-Based Execution Map", fontsize=14, fontweight='bold')
    plt.ylabel("BTC Price (Log Scale)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "regime_plot.png")
    plt.close()
    
    print(f"[Visualizer] All plots saved to {output_dir}")
# Project ARI: Final Backtest Performance Audit
**Date**: 2026-03-20
**Competition**: Techkriti '26 – Beat the Market Challenge

## 1. Core Performance Metrics
| Metric | Strategy (ARI) | Benchmark (BTC B&H) |
| :--- | :--- | :--- |
| **Cumulative Return** | **+49.51%** | +22.35% |
| **Alpha vs Benchmark** | **+27.16%** | - |
| **Net Profit** | **+48.17%** | - |
| **Max Drawdown** | **-24.26%** | -60.0% (est) |
| **Sharpe Ratio** | **0.5191** | - |

## 2. Risk & Execution Statistics
* **Total Trading Days**: 1,050
* **Win Rate**: 38.73%
* **Profit Factor**: 1.2002
* **Transaction Fee Drag**: 1.49% (at 0.15% per trade)
* **Stop-Loss Triggers**: 19 events

## 3. Alpha Attribution (Defensive Engineering)
Project ARI achieved outperformance primarily through capital preservation during the 2022 market downturn:
* **Hard Cash Lock Duration**: 531 Days (50.5% of the total period).
* **Equity Saved by Entropy**: **+23.77%** of starting capital was preserved by the Shannon Entropy Kill-Switch.
* **Regime Routing Efficiency**: Suppressed 273 low-conviction signals via the Regime Gate.

## 4. Benchmark Comparison Analysis
While a naive LightGBM model without ARI logic would have resulted in a **-70.04% loss**, the Adaptive Regime Intelligence engine successfully pivoted logic to maintain a positive equity curve. The strategy remained **5.74% above the disqualification threshold** (-30% MDD) at all times.

## 5. Conclusion
Project ARI proves that unsupervised regime detection combined with information-theoretic risk gating can generate significant Alpha in non-stationary markets while adhering to strict institutional risk constraints.
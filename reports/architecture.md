# Project ARI: System Architecture & Technical Blueprint

## 1. Executive Summary
Project ARI (Adaptive Regime Intelligence) is a modular quantitative trading framework designed for the BTC/USDT pair. The system addresses market non-stationarity by employing unsupervised regime discovery to route data between specialized execution engines.

## 2. The Processing Pipeline
The system operates in four distinct stages to ensure data integrity and algorithmic adaptability:

### Stage 1: Data Preprocessing & Winsorization
* **Outlier Management**: To handle high-volatility "fat tails" in crypto, features are Winsorized at ±3.0σ.
* **Feature Engineering**: 13 primary signals are generated, including VWAP Deviation, RSI-14, and Volume Z-scores.

### Stage 2: HMM Regime Detection
* **Model**: A Gaussian Hidden Markov Model (HMM) identifies 4 latent market states:
    * **SIDEWAYS**: Optimized for Mean-Reversion.
    * **RECOVERY**: Transition state for Trend-Following.
    * **BULL**: High-conviction Trend-Following.
    * **BEAR**: Defensive Mean-Reversion.

### Stage 3: Information-Theoretic Risk Gating
* **Shannon Entropy Kill-Switch**: The system calculates the entropy ($H$) of state probabilities.
* **Hard Cash Lock**: If $H \ge 0.50$, the market is deemed "too chaotic" for reliable prediction, and the system forces a 100% Cash position.

### Stage 4: Dual-Logic Execution Router
* **LGB Engine**: Utilizes LightGBM for gradient-boosted return prediction during Sideways/Bear regimes.
* **Trend Engine**: Utilizes a Moving Average Convergence logic for Bull/Recovery regimes.

## 3. Risk Management & Constraints
* **Stop-Loss**: Fixed -5.0% circuit breaker per trade.
* **Transaction Fees**: Realistic 0.15% fee modeling applied to all trades.
* **Execution**: Only enters Long positions or holds Cash; no Shorting enabled to reduce liquidation risk.
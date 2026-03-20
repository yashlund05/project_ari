# Project ARI: Adaptive Regime Intelligence (Techkriti '26)

## **Executive Summary**
Project ARI is an institutional-grade algorithmic trading framework for BTC/USDT. Unlike static models, ARI uses an **HMM-based Regime Router** to adapt its strategy in real-time.

**Key Performance (2021–2023):**
* **Strategy Net Return**: +49.51% (vs. BTC +22.35%)
* **Alpha Generation**: +27.16%
* **Max Drawdown**: -24.26% (Safely above the -30% penalty threshold)
* **Equity Saved**: +23.77% via Shannon Entropy Kill-Switch

---

## **Technical Architecture**
The system follows a modular **Object-Oriented (OOP)** pipeline:

1.  **Preprocessing & Winsorization**: Outlier suppression at ±3.0σ.
2.  **HMM Regime Detection**: Classifies 4 market states (BULL, BEAR, SIDEWAYS, RECOVERY).
3.  **Shannon Entropy Kill-Switch**: Measures market "chaos." If Entropy ≥ 0.5, the system triggers a **Hard Cash Lock**.
4.  **Dual-Logic Routing**:
    * **LGB Engine**: Mean-reversion for Sideways/Bear markets.
    * **Trend Engine**: Momentum-following for Bull/Recovery markets.

---

## **Risk Management**
* **Stop-Loss**: Automated -5.0% circuit breaker per trade.
* **Fee Modeling**: All returns account for a realistic **0.15% transaction fee**.
* **Survival**: Spent 531 days in cash to avoid the 2022 market crash.

---

## **Installation & Reproduction**
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Strategy**: Execute all cells in `master.ipynb`.
3. **View Results**: Performance metrics and plots are output to `results/plots/`.

## **Quick Start: Reproducing Results**
To reproduce the +27.16% Alpha scorecard and equity plots:
1. Ensure `data/raw/btc_dataset.csv` and `data/processed/predictions.csv` are present.
2. Run all cells in `master.ipynb`.
3. The system will automatically execute the Preprocess -> Detect -> Generate -> Backtest pipeline.
4. Final metrics will print as a 'Competition Scorecard' and plots will save to `results/plots/`.

---
**Developer:** Yash | 2nd Year AIML Engineering (Pune)
**Competition:** Techkriti '26 – IIT Kanpur (Beat the Market Challenge)
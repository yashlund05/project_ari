# Adaptive BTC Trading Strategy – Beat the Market (Techkriti '26)

## Overview

This project presents an adaptive algorithmic trading framework designed to trade Bitcoin (BTC/USDT) using historical market data.

The system integrates:

- Machine learning based return prediction
- Market regime detection
- Rule-based strategy engine
- Volatility-aware position sizing
- Custom backtesting system

The goal is to outperform the traditional **Buy-and-Hold strategy** while maintaining controlled risk exposure.

---

## Problem Statement

This project is developed as part of the **Techkriti '26 – Beat the Market FinTech Challenge**.

Participants must design an algorithmic trading strategy that:

- Uses historical BTC market data (2020–2023)
- Implements custom trading logic
- Includes risk management mechanisms
- Performs better than the Buy-and-Hold benchmark
- Uses a **custom backtesting engine** (no external backtesting libraries)

---

## System Architecture

The trading framework consists of the following pipeline:

```
Market Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Return Prediction Model
↓
Market Regime Detection
↓
Strategy Engine
↓
Risk Management
↓
Custom Backtesting
```

---

## Feature Engineering

The system extracts multiple quantitative indicators from the raw dataset:

- Log Returns
- Volatility (7-day and 30-day)
- Momentum indicators
- Relative Strength Index (RSI)
- VWAP deviation
- Volume anomaly scores

These features help capture both **price dynamics and market sentiment**.

---

## Strategy Logic

The trading system generates three types of signals:

- **LONG** – Enter a long position
- **SHORT** – Enter a short position
- **HOLD** – Stay out of the market

Signals are generated using:

1. Machine learning return prediction
2. Market regime classification
3. Volatility conditions

---

## Risk Management

Risk management is implemented through:

- Volatility-based position sizing
- Regime-aware exposure adjustment
- Stop-loss and loss-limiting mechanisms
- Transaction cost modeling (0.15%)

---

## Backtesting Engine

A **custom backtesting engine** simulates trading over historical BTC data.

The engine evaluates performance using:

- Net Profit
- Gross Profit / Gross Loss
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Win Rate
- Total Closed Trades

---

## Current Development Status

The current repository includes the core architecture of the trading framework.

Work in progress:

- model optimization
- extended backtesting experiments
- parameter tuning
- strategy performance evaluation

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Future Improvements

Planned extensions include:

- reinforcement learning trading agents
- ensemble prediction models
- real-time trading simulation
- improved regime detection models

---

## Disclaimer

This project is intended for **research and educational purposes only** and does not constitute financial advice.
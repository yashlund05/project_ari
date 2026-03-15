# System Architecture – Adaptive BTC Trading Strategy

## Overview

This project implements an adaptive algorithmic trading system designed for the BTC/USDT market.  
The system combines machine learning prediction, market regime detection, and rule-based risk management to generate trading decisions.

The architecture is designed to be modular so that individual components can be improved or replaced independently.

---

## System Pipeline

The complete trading pipeline consists of the following stages:

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
Custom Backtesting Engine  

---

## 1. Data Preprocessing

The raw BTC/USDT dataset contains historical OHLCV data.

Steps performed:

- timestamp validation
- missing data handling
- correction of invalid price values
- normalization of volume

This step ensures the dataset is suitable for quantitative analysis.

---

## 2. Feature Engineering

Multiple technical indicators are extracted from the price and volume data to capture different market characteristics.

Key features include:

- Log returns
- Volatility indicators (7-day and 30-day)
- Momentum indicators
- Relative Strength Index (RSI)
- VWAP deviation
- Volume anomaly detection

These features serve as inputs for the prediction model and regime detection logic.

---

## 3. Return Prediction Model

A machine learning model predicts short-term future returns based on engineered features.

The prediction output is used to estimate potential market direction and assist the trading strategy in generating signals.

---

## 4. Market Regime Detection

Financial markets behave differently during different conditions.

The system identifies three market regimes:

- Bull Market
- Bear Market
- Sideways Market

Regime classification helps the strategy adapt its behavior depending on market volatility and trend strength.

---

## 5. Strategy Engine

The strategy engine combines:

- predicted returns
- regime classification
- market indicators

to generate trading signals:

LONG – buy Bitcoin  
SHORT – sell or short position  
HOLD – remain out of the market

The objective is to capture profitable trends while avoiding uncertain market conditions.

---

## 6. Risk Management

Risk management is implemented through several mechanisms:

- volatility-based position sizing
- regime-based exposure control
- capital allocation limits
- transaction cost modelling

These mechanisms help protect capital during adverse market movements.

---

## 7. Custom Backtesting Engine

A custom backtesting engine evaluates strategy performance on historical data.

This engine simulates trade execution and incorporates transaction costs to ensure realistic results.

Performance metrics include:

- Net Profit
- Gross Profit / Gross Loss
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Win Rate
- Total Closed Trades

---

## Design Goals

The system is designed to achieve the following:

- adaptability to changing market conditions
- strong risk management
- transparent trading logic
- reproducible research results
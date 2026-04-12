# FPL-LSTM-Prediction-EE4146.
EE4146 Machine Learning Project: Predicting Fantasy Premier League (FPL) points using position-specific LSTM neural networks to determine the optimal historical lookback window per player position.
-------------------------------

# FPL-LSTM-Prediction-EE4146
**City University of Hong Kong - EE4146 Final Project**

## Overview
This repository contains a full machine learning pipeline designed to predict Fantasy Premier League (FPL) points. We investigated whether the optimal historical memory depth (lookback window) depends on player position using Long Short-Term Memory (LSTM) neural networks.

## Repository Structure
* `/data/`: Raw FPL datasets (2023-2026).
* `/final_data/`: Enriched datasets featuring 16 engineered columns (opponent strength, team momentum).
* `/notebooks/`: The modularized code pipeline (Data Prep -> Baseline LSTM -> Enhanced LSTM -> Ablation Study).
* `/results/`: Evaluation metrics, CSV exports, and final comparative visualizations.

## Key Findings
* Midfielders require a 10-gameweek lookback window, while Defenders only require a 4-week window.
* Complex context features improved training metrics but caused overfitting in live testing.
* Goalkeepers rely heavily on personal form, whereas Forwards depend critically on match context.

# FPL LSTM Prediction - Position-Specific Lookback Windows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)

This repository contains the complete codebase and experimental framework for the research project:
**"Predicting Fantasy Premier League Performance using LSTM: An Analysis of Position-Specific Lookback Windows."**

---

## 📌 Project Overview

Fantasy Premier League (FPL) performance prediction is challenging due to player variance, fixture difficulty, team form, and role-specific duties. This project systematically evaluates whether the optimal historical lookback window length ($w \in [1, 15]$ gameweeks) for predicting next-gameweek FPL points differs across player positions (**Goalkeepers [GK]**, **Defenders [DEF]**, **Midfielders [MID]**, and **Forwards [FWD]**).

Key components of the framework include:
- **Baseline Data Processing**: Preprocessing raw gameweek data and constructing a standard 9-feature baseline set.
- **Enhanced Feature Engineering**: Engineering team momentum, real-time league standings, opponent strength metrics, and defensive/attacking context.
- **Position-Specific LSTM Grid Search**: Training recurrent neural network models over lookback sequence windows from 1 to 15 gameweeks for each position.
- **Ablation Study**: Quantifying feature group contributions (Personal Form, Expected Stats, Match Context) to isolate key predictive drivers by position.

---

## 📁 Detailed Repository Structure

```text
.
├── raw_data/                       # Raw season CSV datasets
│   ├── raw_23_24.csv               # Gameweek data for 2023-24 season
│   ├── raw_24_25.csv               # Gameweek data for 2024-25 season
│   └── raw_25_26.csv               # Gameweek data for 2025-26 season
├── data/                           # Generated baseline CSVs split by position
│   ├── GK_all_seasons.csv
│   ├── DEF_all_seasons.csv
│   ├── MID_all_seasons.csv
│   └── FWD_all_seasons.csv
├── final_data/                     # Generated enhanced feature CSVs split by position
│   ├── GK_all_seasons_final.csv
│   ├── DEF_all_seasons_final.csv
│   ├── MID_all_seasons_final.csv
│   └── FWD_all_seasons_final.csv
├── notebooks/                      # Interactive Jupyter Notebooks
│   ├── 01_data_preparation.ipynb   # Step-by-step cleaning & feature engineering
│   └── 02_modeling_ablation.ipynb  # Grid search, ablation, predictions, and plots
├── results/                        # Output metrics and visualizations
│   ├── baseline_mae_results.csv    # Grid search MAE results for baseline model
│   ├── enhanced_mae_results.csv    # Grid search MAE results for enhanced model
│   ├── ablation_results.csv        # Ablation study MAE degradation scores
│   ├── figures/                    # Saved plot visualizations
│   └── README.md                   # Results documentation
├── main.py                         # End-to-end execution pipeline script
├── requirements.txt                # Python environment dependencies
└── README.md                       # Main project documentation
```

---

## 📊 Feature Definitions

### 1. Baseline Features (9)
- `total_points`: Points scored in the current gameweek.
- `goals_scored`: Number of goals scored.
- `assists`: Number of assists provided.
- `expected_goals` ($xG$): Expected goals metric.
- `expected_assists` ($xA$): Expected assists metric.
- `minutes`: Minutes played in the gameweek.
- `engineered_form`: 4-gameweek rolling average of total points.
- `was_home`: Binary indicator (1 if home match, 0 if away).
- `clean_sheets`: Clean sheet achieved (1 or 0).

### 2. Enhanced Features (16 Total)
Includes all 9 baseline features plus 7 context-aware features:
- `league_position`: Current league standing rank of the player's team.
- `opponent_league_position`: Current league standing rank of the opponent team.
- `team_total_points`: Cumulative FPL points scored by player's team in the gameweek.
- `opp_att_points`: Attacking points scored by opponent's forwards/midfielders.
- `opp_def_points`: Defensive points conceded/scored by opponent's defenders/goalkeeper.
- `league_pos_trend`: 3-gameweek change in league position for player's team.
- `opponent_position_trend`: 3-gameweek change in league position for opponent team.

---

## ⚙️ Model Architecture & Methodology

- **Model Type**: Stacked Long Short-Term Memory (LSTM) network.
- **Layers**:
  - `LSTM(64)` with `tanh` activation & sequence output.
  - `LSTM(32)` with `tanh` activation.
  - `Dense(16)` with `relu` activation.
  - `Dropout(0.2)` regularization.
  - `Dense(1)` linear output predicting `Next_GW_Points`.
- **Training Setup**:
  - Optimizer: `Adam`
  - Loss Function: Mean Squared Error (`mse`)
  - Evaluation Metric: Mean Absolute Error (`mae`)
  - Callbacks: Early Stopping on validation loss with patience of 5 epochs.
  - Data Split: 2023–24 (Train), 2024–25 (Validation), 2025–26 (Test).

---

## 🚀 Quick Start & Usage

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/abmhk10/FPL-LSTM-Prediction-EE4146..git
cd FPL-LSTM-Prediction-EE4146.

python3 -m venv fpl_env
source fpl_env/bin/activate
pip install -r requirements.txt
```

### 2. Running the Full Pipeline

Run the end-to-end pipeline using `main.py`:

```bash
python3 main.py
```

`main.py` performs all pipeline stages automatically:
1. Preprocesses raw season files in `raw_data/` and saves baseline position datasets to `data/`.
2. Computes contextual league features and saves enhanced position datasets to `final_data/`.
3. Performs LSTM lookback grid search ($w=1..15$) for baseline and enhanced datasets.
4. Conducts the ablation study across feature groups.
5. Exports CSV metrics and comparison plot figures to `results/`.

---

## 📈 Results Summary

| Position | Baseline Optimal Window ($w$) | Baseline MAE | Enhanced Optimal Window ($w$) | Enhanced MAE | Relative Improvement |
|---|---:|---:|---:|---:|---:|
| **GK**  | 7  | 2.078 | 9  | 2.034 | **+2.1%** |
| **DEF** | 4  | 2.296 | 5  | 2.296 | **0.0%**  |
| **MID** | 10 | 1.972 | 5  | 1.923 | **+2.5%** |
| **FWD** | 9  | 2.410 | 11 | 2.365 | **+1.9%** |

### Ablation Findings
- **Goalkeepers (GK)** show highest sensitivity to **Personal Form** (+5.0% error increase when removed).
- **Forwards (FWD)** rely heavily on **Match Context** (+6.3% error increase when removed).

---

## 👥 Authors

- **Aadib Bukhari Mafaz** (57975895)
- **KAPYA Zachariah Muya** (58494409)

*Department of Electrical Engineering, City University of Hong Kong*

# FPL LSTM Prediction - Position-Specific Lookback Windows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)

This repository contains the complete codebase for the research project **"Predicting Fantasy Premier League Performance using LSTM: An Analysis of Position-Specific Lookback Windows."**

We systematically evaluate whether the optimal historical lookback window for predicting next-gameweek FPL points differs by player position (GK, DEF, MID, FWD). The project includes:

- Full data preprocessing (baseline 9-feature dataset)
- Advanced feature engineering (opponent strength, real-time league positions, team momentum -> 16 features)
- LSTM grid search over 1-15 gameweek windows
- Ablation study to quantify feature importance
- Live prediction tests on the unseen 2025-26 season

---

## Code Availability

The complete implementation, including data preprocessing scripts, baseline and enhanced LSTM models, grid search, ablation study, and final prediction code, is available in this project repository:

- [https://github.com/abmhk10/FPL-LSTM-Prediction-EE4146.](https://github.com/abmhk10/FPL-LSTM-Prediction-EE4146.)

> **Important:** The trailing `.` in the repository name is intentional. Please keep it when cloning (the clone URL ends with `..git` because the repo name already ends with a dot).

If you prefer to run the code locally, follow the setup instructions below. If you do not have a local GPU or prefer a hosted environment, use the notebook workflow in Google Colab.

---

## Repository Structure

```text
.
├── raw_data/                      # Raw Vaastav CSV files
├── notebooks/
│   ├── 01_data_preparation.ipynb  # Initial prep + opponent feature engineering
│   └── 02_modeling_ablation.ipynb # Grid search, ablation, predictions, plots
├── main.py                        # End-to-end pipeline (run everything at once)
├── data/                          # Baseline 9-feature CSVs (generated)
├── final_data/                    # Enhanced 16-feature CSVs (generated)
├── results/                       # All outputs (CSVs, PNGs)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Quick Start

### Option 1: Run Everything with a Single Command (main.py)

1. **Clone the repository:**

	```bash
	git clone https://github.com/abmhk10/FPL-LSTM-Prediction-EE4146..git
	cd FPL-LSTM-Prediction-EE4146.
	```

2. **Raw data is already included** in `raw_data/` after cloning.

	Included files:
	- `raw_23_24.csv`
	- `raw_24_25.csv`
	- `raw_25_26.csv`

	(Optional) Replace these files with your own Vaastav exports if needed, keeping the same filenames.

3. **Create and activate a virtual environment (recommended):**

	```bash
	python3 -m venv fpl_env
	source fpl_env/bin/activate      # On Windows: fpl_env\Scripts\activate
	```

4. **Install dependencies:**

	```bash
	pip install -r requirements.txt
	```

5. **Run the full pipeline:**

	```bash
	python3 main.py
	```

The script automatically creates `data/`, `final_data/`, and `results/` folders and writes all intermediate/final outputs.

This pipeline was developed and tested on **Ubuntu 22.04 (Jammy)** and should run without modification on recent Ubuntu distributions.

### Windows Setup

```powershell
python -m venv fpl_env
fpl_env\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Mac Setup

```bash
python3 -m venv fpl_env
source fpl_env/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Option 2: Run the Notebooks in Google Colab

If you prefer an interactive environment or do not have a local GPU, use the provided notebooks.

- Upload `01_data_preparation.ipynb` and `02_modeling_ablation.ipynb` to Colab.
- Mount Google Drive when prompted.
- Place the raw CSV files in the expected Drive folder (or update notebook paths).
- Run all cells in order.

---

## Results Summary

| Position | Baseline Optimal Window | Baseline MAE | Enhanced Optimal Window | Enhanced MAE | Improvement |
|---|---:|---:|---:|---:|---:|
| GK  | 7  | 2.078 | 9  | 2.034 | +2.1% |
| DEF | 4  | 2.296 | 5  | 2.296 | 0.0%  |
| MID | 10 | 1.972 | 5  | 1.923 | +2.5% |
| FWD | 9  | 2.410 | 11 | 2.365 | +1.9% |

### Ablation Highlights

- Goalkeepers rely heavily on personal form (+5.0% MAE degradation when removed).
- Forwards depend critically on match context (+6.3% MAE degradation).

For full discussion and methodology details, refer to the accompanying research report.

---

## Libraries Used

- TensorFlow / Keras (v2.19.0) - LSTM networks
- Pandas - data manipulation
- Scikit-Learn - Min-Max scaling
- Matplotlib / Seaborn - visualizations

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Authors

- Aadib Bukhari Mafaz (57975895)
- KAPYA Zachariah Muya (58494409)

Department of Electrical Engineering, City University of Hong Kong

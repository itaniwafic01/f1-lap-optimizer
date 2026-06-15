# F1 Qualifying Lap Time Optimizer — Red Bull Ring

A machine learning project to predict and optimize Max Verstappen's qualifying lap time sector-by-sector at the Red Bull Ring (Austrian GP).

## Goal
Predict S1, S2, and S3 sector times using post-2022 FastF1 data, then sum them for a full lap prediction. Validated blind against the 2025 Austrian GP qualifying session.

## Project Structure
```
f1-lap-optimizer/
├── data/
│   ├── raw/          # FastF1 cache (git-ignored)
│   └── processed/    # cleaned, feature-engineered CSVs
├── notebooks/        # EDA and model experiments
├── src/              # reusable modules (data loading, features, model)
├── models/           # saved model files
└── requirements.txt
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Source
[FastF1](https://docs.fastf1.dev/) — official F1 timing and telemetry data, 2022–2025.

## Stack
- **Data**: FastF1, pandas, numpy
- **Model**: XGBoost / LightGBM
- **Viz**: matplotlib, seaborn

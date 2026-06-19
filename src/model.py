import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import pickle

INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "2026_features.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# features the model uses — no raw lap/sector times (would be leakage)
BASE_FEATURES = [
    "LapLengthKm", "NumCorners", "NumDRS", "AvgSpeedKmh", "DownforceLevel",
    "AirTemp", "TrackTemp", "Humidity", "WindSpeed",
    "SessionTypeCode", "CircuitTypeCode",
    "DriverFormDelta",
]

SECTOR_TARGETS = {
    "S1": "S1Seconds_pct",
    "S2": "S2Seconds_pct",
    "S3": "S3Seconds_pct",
}


def get_driver_team_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("Driver_") or c.startswith("Team_")]


def train_sector_model(df: pd.DataFrame, sector: str, target_col: str):
    driver_team_cols = get_driver_team_cols(df)
    feature_cols = BASE_FEATURES + driver_team_cols

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    model.fit(X, y)

    train_pred = model.predict(X)
    train_mae = mean_absolute_error(y, train_pred)

    print(f"  {sector}: CV MAE = {cv_mae:.4f}%  |  Train MAE = {train_mae:.4f}%")
    return model, feature_cols


def save_model(model, feature_cols: list, sector: str):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"model_{sector.lower()}.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)


def load_model(sector: str):
    with open(MODEL_DIR / f"model_{sector.lower()}.pkl", "rb") as f:
        return pickle.load(f)


def get_feature_importance(model, feature_cols: list, top_n: int = 10) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return importance


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Training on {len(df)} laps, {len(BASE_FEATURES)} base features + driver/team dummies\n")

    models = {}
    for sector, target in SECTOR_TARGETS.items():
        print(f"Training {sector} model...")
        model, feature_cols = train_sector_model(df, sector, target)
        save_model(model, feature_cols, sector)
        models[sector] = (model, feature_cols)

    print("\nTop features per sector:")
    for sector, (model, feature_cols) in models.items():
        print(f"\n  {sector}:")
        imp = get_feature_importance(model, feature_cols)
        for _, row in imp.iterrows():
            print(f"    {row['feature']:<30} {row['importance']:.4f}")

    print("\nModels saved to models/")

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "2026_features.csv"

# Austria circuit characteristics (Round 8)
AUSTRIA = {
    "LapLengthKm": 4.318,
    "NumCorners": 10,
    "NumDRS": 3,
    "AvgSpeedKmh": 235,
    "DownforceLevel": 1,
    "CircuitTypeCode": 0,   # permanent
    "SessionTypeCode": 0,   # Q
}

# estimated Austria qualifying weather (late June, Red Bull Ring)
AUSTRIA_WEATHER = {
    "AirTemp": 24.0,
    "TrackTemp": 45.0,
    "Humidity": 40.0,
    "WindSpeed": 2.5,
}

# sector medians based on 2025 Austrian GP qualifying baseline
AUSTRIA_MEDIANS = {
    "S1Seconds": 16.45,
    "S2Seconds": 29.10,
    "S3Seconds": 19.55,
}


def load_model(sector: str):
    with open(MODEL_DIR / f"model_{sector.lower()}.pkl", "rb") as f:
        return pickle.load(f)


def get_driver_form(df: pd.DataFrame, driver: str) -> float:
    driver_rows = df[df["Driver"] == driver]
    if driver_rows.empty:
        return 0.0
    return driver_rows["DriverFormDelta"].iloc[-1]


def build_input_row(df: pd.DataFrame, driver: str, team: str) -> pd.DataFrame:
    driver_form = get_driver_form(df, driver)

    row = {**AUSTRIA, **AUSTRIA_WEATHER, "DriverFormDelta": driver_form}

    # driver dummies
    driver_cols = [c for c in df.columns if c.startswith("Driver_")]
    for col in driver_cols:
        row[col] = 1 if col == f"Driver_{driver}" else 0

    # team dummies
    team_cols = [c for c in df.columns if c.startswith("Team_")]
    for col in team_cols:
        row[col] = 1 if col == f"Team_{team}" else 0

    return pd.DataFrame([row])


def predict_lap(driver: str, team: str, df: pd.DataFrame) -> dict:
    input_row = build_input_row(df, driver, team)

    results = {}
    for sector in ["S1", "S2", "S3"]:
        payload = load_model(sector)
        model = payload["model"]
        feature_cols = payload["feature_cols"]

        input_aligned = input_row.reindex(columns=feature_cols, fill_value=0)
        pct_delta = model.predict(input_aligned)[0]

        median = AUSTRIA_MEDIANS[f"{sector}Seconds"]
        predicted_seconds = median * (1 + pct_delta / 100)
        results[sector] = predicted_seconds

    results["LapTime"] = results["S1"] + results["S2"] + results["S3"]
    return results


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    # predict for all drivers in our dataset
    drivers = {
        "VER": "Red Bull Racing",
        "ANT": "Mercedes",
        "RUS": "Mercedes",
        "LEC": "Ferrari",
        "HAM": "Ferrari",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "HAD": "Red Bull Racing",
        "GAS": "Alpine",
        "LAW": "Racing Bulls",
    }

    print("=" * 55)
    print(f"{'2026 Austrian GP — Predicted Q3 Lap Times':^55}")
    print(f"{'Red Bull Ring':^55}")
    print("=" * 55)
    print(f"{'Pos':<5}{'Driver':<8}{'Team':<22}{'S1':>7}{'S2':>7}{'S3':>7}{'Lap':>10}")
    print("-" * 55)

    predictions = []
    for driver, team in drivers.items():
        result = predict_lap(driver, team, df)
        predictions.append({"Driver": driver, "Team": team, **result})

    predictions.sort(key=lambda x: x["LapTime"])

    for pos, pred in enumerate(predictions, 1):
        lap_str = format_time(pred["LapTime"])
        print(f"  {pos:<4}{pred['Driver']:<8}{pred['Team']:<22}"
              f"{pred['S1']:>7.3f}{pred['S2']:>7.3f}{pred['S3']:>7.3f}{lap_str:>10}")

    print("=" * 55)
    print(f"\nEstimated weather: {AUSTRIA_WEATHER['AirTemp']}°C air | "
          f"{AUSTRIA_WEATHER['TrackTemp']}°C track | "
          f"{AUSTRIA_WEATHER['WindSpeed']} m/s wind")
    print("Sector medians based on 2025 Austrian GP qualifying baseline.")
    print("Update AUSTRIA_MEDIANS and AUSTRIA_WEATHER before race weekend.")

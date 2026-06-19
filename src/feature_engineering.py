import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "2026_qualifying_training.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "2026_features.csv"

# physical characteristics for each circuit in our dataset + Austria (target)
CIRCUIT_FEATURES = {
    1: {"CircuitName": "Australia",  "LapLengthKm": 5.278, "NumCorners": 16, "NumDRS": 4, "AvgSpeedKmh": 218, "MaxSpeedKmh": 320, "DownforceLevel": 3},
    2: {"CircuitName": "China",      "LapLengthKm": 5.451, "NumCorners": 16, "NumDRS": 2, "AvgSpeedKmh": 210, "MaxSpeedKmh": 327, "DownforceLevel": 3},
    3: {"CircuitName": "Japan",      "LapLengthKm": 5.807, "NumCorners": 18, "NumDRS": 2, "AvgSpeedKmh": 228, "MaxSpeedKmh": 330, "DownforceLevel": 3},
    4: {"CircuitName": "Miami",      "LapLengthKm": 5.412, "NumCorners": 19, "NumDRS": 3, "AvgSpeedKmh": 215, "MaxSpeedKmh": 320, "DownforceLevel": 3},
    5: {"CircuitName": "Canada",     "LapLengthKm": 4.361, "NumCorners": 14, "NumDRS": 3, "AvgSpeedKmh": 210, "MaxSpeedKmh": 328, "DownforceLevel": 2},
    6: {"CircuitName": "Monaco",     "LapLengthKm": 3.337, "NumCorners": 19, "NumDRS": 1, "AvgSpeedKmh": 163, "MaxSpeedKmh": 298, "DownforceLevel": 5},
    7: {"CircuitName": "Barcelona",  "LapLengthKm": 4.657, "NumCorners": 14, "NumDRS": 2, "AvgSpeedKmh": 213, "MaxSpeedKmh": 322, "DownforceLevel": 3},
    8: {"CircuitName": "Austria",    "LapLengthKm": 4.318, "NumCorners": 10, "NumDRS": 3, "AvgSpeedKmh": 235, "MaxSpeedKmh": 330, "DownforceLevel": 1},
}

CIRCUIT_DF = pd.DataFrame.from_dict(CIRCUIT_FEATURES, orient="index").rename_axis("Round").reset_index()


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(CIRCUIT_DF, on="Round", how="left")
    return df


def add_relative_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Express each lap time as delta from the session median — removes circuit scale effect."""
    for col in ["LapTimeSeconds", "S1Seconds", "S2Seconds", "S3Seconds"]:
        session_median = df.groupby(["Round", "SessionType"])[col].transform("median")
        df[f"{col}_delta"] = df[col] - session_median
        df[f"{col}_pct"] = (df[col] - session_median) / session_median * 100
    return df


def add_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling average of each driver's relative lap time delta — captures current form."""
    df = df.sort_values(["Driver", "Round"])
    df["DriverFormDelta"] = (
        df.groupby("Driver")["LapTimeSeconds_delta"]
        .transform(lambda x: x.expanding().mean())
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["CircuitTypeCode"] = df["CircuitType"].map({"permanent": 0, "street": 1})
    df["SessionTypeCode"] = df["SessionType"].map({"Q": 0, "SQ": 1})
    driver_dummies = pd.get_dummies(df["Driver"], prefix="Driver")
    team_dummies = pd.get_dummies(df["Team"], prefix="Team")
    df = pd.concat([df, driver_dummies, team_dummies], axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_circuit_features(df)
    df = add_relative_performance(df)
    df = add_driver_form(df)
    df = encode_categoricals(df)
    df = df.dropna(subset=["S1Seconds", "S2Seconds", "S3Seconds",
                            "LapTimeSeconds_delta", "DriverFormDelta"])
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Input: {len(df)} laps")

    df = build_features(df)
    print(f"Output: {len(df)} laps, {len(df.columns)} features")

    print("\nCircuit features added:")
    print(df[["Round", "CircuitName", "LapLengthKm", "NumCorners", "NumDRS",
              "AvgSpeedKmh", "DownforceLevel"]].drop_duplicates("Round").to_string(index=False))

    print("\nSample relative performance (LapTimeSeconds_delta):")
    print(df.groupby("Driver")["LapTimeSeconds_delta"].mean().sort_values().to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

fastf1.Cache.enable_cache(str(CACHE_DIR))

YEAR = 2026
TARGET_ROUND = 8  # Austria — blind prediction

# rounds and which sessions to pull (sprint weekends get both Q and SQ)
ROUND_SESSIONS = {
    1: ["Q"],         # Australia
    2: ["Q", "SQ"],   # China (sprint)
    3: ["Q"],         # Japan
    4: ["Q", "SQ"],   # Miami (sprint)
    5: ["Q", "SQ"],   # Canada (sprint)
    6: ["Q"],         # Monaco
    7: ["Q"],         # Barcelona
}

TEAM_TIER = {
    "Red Bull Racing": 1,
    "Ferrari": 2,
    "Mercedes": 3,
    "McLaren": 4,
    "Aston Martin": 5,
    "Alpine": 6,
    "Williams": 7,
    "RB": 8,
    "Kick Sauber": 9,
    "Haas F1 Team": 10,
}

CIRCUIT_TYPE = {
    1: "street",      # Australia
    2: "permanent",   # China
    3: "permanent",   # Japan
    4: "permanent",   # Miami
    5: "permanent",   # Canada
    6: "street",      # Monaco
    7: "permanent",   # Barcelona
}


def load_session_laps(session, round_num, session_type) -> pd.DataFrame:
    fastest = (
        session.laps.pick_quicklaps()
        .groupby("Driver")["LapTime"]
        .min()
        .sort_values()
    )
    # SQ has all 20 drivers — take top 10 to match Q3 scope
    top_drivers = list(fastest.head(10).index)
    print(f"    Top drivers: {top_drivers}")

    laps = session.laps.pick_drivers(top_drivers).pick_quicklaps()
    laps = laps[["Driver", "Team", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                  "Compound", "TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                  "IsPersonalBest", "LapNumber"]].copy()

    weather = session.weather_data
    if not weather.empty:
        avg_weather = weather[["AirTemp", "TrackTemp", "Humidity", "WindSpeed"]].mean()
        for col, val in avg_weather.items():
            laps[col] = val

    laps["Round"] = round_num
    laps["SessionType"] = session_type
    laps["CircuitType"] = CIRCUIT_TYPE.get(round_num, "permanent")
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["S1Seconds"] = laps["Sector1Time"].dt.total_seconds()
    laps["S2Seconds"] = laps["Sector2Time"].dt.total_seconds()
    laps["S3Seconds"] = laps["Sector3Time"].dt.total_seconds()
    laps["TeamTier"] = laps["Team"].map(TEAM_TIER).fillna(10)

    laps = laps.dropna(subset=["LapTimeSeconds", "S1Seconds", "S2Seconds", "S3Seconds"])

    for col in ["S1Seconds", "S2Seconds", "S3Seconds"]:
        mean, std = laps[col].mean(), laps[col].std()
        laps = laps[abs(laps[col] - mean) <= 3 * std]

    return laps


def load_qualifying_laps(round_sessions=ROUND_SESSIONS) -> pd.DataFrame:
    all_laps = []

    for round_num, sessions in round_sessions.items():
        for session_type in sessions:
            print(f"\nLoading 2026 Round {round_num} — {session_type}...")
            try:
                session = fastf1.get_session(YEAR, round_num, session_type)
                session.load(telemetry=False, weather=True, messages=False)
                laps = load_session_laps(session, round_num, session_type)
                all_laps.append(laps)
                print(f"  -> {len(laps)} clean laps loaded")
            except Exception as e:
                print(f"  -> Failed: {e}")

    df = pd.concat(all_laps, ignore_index=True)
    return df


def save_processed(df: pd.DataFrame, filename="2026_qualifying_training.csv"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} laps to {path}")


if __name__ == "__main__":
    df = load_qualifying_laps()
    save_processed(df)
    print(f"\nTotal laps: {len(df)}")
    print(f"Drivers: {sorted(df['Driver'].unique())}")
    print("\nLaps per round/session:")
    print(df.groupby(["Round", "SessionType"])["Driver"].count().rename("Laps"))

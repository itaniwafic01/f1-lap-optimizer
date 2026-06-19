import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

fastf1.Cache.enable_cache(str(CACHE_DIR))

YEARS = [2022, 2023, 2024, 2025]
CIRCUIT = "Austria"
SESSION_TYPE = "Q"
DRIVER = "VER"


def load_qualifying_laps(years=YEARS) -> pd.DataFrame:
    all_laps = []

    for year in years:
        print(f"\nLoading {year} Austrian GP Qualifying...")
        try:
            session = fastf1.get_session(year, CIRCUIT, SESSION_TYPE)
            session.load(telemetry=False, weather=True, messages=False)

            laps = session.laps.pick_drivers(DRIVER).pick_quicklaps()
            laps = laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                          "Compound", "TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                          "IsPersonalBest", "LapNumber"]].copy()

            weather = session.weather_data
            if not weather.empty:
                avg_weather = weather[["AirTemp", "TrackTemp", "Humidity", "WindSpeed"]].mean()
                for col, val in avg_weather.items():
                    laps[col] = val

            laps["Year"] = year
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
            laps["S1Seconds"] = laps["Sector1Time"].dt.total_seconds()
            laps["S2Seconds"] = laps["Sector2Time"].dt.total_seconds()
            laps["S3Seconds"] = laps["Sector3Time"].dt.total_seconds()

            laps = laps.dropna(subset=["LapTimeSeconds", "S1Seconds", "S2Seconds", "S3Seconds"])
            all_laps.append(laps)
            print(f"  -> {len(laps)} clean laps loaded")

        except Exception as e:
            print(f"  -> Failed for {year}: {e}")

    df = pd.concat(all_laps, ignore_index=True)
    return df


def save_processed(df: pd.DataFrame, filename="ver_austria_qualifying.csv"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} laps to {path}")


if __name__ == "__main__":
    df = load_qualifying_laps()
    save_processed(df)
    print("\nSample:")
    print(df[["Year", "LapTimeSeconds", "S1Seconds", "S2Seconds", "S3Seconds", "Compound", "TyreLife"]].to_string())

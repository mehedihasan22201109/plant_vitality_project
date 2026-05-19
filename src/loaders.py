
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent  # D:/4-1
_RAW = _BASE / "data" / "raw"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names; replace spaces, dots, slashes with underscores; strip whitespace."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"[ ./\\]+", "_", regex=True)
    )
    return df


def _biggest_csv(folder: Path) -> pd.DataFrame | None:
    """Find the largest CSV file in *folder* recursively and return it as a DataFrame, or None."""
    try:
        csvs = list(folder.rglob("*.csv"))
        if not csvs:
            return None
        biggest = max(csvs, key=lambda p: p.stat().st_size)
        return pd.read_csv(biggest)
    except Exception:
        return None


def _year_from(series: pd.Series) -> pd.Series:
    """Parse a series to datetime and extract the year as int."""
    return pd.to_datetime(series, errors="coerce").dt.year


def _safe(df: pd.DataFrame, keep_cols: list[str], dropna_col: str = "Year") -> pd.DataFrame:
    """Keep only columns that exist in df, then drop rows with nulls in dropna_col."""
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    if dropna_col in df.columns:
        df = df.dropna(subset=[dropna_col])
    return df


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def _load_fao_yield() -> pd.DataFrame | None:
    try:
        path = _RAW / "crop_yield_fao" / "yield.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # Filter rows where element contains "yield" — case-insensitive
        df = df[df["element"].str.contains("yield", na=False, case=False)]
        # Filter years 2000-2020
        df = df[(df["year"] >= 2000) & (df["year"] <= 2020)]
        out = pd.DataFrame({
            "Year": df["year"].astype(int),
            "Country": df["area"],
            "Crop_Type": df["item"],
            "yield_hgha": pd.to_numeric(df["value"], errors="coerce"),
        })
        return _safe(out, ["Year", "Country", "Crop_Type", "yield_hgha"])
    except Exception:
        return None


def _load_fao_rain() -> pd.DataFrame | None:
    try:
        path = _RAW / "crop_yield_fao" / "rainfall.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # Rainfall values are stored as strings — coerce to numeric
        df["average_rain_fall_mm_per_year"] = pd.to_numeric(
            df["average_rain_fall_mm_per_year"], errors="coerce"
        )
        out = pd.DataFrame({
            "Year": pd.to_numeric(df["year"], errors="coerce"),
            "Country": df["area"],
            "rainfall_mm": df["average_rain_fall_mm_per_year"],
        })
        out = _safe(out, ["Year", "Country", "rainfall_mm"])
        out = out.groupby(["Year", "Country"], as_index=False)["rainfall_mm"].mean()
        return out
    except Exception:
        return None


def _load_fao_temp() -> pd.DataFrame | None:
    try:
        path = _RAW / "crop_yield_fao" / "temp.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # column is "country" (lowercase) — rename to "Country"
        if "country" in df.columns:
            df = df.rename(columns={"country": "Country"})
        out = pd.DataFrame({
            "Year": pd.to_numeric(df["year"], errors="coerce"),
            "Country": df["Country"],
            "avg_temp": pd.to_numeric(df["avg_temp"], errors="coerce"),
        })
        out = _safe(out, ["Year", "Country", "avg_temp"])
        out = out.groupby(["Year", "Country"], as_index=False)["avg_temp"].mean()
        return out
    except Exception:
        return None


def _load_crop_rec(folder_name: str = "crop_recommendation") -> pd.DataFrame | None:
    try:
        folder = _RAW / folder_name
        # Try a standard filename first, fall back to largest CSV
        standard = folder / "Crop_recommendation.csv"
        if standard.exists():
            df = pd.read_csv(standard)
        else:
            df = _biggest_csv(folder)
            if df is None:
                return None
        df = _norm(df)
        # Rename columns to canonical names
        rename_map = {
            "n": "nitrogen_N",
            "p": "phosphorous_P",
            "k": "potassium_K",
            "temperature": "avg_temp",
            "humidity": "humidity_pct",
            "ph": "soil_pH",
            "rainfall": "rainfall_mm",
            "label": "Crop_Type",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        keep = ["Crop_Type", "nitrogen_N", "phosphorous_P", "potassium_K",
                "avg_temp", "humidity_pct", "soil_pH", "rainfall_mm"]
        return _safe(df, keep, dropna_col="Crop_Type")
    except Exception:
        return None


def _load_bd_agro() -> pd.DataFrame | None:
    try:
        path = _RAW / "bd_agroclimatic" / "Bangladesh Agroclimatic Crop Yield (2000-2024).csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # Root Zone Soil Wetness is 0-1 → multiply by 100
        if "root_zone_soil_wetness" in df.columns:
            df["root_zone_soil_wetness"] = pd.to_numeric(
                df["root_zone_soil_wetness"], errors="coerce"
            ) * 100
        # All Sky Surface Total PAR → divide by 2.3 for sunshine_hours
        if "all_sky_surface_total_par" in df.columns:
            df["sunshine_hours"] = pd.to_numeric(
                df["all_sky_surface_total_par"], errors="coerce"
            ) / 2.3
        # Avg temp: use earth_skin_temp if available, else try avg_temp-like cols
        temp_col = None
        for candidate in ["earth_skin_temp", "avg_temp"]:
            if candidate in df.columns:
                temp_col = candidate
                break
        min_temp_col = "min_temp" if "min_temp" in df.columns else None
        max_temp_col = "max_temp" if "max_temp" in df.columns else None
        # Rainfall: precipitation_corrected_sum
        rain_col = "precipitation_corrected_sum" if "precipitation_corrected_sum" in df.columns else None
        # Wind: max_wind_speed
        wind_col = "max_wind_speed" if "max_wind_speed" in df.columns else None
        # Humidity
        humidity_col = "humidity" if "humidity" in df.columns else None

        out = pd.DataFrame()
        out["Year"] = pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce")
        out["Country"] = "Bangladesh"
        out["avg_temp"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else float("nan")
        out["min_temp"] = pd.to_numeric(df[min_temp_col], errors="coerce") if min_temp_col else float("nan")
        out["max_temp"] = pd.to_numeric(df[max_temp_col], errors="coerce") if max_temp_col else float("nan")
        out["wind_speed_kmh"] = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else float("nan")
        out["rainfall_mm"] = pd.to_numeric(df[rain_col], errors="coerce") if rain_col else float("nan")
        out["sunshine_hours"] = df["sunshine_hours"] if "sunshine_hours" in df.columns else float("nan")
        out["soil_moisture_pct"] = df["root_zone_soil_wetness"] if "root_zone_soil_wetness" in df.columns else float("nan")
        out["humidity_pct"] = pd.to_numeric(df[humidity_col], errors="coerce") if humidity_col else float("nan")

        keep = ["Year", "Country", "avg_temp", "min_temp", "max_temp",
                "wind_speed_kmh", "rainfall_mm", "sunshine_hours",
                "soil_moisture_pct", "humidity_pct"]
        return _safe(out, keep)
    except Exception:
        return None


def _load_earth_temp() -> pd.DataFrame | None:
    try:
        path = _RAW / "earth_temp" / "GlobalLandTemperaturesByCountry.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        df["Year"] = _year_from(df["dt"])
        df = df[(df["Year"] >= 2000) & (df["Year"] <= 2020)]
        df["averagetemperature"] = pd.to_numeric(df["averagetemperature"], errors="coerce")
        if "country" in df.columns:
            df = df.rename(columns={"country": "Country"})
        out = df[["Year", "Country", "averagetemperature"]].copy()
        out = out.rename(columns={"averagetemperature": "avg_temp"})
        out = _safe(out, ["Year", "Country", "avg_temp"])
        out = out.groupby(["Year", "Country"], as_index=False)["avg_temp"].mean()
        return out
    except Exception:
        return None


def _load_env_sensor() -> pd.DataFrame | None:
    try:
        path = _RAW / "env_sensor" / "iot_telemetry_data.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # ts is unix timestamp in seconds
        df["Year"] = pd.to_datetime(
            pd.to_numeric(df["ts"], errors="coerce"), unit="s", errors="coerce"
        ).dt.year
        # CO values are raw 0-1 range → multiply by 1000 for ppb
        df["co"] = pd.to_numeric(df["co"], errors="coerce") * 1000
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        out = df[["Year", "co", "humidity", "temp"]].copy()
        out = out.rename(columns={"co": "CO_ppb", "humidity": "humidity_pct",
                                   "temp": "avg_temp", "lpg": "LPG_ppm",
                                   "smoke": "smoke_ppm"})
        out = _safe(out, ["Year", "CO_ppb", "humidity_pct", "avg_temp",
                          "LPG_ppm", "smoke_ppm"])
        out = out.groupby("Year", as_index=False).mean()
        return out
    except Exception:
        return None


def _load_dhaka_air() -> pd.DataFrame | None:
    try:
        path = _RAW / "dhaka_air" / "dhaka_air_quality_2000_2025.csv"
        df = pd.read_csv(path)
        df = _norm(df)
        # Extract year from datetime column
        df["Year"] = _year_from(df["datetime"])
        # After _norm, PM2.5 becomes pm2_5 → rename to PM25_ugm3
        col_rename = {
            "aqi": "AQI",
            "pm2_5": "PM25_ugm3",
            "pm10": "PM10_ugm3",
            "no2": "NO2_ppb",
            "so2": "SO2_ppb",
            "co": "CO_ppb",
            "o3": "O3_ppb",
            "temperature": "avg_temp",
            "humidity": "humidity_pct",
            "wind_speed": "wind_speed_kmh",
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
        df["Country"] = "Bangladesh"
        keep = ["Year", "Country", "AQI", "PM25_ugm3", "PM10_ugm3",
                "NO2_ppb", "SO2_ppb", "CO_ppb", "O3_ppb",
                "avg_temp", "humidity_pct", "wind_speed_kmh"]
        return _safe(df, keep)
    except Exception:
        return None


def _load_xls_soil() -> pd.DataFrame | None:
    try:
        # Prefer the real-time soil sensor file if present; fall back to any XLS in project
        preferred_names = ["The Real Time Soil Data"]
        candidates = list(_BASE.glob("*.xls")) + list(_BASE.glob("*.xlsx"))
        candidates += list(_RAW.rglob("*.xls")) + list(_RAW.rglob("*.xlsx"))
        if not candidates:
            return None
        # Sort so files whose name contains a preferred keyword come first
        candidates.sort(
            key=lambda p: (0 if any(kw in p.name for kw in preferred_names) else 1, p.name)
        )
        xls_path = candidates[0]
        try:
            df = pd.read_excel(xls_path, engine="xlrd")
        except Exception:
            df = pd.read_excel(xls_path)
        df = _norm(df)

        # Match by exact name first, then by startswith prefix (handles "temp(℃)", "hum(%)", etc.)
        def _find_col_prefix(df, *prefixes):
            for p in prefixes:
                if p in df.columns:
                    return p
            for p in prefixes:
                matches = [c for c in df.columns if c.startswith(p)]
                if matches:
                    return matches[0]
            return None

        year_col = _find_col_prefix(df, "year", "date", "dt", "timestamp", "time")
        temp_col = _find_col_prefix(df, "avg_temp", "temperature", "earth_skin_temp", "temp")
        hum_col  = _find_col_prefix(df, "humidity_pct", "humidity", "hum", "rh")
        ph_col   = _find_col_prefix(df, "ph", "soil_ph")
        n_col    = _find_col_prefix(df, "nitrogen_n", "nitrogen", "n(")
        p_col    = _find_col_prefix(df, "phosphorous_p", "phosphorous", "phosphorus", "p(")
        k_col    = _find_col_prefix(df, "potassium_k", "potassium", "k(")
        fert_col = _find_col_prefix(df, "soil_fertility_idx", "soil_fertility", "fertility_index",
                                    "fertility_idx", "fertility")
        cond_col = _find_col_prefix(df, "conductivity", "ec", "electrical_conductivity")

        out = pd.DataFrame()
        if year_col:
            raw_year = df[year_col]
            numeric_year = pd.to_numeric(raw_year, errors="coerce")
            if numeric_year.isna().all():
                parsed = _year_from(raw_year)
                out["Year"] = parsed.fillna(2026).astype(int)
            else:
                out["Year"] = numeric_year.fillna(2026).astype(int)
        else:
            out["Year"] = 2026

        out["Country"] = "Bangladesh"
        out["avg_temp"]         = pd.to_numeric(df[temp_col],  errors="coerce") if temp_col  else float("nan")
        out["humidity_pct"]     = pd.to_numeric(df[hum_col],   errors="coerce") if hum_col   else float("nan")
        out["soil_pH"]          = pd.to_numeric(df[ph_col],    errors="coerce") if ph_col    else float("nan")
        out["nitrogen_N"]       = pd.to_numeric(df[n_col],     errors="coerce") if n_col     else float("nan")
        out["phosphorous_P"]    = pd.to_numeric(df[p_col],     errors="coerce") if p_col     else float("nan")
        out["potassium_K"]      = pd.to_numeric(df[k_col],     errors="coerce") if k_col     else float("nan")
        out["soil_fertility_idx"] = pd.to_numeric(df[fert_col], errors="coerce") if fert_col else float("nan")
        out["conductivity"]     = pd.to_numeric(df[cond_col],  errors="coerce") if cond_col  else float("nan")

        keep = ["Year", "Country", "avg_temp", "humidity_pct", "soil_pH",
                "nitrogen_N", "phosphorous_P", "potassium_K",
                "soil_fertility_idx", "conductivity"]
        return _safe(out, keep)
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame | None]:
    """Load all datasets and return a dict mapping keys to DataFrames (or None)."""
    return {
        "fao_yield":  _load_fao_yield(),
        "fao_rain":   _load_fao_rain(),
        "fao_temp":   _load_fao_temp(),
        "crop_rec":   _load_crop_rec("crop_recommendation"),
        "crop_soil":  _load_crop_rec("crop_soil"),
        "bd_agro":    _load_bd_agro(),
        "earth_temp": _load_earth_temp(),
        "env_sensor": _load_env_sensor(),
        "dhaka_air":  _load_dhaka_air(),
        "xls_soil":   _load_xls_soil(),
    }


def print_status(datasets: dict[str, pd.DataFrame | None]) -> None:
    """Print a formatted table showing each dataset name and its shape (or 'not available')."""
    header = f"{'Dataset':<20} {'Status'}"
    print(header)
    print("-" * 50)
    for name, df in datasets.items():
        if df is None:
            status = "not available"
        else:
            status = f"{df.shape[0]} rows x {df.shape[1]} cols  |  columns: {list(df.columns)}"
        print(f"{name:<20} {status}")


# ---------------------------------------------------------------------------
# CLI entry-point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = load_all()
    print_status(datasets)

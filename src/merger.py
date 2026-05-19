
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Crop normalisation helpers
# ---------------------------------------------------------------------------

CROP_MAP = {
    "rice, paddy": "rice", "rice paddy": "rice", "paddy": "rice", "rice": "rice",
    "wheat": "wheat", "maize": "maize", "corn": "maize",
    "potatoes": "potato", "potato": "potato",
    "jute": "jute", "sugarcane": "sugarcane", "sugar cane": "sugarcane",
    "cotton lint": "cotton", "cotton": "cotton",
    "lentils": "lentil", "lentil": "lentil",
    "mango, mangosteens, guavas": "mango", "mango": "mango",
    "bananas": "banana", "banana": "banana",
    "mustard seed": "mustard", "rapeseed": "mustard",
    "groundnuts, with shell": "groundnut", "groundnuts": "groundnut",
}

CROP_SEASON = {
    "rice": "Kharif", "jute": "Kharif", "maize": "Kharif", "sugarcane": "Kharif",
    "wheat": "Rabi", "lentil": "Rabi", "potato": "Rabi", "mustard": "Rabi",
}


def _norm_crop(name: str) -> str:
    n = str(name).strip().lower()
    return CROP_MAP.get(n, n.replace(",", "").replace(" ", "_")[:20])


def _season(crop_key: str) -> str:
    return CROP_SEASON.get(crop_key, "Kharif")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(datasets: dict, bangladesh_only: bool = False) -> pd.DataFrame:
    """
    Execute the 7-step merge pipeline.

    Parameters
    ----------
    datasets : dict
        Mapping of dataset keys to DataFrames (values may be None).
    bangladesh_only : bool
        If True, restrict output to rows where Country == "Bangladesh".

    Returns
    -------
    pd.DataFrame
        Unified, feature-engineered DataFrame ready for modelling.
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Validate backbone
    # -----------------------------------------------------------------------
    print("Step 1: Validate backbone and prepare FAO yield data ...")

    if datasets.get("fao_yield") is None:
        raise ValueError("FAO yield dataset required")

    fao = datasets["fao_yield"].copy()

    # Ensure Year is numeric
    fao["Year"] = pd.to_numeric(fao["Year"], errors="coerce")
    fao = fao[(fao["Year"] >= 2000) & (fao["Year"] <= 2020)].copy()

    if bangladesh_only:
        fao = fao[fao["Country"] == "Bangladesh"].copy()

    fao["crop_key"] = fao["Crop_Type"].apply(_norm_crop)

    print(f"Step 1: Backbone ready -> {fao.shape[0]} rows, {fao.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 2 — Merge climate features (left join on Year + Country)
    # -----------------------------------------------------------------------
    print("Step 2: Merging climate features ...")

    df = fao.copy()

    # Helper: safe left merge
    def _left_merge(base: pd.DataFrame, right: pd.DataFrame | None,
                    on: list, cols: list, prefix: str) -> pd.DataFrame:
        """Merge selected columns from right into base with a prefix."""
        if right is None:
            return base
        keep = [c for c in cols if c in right.columns]
        if not keep:
            return base
        sub = right[on + keep].copy()
        rename_map = {c: f"{prefix}__{c}" for c in keep}
        sub = sub.rename(columns=rename_map)
        # drop duplicates on merge keys to avoid row explosion
        sub = sub.drop_duplicates(subset=on)
        return base.merge(sub, on=on, how="left")

    # fao_rain → rainfall_mm
    df = _left_merge(df, datasets.get("fao_rain"),
                     on=["Year", "Country"],
                     cols=["rainfall_mm"],
                     prefix="faorain")

    # fao_temp → avg_temp
    df = _left_merge(df, datasets.get("fao_temp"),
                     on=["Year", "Country"],
                     cols=["avg_temp"],
                     prefix="faotemp")

    # bd_agro → avg_temp, min_temp, max_temp, wind_speed_kmh,
    #           rainfall_mm, sunshine_hours, soil_moisture_pct, humidity_pct
    bd_agro_cols = [
        "avg_temp", "min_temp", "max_temp", "wind_speed_kmh",
        "rainfall_mm", "sunshine_hours", "soil_moisture_pct", "humidity_pct",
    ]
    df = _left_merge(df, datasets.get("bd_agro"),
                     on=["Year", "Country"],
                     cols=bd_agro_cols,
                     prefix="bdagro")

    # earth_temp → avg_temp
    df = _left_merge(df, datasets.get("earth_temp"),
                     on=["Year", "Country"],
                     cols=["avg_temp"],
                     prefix="earthtemp")

    # dhaka_air → avg_temp, humidity_pct, wind_speed_kmh
    df = _left_merge(df, datasets.get("dhaka_air"),
                     on=["Year", "Country"],
                     cols=["avg_temp", "humidity_pct", "wind_speed_kmh"],
                     prefix="dhaka")

    # ---- Consolidate duplicate climate columns via fillna priority chains ----

    def _first_non_null(df: pd.DataFrame, cols: list) -> pd.Series:
        """Return first non-null value across ordered list of columns."""
        result = pd.Series([np.nan] * len(df), index=df.index)
        for c in reversed(cols):  # lowest priority first so higher priority overwrites
            if c in df.columns:
                result = result.where(df[c].isna(), df[c])
        # Actually build from highest priority down
        result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        for c in cols:  # highest priority LAST won't work — do highest first
            pass
        # Re-implement clearly: highest priority first, fill NaN with lower priority
        series_list = [df[c] for c in cols if c in df.columns]
        if not series_list:
            return pd.Series([np.nan] * len(df), index=df.index)
        result = series_list[0].copy().astype(float)
        for s in series_list[1:]:
            result = result.fillna(s.astype(float))
        return result

    # avg_temp: bd_agro > dhaka_air > earth_temp > fao_temp
    df["avg_temp"] = _first_non_null(df, [
        "bdagro__avg_temp", "dhaka__avg_temp", "earthtemp__avg_temp", "faotemp__avg_temp"
    ])

    # rainfall_mm: bd_agro > fao_rain > dhaka_air (dhaka_air has no rainfall col here)
    df["rainfall_mm"] = _first_non_null(df, [
        "bdagro__rainfall_mm", "faorain__rainfall_mm"
    ])

    # humidity_pct: bd_agro > dhaka_air
    df["humidity_pct"] = _first_non_null(df, [
        "bdagro__humidity_pct", "dhaka__humidity_pct"
    ])

    # wind_speed_kmh: bd_agro > dhaka_air
    df["wind_speed_kmh"] = _first_non_null(df, [
        "bdagro__wind_speed_kmh", "dhaka__wind_speed_kmh"
    ])

    # single-source columns
    for col, src in [
        ("min_temp",          "bdagro__min_temp"),
        ("max_temp",          "bdagro__max_temp"),
        ("sunshine_hours",    "bdagro__sunshine_hours"),
        ("soil_moisture_pct", "bdagro__soil_moisture_pct"),
    ]:
        if src in df.columns:
            df[col] = df[src].astype(float)
        else:
            df[col] = np.nan

    # Drop all prefixed temp columns
    temp_cols = [c for c in df.columns if "__" in c]
    df = df.drop(columns=temp_cols)

    print(f"Step 2: Climate features merged -> {df.shape[0]} rows, {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 3 — Merge soil features (left join on crop_key)
    # -----------------------------------------------------------------------
    print("Step 3: Merging soil features ...")

    SOIL_AGGS = ["nitrogen_N", "phosphorous_P", "potassium_K",
                 "soil_pH", "humidity_pct", "avg_temp", "rainfall_mm"]

    def _build_soil_lookup(src: pd.DataFrame | None) -> pd.DataFrame | None:
        if src is None:
            return None
        src = src.copy()
        src["crop_key"] = src["Crop_Type"].apply(_norm_crop) if "Crop_Type" in src.columns else src.get("crop_key", pd.Series(dtype=str))
        agg_cols = [c for c in SOIL_AGGS if c in src.columns]
        if not agg_cols:
            return None
        grp = src.groupby("crop_key")[agg_cols].median().reset_index()
        return grp

    crop_rec_lookup = _build_soil_lookup(datasets.get("crop_rec"))
    crop_soil_lookup = _build_soil_lookup(datasets.get("crop_soil"))

    # Combine lookups: crop_rec as primary, fill with crop_soil
    if crop_rec_lookup is not None and crop_soil_lookup is not None:
        soil_lookup = crop_rec_lookup.set_index("crop_key").combine_first(
            crop_soil_lookup.set_index("crop_key")
        ).reset_index()
    elif crop_rec_lookup is not None:
        soil_lookup = crop_rec_lookup
    elif crop_soil_lookup is not None:
        soil_lookup = crop_soil_lookup
    else:
        soil_lookup = None

    if soil_lookup is not None:
        lookup_cols = [c for c in SOIL_AGGS if c in soil_lookup.columns]
        suffix_map = {c: f"soillkp__{c}" for c in lookup_cols}
        soil_merge = soil_lookup[["crop_key"] + lookup_cols].rename(columns=suffix_map)
        df = df.merge(soil_merge, on="crop_key", how="left")

        # Fill existing NaN values with lookup values
        for c in lookup_cols:
            src_col = f"soillkp__{c}"
            if src_col in df.columns:
                if c in df.columns:
                    df[c] = df[c].fillna(df[src_col])
                else:
                    df[c] = df[src_col]
        drop_lkp = [c for c in df.columns if c.startswith("soillkp__")]
        df = df.drop(columns=drop_lkp)

    # Apply xls_soil as Bangladesh-specific baseline for NPK/pH
    xls = datasets.get("xls_soil")
    if xls is not None:
        xls = xls.copy()
        xls["Year"] = pd.to_numeric(xls["Year"], errors="coerce")
        xls_bd = xls[xls["Country"] == "Bangladesh"].copy() if "Country" in xls.columns else xls.copy()
        xls_npk_cols = [c for c in ["nitrogen_N", "phosphorous_P", "potassium_K", "soil_pH"] if c in xls_bd.columns]
        if xls_npk_cols and "Year" in xls_bd.columns and "Country" in xls_bd.columns:
            xls_sub = xls_bd[["Year", "Country"] + xls_npk_cols].drop_duplicates(subset=["Year", "Country"])
            xls_sub = xls_sub.rename(columns={c: f"xls__{c}" for c in xls_npk_cols})
            df = df.merge(xls_sub, on=["Year", "Country"], how="left")
            for c in xls_npk_cols:
                xls_col = f"xls__{c}"
                if xls_col in df.columns:
                    if c in df.columns:
                        # Only fill NaN for Bangladesh rows
                        mask = (df["Country"] == "Bangladesh") & df[c].isna()
                        df.loc[mask, c] = df.loc[mask, xls_col]
                    else:
                        df[c] = np.nan
                        mask = df["Country"] == "Bangladesh"
                        df.loc[mask, c] = df.loc[mask, xls_col]
            drop_xls = [c for c in df.columns if c.startswith("xls__")]
            df = df.drop(columns=drop_xls)

    # soil_fertility_idx from XLS — Bangladesh-specific field data
    if xls is not None:
        xls_bd2 = xls[xls.get("Country", pd.Series()) == "Bangladesh"].copy() \
                  if "Country" in xls.columns else xls.copy()
        if "soil_fertility_idx" in xls_bd2.columns:
            val = pd.to_numeric(xls_bd2["soil_fertility_idx"], errors="coerce").mean()
            if not np.isnan(val):
                if "soil_fertility_idx" not in df.columns:
                    df["soil_fertility_idx"] = np.nan
                mask = df["Country"] == "Bangladesh"
                df.loc[mask, "soil_fertility_idx"] = df.loc[mask, "soil_fertility_idx"].fillna(val)

    # Ensure soil columns exist
    SOIL_SENTINELS = {
        "nitrogen_N": np.nan, "phosphorous_P": np.nan,
        "potassium_K": np.nan, "soil_pH": np.nan,
        "soil_fertility_idx": np.nan,
    }
    for col, val in SOIL_SENTINELS.items():
        if col not in df.columns:
            df[col] = val

    print(f"Step 3: Soil features merged -> {df.shape[0]} rows, {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 4 — Merge environmental features
    # -----------------------------------------------------------------------
    print("Step 4: Merging environmental features ...")

    # dhaka_air: AQI, PM25_ugm3, NO2_ppb, CO2_ppm proxy
    dhaka_air = datasets.get("dhaka_air")
    if dhaka_air is not None:
        dh_env_cols = [c for c in ["AQI", "PM25_ugm3", "PM10_ugm3", "NO2_ppb", "SO2_ppb", "CO_ppb", "O3_ppb"]
                       if c in dhaka_air.columns]
        if dh_env_cols and "Year" in dhaka_air.columns:
            merge_keys = ["Year", "Country"] if "Country" in dhaka_air.columns else ["Year"]
            dh_sub = dhaka_air[merge_keys + dh_env_cols].drop_duplicates(subset=merge_keys)
            # Rename CO_ppb to CO2_ppm proxy if CO2 not present
            if "CO_ppb" in dh_sub.columns and "CO2_ppm" not in dh_sub.columns:
                dh_sub = dh_sub.rename(columns={"CO_ppb": "CO2_ppm"})
                if "CO2_ppm" in dh_env_cols:
                    dh_env_cols = [c if c != "CO_ppb" else "CO2_ppm" for c in dh_env_cols]
            df = df.merge(dh_sub, on=merge_keys, how="left")

    # env_sensor: CO_ppb + LPG_ppm + smoke_ppm
    env_sensor = datasets.get("env_sensor")
    if env_sensor is not None and "Year" in env_sensor.columns:
        env_cols = [c for c in ["CO_ppb", "humidity_pct", "avg_temp",
                                "LPG_ppm", "smoke_ppm"]
                    if c in env_sensor.columns]
        if env_cols:
            env_sub = env_sensor[["Year"] + env_cols].drop_duplicates(subset=["Year"])
            env_sub = env_sub.rename(columns={c: f"envsens__{c}" for c in env_cols})
            df = df.merge(env_sub, on="Year", how="left")

            if "envsens__CO_ppb" in df.columns:
                if "CO2_ppm" not in df.columns:
                    df["CO2_ppm"] = df["envsens__CO_ppb"]
                else:
                    df["CO2_ppm"] = df["CO2_ppm"].fillna(df["envsens__CO_ppb"])

            # LPG and smoke as new environmental features
            for ec in ["LPG_ppm", "smoke_ppm"]:
                src = f"envsens__{ec}"
                if src in df.columns:
                    df[ec] = df[src]

            drop_env = [c for c in df.columns if c.startswith("envsens__")]
            df = df.drop(columns=drop_env)

    # Ensure all env columns exist (sentinel fill)
    ENV_SENTINELS = {
        "AQI": 150.0, "PM25_ugm3": 25.0, "NO2_ppb": 20.0,
        "CO2_ppm": 403.0, "LPG_ppm": 0.0, "smoke_ppm": 0.0,
    }
    for col, sentinel in ENV_SENTINELS.items():
        if col not in df.columns:
            df[col] = sentinel

    print(f"Step 4: Environmental features merged -> {df.shape[0]} rows, {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 5 — Feature engineering
    # -----------------------------------------------------------------------
    print("Step 5: Feature engineering ...")

    df["Year"] = df["Year"].astype(int)
    df["decade"] = (df["Year"] // 10) * 10

    # Ensure avg_temp exists as numeric
    if "avg_temp" not in df.columns:
        df["avg_temp"] = np.nan
    df["avg_temp"] = pd.to_numeric(df["avg_temp"], errors="coerce")

    # rainfall_mm as numeric
    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = np.nan
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # min_temp / max_temp
    if "min_temp" not in df.columns:
        df["min_temp"] = np.nan
    df["min_temp"] = pd.to_numeric(df["min_temp"], errors="coerce")
    df["min_temp"] = df["min_temp"].fillna(df["avg_temp"] - 6.0)

    if "max_temp" not in df.columns:
        df["max_temp"] = np.nan
    df["max_temp"] = pd.to_numeric(df["max_temp"], errors="coerce")
    df["max_temp"] = df["max_temp"].fillna(df["avg_temp"] + 6.0)

    df["temperature_range"] = df["max_temp"] - df["min_temp"]

    # season
    df["season"] = df["crop_key"].apply(_season)

    # sunshine_hours
    if "sunshine_hours" not in df.columns:
        df["sunshine_hours"] = np.nan
    df["sunshine_hours"] = pd.to_numeric(df["sunshine_hours"], errors="coerce")
    synthetic_sun = (6.5 + (df["avg_temp"] - 22) * 0.12).clip(3.5, 11)
    df["sunshine_hours"] = df["sunshine_hours"].fillna(synthetic_sun)

    # wind_speed_kmh
    if "wind_speed_kmh" not in df.columns:
        df["wind_speed_kmh"] = np.nan
    df["wind_speed_kmh"] = pd.to_numeric(df["wind_speed_kmh"], errors="coerce")
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(10.0)

    # soil_moisture_pct
    if "soil_moisture_pct" not in df.columns:
        df["soil_moisture_pct"] = np.nan
    df["soil_moisture_pct"] = pd.to_numeric(df["soil_moisture_pct"], errors="coerce")
    synthetic_moisture = (df["rainfall_mm"] / 40).clip(10, 85)
    df["soil_moisture_pct"] = df["soil_moisture_pct"].fillna(synthetic_moisture)

    # soil_type
    df["soil_type"] = df["Country"].apply(
        lambda c: "Alluvial" if c == "Bangladesh" else "Loam"
    )

    # fertilizer_kgha
    npk_cols = ["nitrogen_N", "phosphorous_P", "potassium_K"]
    for col in npk_cols:
        df[col] = pd.to_numeric(df.get(col, pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    df["fertilizer_kgha"] = df[npk_cols].sum(axis=1, skipna=True)

    # CO2_ppm
    df["CO2_ppm"] = pd.to_numeric(df["CO2_ppm"], errors="coerce")
    df["CO2_ppm"] = df["CO2_ppm"].fillna(403.0)

    # AQI
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df["AQI"] = df["AQI"].fillna(150.0)

    # PM25_ugm3
    df["PM25_ugm3"] = pd.to_numeric(df["PM25_ugm3"], errors="coerce")
    df["PM25_ugm3"] = df["PM25_ugm3"].fillna(25.0)

    # NO2_ppb
    df["NO2_ppb"] = pd.to_numeric(df["NO2_ppb"], errors="coerce")
    df["NO2_ppb"] = df["NO2_ppb"].fillna(20.0)

    # humidity_pct
    df["humidity_pct"] = pd.to_numeric(df["humidity_pct"], errors="coerce")
    df["humidity_pct"] = df["humidity_pct"].fillna(65.0)

    # rainfall_category
    df["rainfall_category"] = pd.cut(
        df["rainfall_mm"],
        bins=[0, 500, 1500, float("inf")],
        labels=["Low", "Medium", "High"],
        right=True,
    )

    print(f"Step 5: Feature engineering done -> {df.shape[0]} rows, {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 6 — Handle missing values
    # -----------------------------------------------------------------------
    print("Step 6: Handling missing values ...")

    # Drop columns where > 90% values are NaN
    thresh = 0.90
    null_frac = df.isnull().mean()
    drop_high_null = null_frac[null_frac > thresh].index.tolist()
    if drop_high_null:
        print(f"  Dropping {len(drop_high_null)} high-null columns: {drop_high_null}")
        df = df.drop(columns=drop_high_null)

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Median impute numeric
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Mode impute categorical (skip crop_key — dropped below)
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    # Drop crop_key
    if "crop_key" in df.columns:
        df = df.drop(columns=["crop_key"])

    print(f"Step 6: Missing values handled -> {df.shape[0]} rows, {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # STEP 7 — Save and report
    # -----------------------------------------------------------------------
    print("Step 7: Saving merged dataset ...")

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "merged_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"Step 7: Saved -> {out_path}")
    print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns ({df.shape[1]}): {list(df.columns)}")

    return df

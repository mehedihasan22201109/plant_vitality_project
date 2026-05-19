#!/usr/bin/env bash
# ============================================================
# Dataset Downloader — All 10 Tabular + 4 Image Datasets
# Author: Mahinur Akhter (22201100)
#
# Usage (WSL):
#   bash download_datasets.sh
# ============================================================

set -e

PROJECT_DIR="/mnt/d/4-1"
RAW_DIR="$PROJECT_DIR/data/raw"
IMG_DIR="$PROJECT_DIR/data/plant_images"
KAGGLE_DIR="$HOME/.kaggle"

mkdir -p "$RAW_DIR" "$IMG_DIR"

# ── Setup kaggle credentials ──────────────────────────────────────────────────
mkdir -p "$KAGGLE_DIR"
if [ ! -f "$KAGGLE_DIR/kaggle.json" ]; then
    cp "$PROJECT_DIR/kaggle.json" "$KAGGLE_DIR/kaggle.json"
fi
chmod 600 "$KAGGLE_DIR/kaggle.json"

cd "$PROJECT_DIR"

echo "============================================================"
echo "  Downloading 10 Tabular Datasets"
echo "============================================================"

# ── DS1: Crop Yield FAO World Data ───────────────────────────────────────────
echo "[1/10] Crop Yield FAO World Data..."
mkdir -p "$RAW_DIR/crop_yield_fao"
if [ ! -f "$RAW_DIR/crop_yield_fao/yield.csv" ]; then
    kaggle datasets download -d patelris/crop-yield-prediction-dataset \
        -p "$RAW_DIR/crop_yield_fao" --unzip 2>/dev/null || \
    kaggle datasets download -d divyansh22/crop-yield-prediction-dataset \
        -p "$RAW_DIR/crop_yield_fao" --unzip 2>/dev/null || \
    kaggle datasets download -d andreshg/unifiedcropyieldprediction \
        -p "$RAW_DIR/crop_yield_fao" --unzip 2>/dev/null || \
    echo "  [WARN] DS1 download failed — already exists or try manually"
else
    echo "  [SKIP] DS1 already downloaded"
fi

# ── DS2: Crop Recommendation ─────────────────────────────────────────────────
echo "[2/10] Crop Recommendation Dataset..."
mkdir -p "$RAW_DIR/crop_recommendation"
if [ ! -f "$RAW_DIR/crop_recommendation/Crop_recommendation.csv" ]; then
    kaggle datasets download -d atharvaingle/crop-recommendation-dataset \
        -p "$RAW_DIR/crop_recommendation" --unzip 2>/dev/null || \
    kaggle datasets download -d theeyeschic/crop-analysis-and-prediction \
        -p "$RAW_DIR/crop_recommendation" --unzip 2>/dev/null || \
    echo "  [WARN] DS2 download failed"
else
    echo "  [SKIP] DS2 already downloaded"
fi

# ── DS3: Crop and Soil Dataset ────────────────────────────────────────────────
echo "[3/10] Crop and Soil Dataset..."
mkdir -p "$RAW_DIR/crop_soil"
if [ -z "$(ls -A $RAW_DIR/crop_soil 2>/dev/null)" ]; then
    kaggle datasets download -d mahmoudredagamal/crop-and-soil-dataset \
        -p "$RAW_DIR/crop_soil" --unzip 2>/dev/null || \
    kaggle datasets download -d aksahaha/crop-recommendation \
        -p "$RAW_DIR/crop_soil" --unzip 2>/dev/null || \
    kaggle datasets download -d varshitanalluri/crop-recommendation-dataset \
        -p "$RAW_DIR/crop_soil" --unzip 2>/dev/null || \
    echo "  [WARN] DS3 not available — will skip in pipeline"
else
    echo "  [SKIP] DS3 already downloaded"
fi

# ── DS4: Agricultural Land Suitability Bangladesh ────────────────────────────
echo "[4/10] Agricultural Land Suitability Bangladesh..."
mkdir -p "$RAW_DIR/land_suitability_bd"
if [ -z "$(ls -A $RAW_DIR/land_suitability_bd 2>/dev/null)" ]; then
    kaggle datasets download -d devraai/agricultural-land-suitability-in-bangladesh \
        -p "$RAW_DIR/land_suitability_bd" --unzip 2>/dev/null || \
    echo "  [WARN] DS4 not available — will skip in pipeline"
else
    echo "  [SKIP] DS4 already downloaded"
fi

# ── DS5: Bangladesh Agroclimatic Crop Yield 2000-2024 ────────────────────────
echo "[5/10] Bangladesh Agroclimatic Crop Yield..."
mkdir -p "$RAW_DIR/bd_agroclimatic"
if [ -z "$(ls -A $RAW_DIR/bd_agroclimatic 2>/dev/null)" ]; then
    kaggle datasets download -d firozemaliha/bangladesh-agroclimatic-crop-yield-2000-2024 \
        -p "$RAW_DIR/bd_agroclimatic" --unzip 2>/dev/null || \
    echo "  [WARN] DS5 download failed"
else
    echo "  [SKIP] DS5 already downloaded"
fi

# ── DS6: Earth Surface Temperature ───────────────────────────────────────────
echo "[6/10] Climate Change Earth Surface Temperature..."
mkdir -p "$RAW_DIR/earth_temp"
if [ -z "$(ls -A $RAW_DIR/earth_temp 2>/dev/null)" ]; then
    kaggle datasets download -d berkeleyearth/climate-change-earth-surface-temperature-data \
        -p "$RAW_DIR/earth_temp" --unzip 2>/dev/null || \
    echo "  [WARN] DS6 download failed"
else
    echo "  [SKIP] DS6 already downloaded"
fi

# ── DS7: Climate Data Bangladesh 2021-2024 ────────────────────────────────────
echo "[7/10] Climate Data for Bangladesh (2021-2024)..."
mkdir -p "$RAW_DIR/bd_climate"
if [ -z "$(ls -A $RAW_DIR/bd_climate 2>/dev/null)" ]; then
    kaggle datasets download -d azrulm/climate-data-for-b-d \
        -p "$RAW_DIR/bd_climate" --unzip 2>/dev/null || \
    kaggle datasets download -d mnassrib/telecom-churn-datasets \
        -p "/dev/null" 2>/dev/null || true  # dummy to test auth
    echo "  [WARN] DS7 not available — will skip in pipeline"
else
    echo "  [SKIP] DS7 already downloaded"
fi

# ── DS8: Bangladesh Weather Dataset 1901-2023 ─────────────────────────────────
echo "[8/10] Bangladesh Weather Dataset..."
mkdir -p "$RAW_DIR/bd_weather"
if [ -z "$(ls -A $RAW_DIR/bd_weather 2>/dev/null)" ]; then
    kaggle datasets download -d yakinrubaiai/bangladesh-weather-dataset \
        -p "$RAW_DIR/bd_weather" --unzip 2>/dev/null || \
    echo "  [WARN] DS8 not available — will skip in pipeline"
else
    echo "  [SKIP] DS8 already downloaded"
fi

# ── DS9: Environmental Sensor Telemetry Data ──────────────────────────────────
echo "[9/10] Environmental Sensor Telemetry Data..."
mkdir -p "$RAW_DIR/env_sensor"
if [ -z "$(ls -A $RAW_DIR/env_sensor 2>/dev/null)" ]; then
    kaggle datasets download -d garystafford/environmental-sensor-data-132k \
        -p "$RAW_DIR/env_sensor" --unzip 2>/dev/null || \
    echo "  [WARN] DS9 download failed"
else
    echo "  [SKIP] DS9 already downloaded"
fi

# ── DS10: Dhaka Air Quality 2000-2025 ────────────────────────────────────────
echo "[10/10] Dhaka Air Quality 2000-2025..."
mkdir -p "$RAW_DIR/dhaka_air"
if [ -z "$(ls -A $RAW_DIR/dhaka_air 2>/dev/null)" ]; then
    kaggle datasets download -d shakil10945/dhaka-air-quality-2000-2025-synthetic-dataset \
        -p "$RAW_DIR/dhaka_air" --unzip 2>/dev/null || \
    echo "  [WARN] DS10 download failed"
else
    echo "  [SKIP] DS10 already downloaded"
fi

echo ""
echo "============================================================"
echo "  Downloading 4 Image Datasets"
echo "============================================================"

# ── IMG1: PlantVillage ────────────────────────────────────────────────────────
echo "[IMG 1/4] PlantVillage Dataset..."
mkdir -p "$IMG_DIR/plantvillage"
if [ -z "$(find $IMG_DIR/plantvillage -name '*.jpg' -o -name '*.JPG' 2>/dev/null | head -1)" ]; then
    kaggle datasets download -d vipoooool/new-plant-diseases-dataset \
        -p "$IMG_DIR/plantvillage" --unzip 2>/dev/null || \
    echo "  [WARN] IMG1 download failed"
else
    echo "  [SKIP] PlantVillage already downloaded"
fi

# ── IMG2: Rice Leaf Disease ───────────────────────────────────────────────────
echo "[IMG 2/4] Rice Leaf Disease Dataset..."
mkdir -p "$IMG_DIR/rice_disease"
if [ -z "$(ls -A $IMG_DIR/rice_disease 2>/dev/null)" ]; then
    kaggle datasets download -d minhhuy210/rice-diseases-image-dataset \
        -p "$IMG_DIR/rice_disease" --unzip 2>/dev/null || \
    kaggle datasets download -d shayanriyaz/riceleafdisease \
        -p "$IMG_DIR/rice_disease" --unzip 2>/dev/null || \
    echo "  [WARN] IMG2 download failed"
else
    echo "  [SKIP] Rice Disease already downloaded"
fi

# ── IMG3: Crop Disease Detection ─────────────────────────────────────────────
echo "[IMG 3/4] Crop Disease Detection Dataset..."
mkdir -p "$IMG_DIR/crop_disease"
if [ -z "$(ls -A $IMG_DIR/crop_disease 2>/dev/null)" ]; then
    kaggle datasets download -d nirmalsankalana/crop-disease-detection-dataset \
        -p "$IMG_DIR/crop_disease" --unzip 2>/dev/null || \
    echo "  [WARN] IMG3 download failed"
else
    echo "  [SKIP] Crop Disease already downloaded"
fi

# ── IMG4: Plant Seedlings ─────────────────────────────────────────────────────
echo "[IMG 4/4] Plant Seedlings Classification..."
mkdir -p "$IMG_DIR/seedlings"
if [ -z "$(ls -A $IMG_DIR/seedlings 2>/dev/null)" ]; then
    kaggle competitions download -c plant-seedlings-classification \
        -p "$IMG_DIR/seedlings" 2>/dev/null || \
    kaggle datasets download -d vikramtiwari/plant-seedlings \
        -p "$IMG_DIR/seedlings" --unzip 2>/dev/null || \
    echo "  [WARN] IMG4 download failed"
else
    echo "  [SKIP] Plant Seedlings already downloaded"
fi

echo ""
echo "============================================================"
echo "  Download complete. Summary:"
echo "  Tabular data -> $RAW_DIR"
echo "  Image data   -> $IMG_DIR"
echo ""
echo "  Now run the full pipeline:"
echo "  python3 main.py"
echo "============================================================"

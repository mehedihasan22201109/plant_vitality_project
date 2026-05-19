#!/usr/bin/env bash
# ============================================================
# MASTER SCRIPT — Full Pipeline (One Command)
#
# Usage:
#   bash /mnt/d/4-1/run_all.sh              # full (with CNN)
#   bash /mnt/d/4-1/run_all.sh --skip-cnn   # ML only, no TensorFlow
#   bash /mnt/d/4-1/run_all.sh --images     # also download image datasets
# ============================================================

PROJECT="/mnt/d/4-1"
RAW="$PROJECT/data/raw"
IMG="$PROJECT/data/plant_images"
VENV="$PROJECT/venv"
SKIP_CNN=false
DOWNLOAD_IMAGES=false

for arg in "$@"; do
  [[ "$arg" == "--skip-cnn"  ]] && SKIP_CNN=true
  [[ "$arg" == "--images"    ]] && DOWNLOAD_IMAGES=true
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${YELLOW}[...]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }

echo ""
echo "============================================================"
echo "  CROP YIELD PREDICTION & PLANT DISEASE DETECTION"
echo "============================================================"
echo ""

# ── Step 1: Kaggle credentials ───────────────────────────────
info "Step 1: Kaggle credentials..."
mkdir -p ~/.kaggle
cp "$PROJECT/kaggle.json" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
ok "Credentials set (noushad999)"

# ── Step 2: Virtual environment ──────────────────────────────
info "Step 2: Python virtual environment..."
cd "$PROJECT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
ok "Virtual environment ready"

# ── Step 3: Install packages ─────────────────────────────────
info "Step 3: Installing packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    pandas numpy scikit-learn xgboost \
    matplotlib seaborn joblib scipy \
    xlrd openpyxl kaggle Pillow opencv-python

if [ "$SKIP_CNN" = false ]; then
    info "  Installing TensorFlow..."
    pip install --quiet tensorflow 2>/dev/null \
        && ok "  TensorFlow installed" \
        || warn "  TensorFlow install failed — CNN will be skipped"
fi
ok "Packages installed"

# ── Step 4: Directories ──────────────────────────────────────
info "Step 4: Creating directories..."
mkdir -p "$RAW/crop_yield_fao"    "$RAW/crop_recommendation"
mkdir -p "$RAW/bd_agroclimatic"   "$RAW/earth_temp"
mkdir -p "$RAW/env_sensor"        "$RAW/dhaka_air"
mkdir -p "$IMG/plantvillage"      "$IMG/rice_disease"
mkdir -p "$IMG/crop_disease"      "$IMG/seedlings"
mkdir -p "$PROJECT/outputs"       "$PROJECT/models"
mkdir -p "$PROJECT/data/processed"
ok "Directories ready"

# ── Step 5: Download datasets (skip if already downloaded) ───
info "Step 5: Downloading tabular datasets..."
echo ""

dl() {
    local label="$1" slug="$2" dest="$3"
    if [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        ok "  $label (already downloaded)"
        return
    fi
    info "  $label"
    kaggle datasets download -d "$slug" -p "$dest" --unzip -q 2>/dev/null \
        && ok "  $label" \
        || warn "  $label -- FAILED (synthetic fallback)"
}

dl "DS1: Crop Yield FAO"         "patelris/crop-yield-prediction-dataset"                    "$RAW/crop_yield_fao"
dl "DS2: Crop Recommendation"    "atharvaingle/crop-recommendation-dataset"                  "$RAW/crop_recommendation"
dl "DS5: BD Agroclimatic Yield"  "firozemaliha/bangladesh-agroclimatic-crop-yield-2000-2024" "$RAW/bd_agroclimatic"
dl "DS6: Earth Surface Temp"     "berkeleyearth/climate-change-earth-surface-temperature-data" "$RAW/earth_temp"
dl "DS9: Env Sensor IoT"         "garystafford/environmental-sensor-data-132k"               "$RAW/env_sensor"
dl "DS10: Dhaka Air Quality"     "shakil10945/dhaka-air-quality-2000-2025-synthetic-dataset"  "$RAW/dhaka_air"

echo ""
ok "Tabular datasets ready"

# ── Step 6: Image datasets (--images flag only) ───────────────
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo ""
    info "Step 6: Downloading image datasets (may take 30-60 min)..."
    dl "IMG1: PlantVillage"       "vipoooool/new-plant-diseases-dataset"          "$IMG/plantvillage"
    dl "IMG2: Rice Leaf Disease"  "minhhuy210/rice-diseases-image-dataset"         "$IMG/rice_disease"
    dl "IMG3: Crop Disease"       "nirmalsankalana/crop-disease-detection-dataset" "$IMG/crop_disease"
    ok "Image datasets done"
else
    info "Step 6: Image download skipped (add --images flag to download)"
    info "        CNN will run with synthetic images"
fi

# ── Step 7: Run full pipeline ─────────────────────────────────
echo ""
info "Step 7: Running ML pipeline..."
cd "$PROJECT"
source "$VENV/bin/activate"

if [ "$SKIP_CNN" = true ]; then
    python main.py --skip-cnn
else
    python main.py
fi

echo ""
echo "============================================================"
echo -e "${GREEN}  ALL DONE!${NC}"
echo "============================================================"
echo "  Outputs -> $PROJECT/outputs/"
echo "  Models  -> $PROJECT/models/"
echo "  CSV     -> $PROJECT/data/processed/merged_dataset.csv"
echo ""
echo "  Plots saved:"
ls "$PROJECT/outputs/"*.png 2>/dev/null | sed 's|.*/|    |'
echo ""
echo "  Models saved:"
ls "$PROJECT/models/"*.pkl "$PROJECT/models/"*.keras 2>/dev/null | sed 's|.*/|    |'
echo "============================================================"

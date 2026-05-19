from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUTS_DIR   = Path("outputs")
IMG_DIR       = Path("data/plant_images")
XLS_PATH      = Path("parameter report(10).xls")

# Create dirs on import
for _d in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── ML / Tabular ──────────────────────────────────────────────────────────────
TARGET = "yield_hgha"

FEATURE_COLS = [
    "Year", "avg_temp", "rainfall_mm", "humidity_pct",
    "min_temp", "max_temp", "wind_speed_kmh", "sunshine_hours", "season",
    "nitrogen_N", "phosphorous_P", "potassium_K", "soil_pH",
    "soil_moisture_pct", "soil_type", "fertilizer_kgha",
    "AQI", "CO2_ppm", "PM25_ugm3", "NO2_ppb",
    "LPG_ppm", "smoke_ppm",
    "decade", "temperature_range",
]

# Extended feature set including image fusion features (when available)
FUSION_FEATURE_COLS = FEATURE_COLS + [
    "img_avg_confidence", "img_avg_severity",
    "img_avg_yellowing",  "img_avg_browning",
    "img_avg_texture",    "img_disease_rate",
]

YIELD_BINS   = [0, 20_000, 50_000, float("inf")]
YIELD_LABELS = ["Low", "Medium", "High"]

# ── CNN ───────────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 5
EPOCHS_FINE   = 10

# ── Crop normalisation map ────────────────────────────────────────────────────
CROP_MAP = {
    "rice, paddy": "rice", "rice paddy": "rice", "paddy": "rice", "rice": "rice",
    "wheat": "wheat",
    "maize": "maize", "corn": "maize",
    "potatoes": "potato", "potato": "potato",
    "sweet potatoes": "sweet_potato",
    "jute": "jute",
    "sugarcane": "sugarcane", "sugar cane": "sugarcane",
    "sorghum": "sorghum",
    "cotton lint": "cotton", "cotton": "cotton",
    "lentils": "lentil", "lentil": "lentil",
    "chickpeas": "chickpea", "chickpea": "chickpea",
    "mango, mangosteens, guavas": "mango", "mango": "mango",
    "bananas": "banana", "banana": "banana",
    "coconuts": "coconut", "coconut": "coconut",
    "mustard seed": "mustard", "rapeseed": "mustard",
    "groundnuts, with shell": "groundnut", "groundnuts": "groundnut",
    "tea": "tea",
}

CROP_SEASON = {
    "rice": "Kharif", "jute": "Kharif", "maize": "Kharif",
    "sugarcane": "Kharif", "cotton": "Kharif", "groundnut": "Kharif",
    "mango": "Kharif", "banana": "Kharif", "coconut": "Kharif",
    "wheat": "Rabi", "lentil": "Rabi", "potato": "Rabi",
    "mustard": "Rabi", "chickpea": "Rabi", "sweet_potato": "Rabi",
    "tea": "Rabi",
}

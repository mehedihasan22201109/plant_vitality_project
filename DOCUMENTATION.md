# Crop Yield Prediction & Plant Disease Detection — Complete Documentation

**Version:** 1.0
**Authors:** Mahinur Akhter (22201100), Halima Akter Shorna (22201101), Mehedi Hasan (22201109)
**Course:** CSE 4101 — Machine Learning Project
**Institution:** Department of CSE, University of Asia Pacific (UAP)
**Supervisor:** Faria Zarin Subah, Assistant Professor

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Datasets](#6-datasets)
7. [Configuration](#7-configuration)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Pipeline Stages — Step by Step](#9-pipeline-stages--step-by-step)
10. [Module Reference](#10-module-reference)
11. [Trained Models](#11-trained-models)
12. [Output Files](#12-output-files)
13. [Empirical Results](#13-empirical-results)
14. [Reproducing the Report](#14-reproducing-the-report)
15. [Troubleshooting](#15-troubleshooting)
16. [FAQ](#16-faq)
17. [License & Acknowledgements](#17-license--acknowledgements)

---

## 1. Project Overview

This project is an end-to-end machine-learning system that performs two related agricultural tasks simultaneously:

1. **Crop Yield Prediction** — given climatic, soil, and environmental conditions, predict the continuous crop yield (hg/ha) and classify it into Low / Medium / High categories.
2. **Plant Disease Detection** — given a leaf image, classify it into one of 38 disease classes using a fine-tuned CNN (ResNet-50 / MobileNetV2 backbone), with Grad-CAM visual explanations.

The two branches are unified through a **late-fusion layer**: image-derived features (disease class, confidence, severity, color statistics, texture) are aggregated per crop and concatenated to the 22-feature tabular input, producing a richer representation for downstream models.

The pipeline integrates **10 tabular datasets** (FAO, Kaggle, IoT sensors) and **4 image datasets** (PlantVillage, Rice Leaf Disease, Crop Disease, Plant Seedlings) — four of the tabular sources are Bangladesh-specific.

### Use cases
- Academic research on multi-source data integration in agriculture.
- Decision support for agricultural extension workers in Bangladesh.
- Foundation for a future mobile/web farmer-advisory application.

---

## 2. Key Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **8-step merging pipeline** | Unifies 10 heterogeneous datasets into one 22-feature DataFrame. |
| 2 | **Multiple regressors** | Linear Regression, Random Forest, XGBoost, KNN. |
| 3 | **Multiple classifiers** | Random Forest, XGBoost, KNN, Naive Bayes. |
| 4 | **Two-phase CNN training** | Phase 1 (frozen backbone) → Phase 2 (full fine-tune). |
| 5 | **Grad-CAM explainability** | Localises the diseased region of a leaf image. |
| 6 | **SHAP analysis** | Per-feature contribution to yield predictions (bar / beeswarm / waterfall). |
| 7 | **Late-fusion crop profile** | Combines image-derived features with tabular yield features. |
| 8 | **Bangladesh-only mode** | A `--bangladesh` flag filters all data to Bangladesh-specific rows. |
| 9 | **Synthetic mode** | A `--synthetic` flag uses generated data for fast smoke-testing. |
| 10 | **Self-contained reports** | Auto-generated PDF / DOCX / LaTeX reports with all figures & tables. |

---

## 3. System Architecture

```
┌───────────────────┐   ┌───────────────────┐   ┌────────────────────┐   ┌──────────────┐
│ Climate Datasets  │   │  Soil Datasets    │   │  Environmental     │   │ Image Data   │
│ DS1 / 6 / 7 / 8   │   │  DS2 / 3 / 4      │   │  DS9 / DS10        │   │ (PV / Rice)  │
└────────┬──────────┘   └────────┬──────────┘   └─────────┬──────────┘   └──────┬───────┘
         ▼                       ▼                        ▼                     ▼
   ┌──────────┐            ┌──────────┐             ┌───────────┐         ┌──────────┐
   │  Merge   │            │  Merge   │             │   Merge   │         │ ResNet-50│
   │ Climate  │            │  Soil    │             │  Env.     │         │   CNN    │
   └────┬─────┘            └────┬─────┘             └─────┬─────┘         └────┬─────┘
        └────────────┬──────────┴───────────────┬─────────┘                    │
                     ▼                          │                              │
              ┌─────────────┐                   │                              │
              │ Final Join  │ (Year, Country,   │                              │
              │             │  Crop_Type)       │                              │
              └──────┬──────┘                   │                              │
                     ▼                          │                              │
              ┌─────────────┐                   │                              │
              │  Impute &   │                   │                              │
              │  Engineer   │                   │                              │
              └──────┬──────┘                   │                              │
                     ▼                          ▼                              ▼
            ┌──────────────────┐       ┌──────────────────┐         ┌────────────────┐
            │  Regression      │       │ Classification   │         │   Disease      │
            │  RMSE / MAE / R² │       │ Accuracy / F1    │         │   Detection    │
            └──────────────────┘       └──────────────────┘         └────────────────┘
                     │                          │                              │
                     └──────────────┬───────────┘                              │
                                    ▼                                          │
                         ┌──────────────────────┐                              │
                         │   Late-Fusion Layer  │ ◄────────────────────────────┘
                         │   (per-crop profile) │
                         └──────────────────────┘
```

---

## 4. Project Structure

```
D:\4-1\
├── data/
│   ├── raw/                       # Raw downloaded datasets
│   ├── processed/                 # Merged DataFrame (parquet/CSV)
│   ├── plant_images/              # PlantVillage etc.
│   └── test_images/               # Test images for inference
├── src/
│   ├── config.py                  # Constants, paths, feature list
│   ├── loaders.py                 # Dataset-specific loaders
│   ├── merger.py                  # 8-step merging pipeline
│   ├── models.py                  # Regressors & classifiers
│   ├── cnn.py                     # CNN model definition + training
│   ├── fusion.py                  # Late-fusion crop profile
│   ├── explainability.py          # SHAP, Grad-CAM, feature importance
│   └── utils.py                   # Helpers (logging, plotting)
├── models/
│   ├── best_regressor.pkl         # XGBoost regressor (best)
│   ├── best_classifier.pkl        # XGBoost classifier (best)
│   ├── all_regressors.pkl         # All four regressors
│   ├── all_classifiers.pkl        # All four classifiers
│   ├── plant_disease_cnn.keras    # Fine-tuned CNN (ResNet-50)
│   ├── preprocessor.pkl           # ColumnTransformer
│   └── feature_names.pkl          # Saved feature names
├── outputs/
│   ├── *.png                      # All figures (EDA, results, SHAP, Grad-CAM)
│   ├── *.csv                      # Tabulated results (regression, classification, fusion)
│   └── classification_report.txt  # Per-class CNN metrics
├── report/                        # Generated reports (PDF/DOCX)
├── report_latex/                  # LaTeX source + compiled PDF
├── main.py                        # Top-level pipeline entry-point
├── predict.py                     # CLI for single-image / single-row inference
├── extract_image_features.py      # Image feature extraction utility
├── generate_report.py             # PDF/DOCX report generator
├── make_report_doc.py             # Alternate doc generator
├── test_models.py                 # Model unit tests
├── download_datasets.sh           # Kaggle download script
├── run_all.sh                     # End-to-end shell runner
├── requirements.txt               # Python dependencies
├── kaggle.json                    # Kaggle API credentials
├── icon.png                       # UAP logo (used in cover page)
├── DOCUMENTATION.md               # ← This file
└── report.tex                     # LaTeX research paper source
```

---

## 5. Installation & Setup

### Prerequisites

- **Python** 3.10 or higher
- **pip** (or `uv` / `poetry` if preferred)
- **Kaggle account** with API credentials (for dataset downloads)
- **GPU recommended** for CNN training (CPU works but is slow)
- **OS:** Windows, Linux, or macOS (paths use `pathlib`, so cross-platform)

### Step 1 — Clone / Set up project directory

```bash
cd D:\4-1
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS / WSL
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

The full dependency list:

```
pandas>=1.5.0       numpy>=1.23.0       scikit-learn>=1.2.0
xgboost>=1.7.0      matplotlib>=3.6.0   seaborn>=0.12.0
tensorflow>=2.11.0  Pillow>=9.4.0       opencv-python>=4.7.0
kaggle>=1.5.13      openpyxl>=3.1.0     xlrd>=2.0.1
joblib>=1.2.0       scipy>=1.10.0       jupyter>=1.0.0
```

### Step 4 — Configure Kaggle credentials

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 5 — Download datasets

```bash
bash download_datasets.sh
```

Or download manually from the URLs in Section 6.

---

## 6. Datasets

### 6.1 Tabular Datasets

| # | Dataset | Domain | Years | Rows | Source |
|---|---------|--------|-------|------|--------|
| 1 | Crop Yield FAO World Data | Soil + Climate | 1961–2016 (55 yr) | ~56,717 | kaggle.com/patleris/crop-yield-eda-viz |
| 2 | Crop Recommendation (NPK, pH) | Soil | 2020 | 2,200 | kaggle.com/theeyeschic |
| 3 | Crop and Soil DataSet | Soil + Climate | 2010–2023 (13 yr) | 8,000 | kaggle.com/mahmoudredagamal |
| 4 | ★ Agri Land Suitability BD | Soil + Climate | 2020–2024 (5 yr) | ~9.1M | kaggle.com/devraai |
| 5 | ★ BD Agroclimatic Yield | Climate + Crop | 2000–2024 (24 yr) | 150 | kaggle.com (BD agroclimatic) |
| 6 | Climate Change Surface Temp. | Climate | 1901–2015 (114 yr) | ~39,900 | Berkeley Earth / Kaggle |
| 7 | ★ Climate Data for Bangladesh | Climate | 2021–2024 (4 yr) | 1,450–1,460 | kaggle.com/azrulm |
| 8 | ★ Bangladesh Weather Dataset | Climate | 1901–2023 (122 yr) | ~1,386 | kaggle.com/yakinrubaiai |
| 9 | Environmental Sensor Telemetry | Environmental | ~10 yr (IoT) | 405,184 | kaggle.com/garystafford |
| 10 | Dhaka Air Quality 2000–2025 | Environmental | 2000–2025 (25 yr) | 225,000 | kaggle.com/shalik10945 |

★ = Bangladesh-specific.

### 6.2 Image Datasets

| # | Dataset | Crops | Classes | Images |
|---|---------|-------|---------|--------|
| 1 | PlantVillage | Tomato, Potato, Pepper, Corn, Apple, Grape, etc. | 38 | ~54,000 |
| 2 | Rice Leaf Disease | Rice (Bangladesh) | 4 | ~5,932 |
| 3 | Crop Disease Detection | Rice, Wheat, Maize, Jute | 8 | ~15,000 |
| 4 | Plant Seedlings | 12 species | 12 | ~5,539 |

### 6.3 Local XLS sources

- `parameter report(10).xls` — soil parameter summary report.
- `The Real Time Soil Data .. 1007.xls` — IoT soil-sensor logs (1006 rows, 4 BD locations: Garden / University / Ekuria / Jiya Uddan, Apr–May 2026).

---

## 7. Configuration

All project-wide constants live in `src/config.py`. Modify this file to change paths, feature lists, or hyperparameters.

### 7.1 Paths

```python
RAW_DIR       = Path("data/raw")          # Raw input
PROCESSED_DIR = Path("data/processed")    # Merged output
MODELS_DIR    = Path("models")            # Trained model artefacts
OUTPUTS_DIR   = Path("outputs")           # Figures / CSVs
IMG_DIR       = Path("data/plant_images") # PlantVillage / Rice / etc.
```

### 7.2 ML configuration

```python
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

YIELD_BINS   = [0, 20_000, 50_000, float("inf")]
YIELD_LABELS = ["Low", "Medium", "High"]
```

### 7.3 CNN configuration

```python
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 5     # Phase 1: only train new head
EPOCHS_FINE   = 10    # Phase 2: full fine-tune
```

### 7.4 Crop normalisation

`CROP_MAP` provides a many-to-one mapping that resolves dataset-specific naming inconsistencies (e.g. `"rice, paddy" → "rice"`, `"corn" → "maize"`).
`CROP_SEASON` then assigns each canonical crop to its agronomic season (Kharif vs. Rabi).

---

## 8. Running the Pipeline

### 8.1 Full pipeline (real data, all components)

```bash
python main.py
```

### 8.2 Bangladesh-only mode

```bash
python main.py --bangladesh
```

Filters all data to `country == "Bangladesh"` rows before the merge.

### 8.3 Skip CNN (faster, tabular-only)

```bash
python main.py --skip-cnn
```

Useful for iterating on the tabular models without retraining the CNN (~30 minutes on CPU).

### 8.4 Synthetic-data smoke test

```bash
python main.py --synthetic
```

Generates a 5,000-row synthetic DataFrame using the schema from `config.py`. Completes in under 1 minute.

### 8.5 WSL one-liner (full real training)

```bash
wsl bash -c "cd /mnt/d/4-1 && source venv/bin/activate && python3 main.py"
```

### 8.6 Single-image / single-row inference

```bash
python predict.py --image path/to/leaf.jpg
python predict.py --tabular '{"Year": 2024, "Country": "Bangladesh", ...}'
```

### 8.7 Re-extract image features only

```bash
python extract_image_features.py --input data/test_images/ --output outputs/image_features.csv
```

### 8.8 End-to-end shell runner

```bash
bash run_all.sh
```

This runs: download → merge → train → evaluate → generate report → save outputs.

---

## 9. Pipeline Stages — Step by Step

### Stage 1: Data Ingestion (`src/loaders.py`)

Each dataset has a dedicated loader function:

| Loader | Reads | Returns |
|--------|-------|---------|
| `load_fao_yield()` | `data/raw/yield.csv` | DataFrame (Year, Country, Crop, Yield) |
| `load_fao_temp()` | `data/raw/temp.csv` | DataFrame (Year, Country, avg_temp) |
| `load_fao_rainfall()` | `data/raw/rainfall.csv` | DataFrame (Year, Country, rainfall_mm) |
| `load_crop_recommendation()` | `data/raw/Crop_recommendation.csv` | DataFrame with N, P, K, pH, etc. |
| `load_bd_agroclimatic()` | `data/raw/Bangladesh_*.csv` | BD-specific yield records |
| `load_environmental_sensor()` | `data/raw/iot_telemetry_data.csv` | Aggregated to annual means |
| `load_air_quality()` | `data/raw/dhaka_air_quality_*.csv` | AQI, PM2.5, NO2, etc. |
| `load_xls_soil()` | `parameter report(10).xls` + IoT XLS | Soil sensor data |

**Behaviour:** A loader returns an empty DataFrame (with a warning) if the source file is missing — graceful skip.

### Stage 2: Merging (`src/merger.py`)

The 8-step merge pipeline:

1. **Standardise** — rename columns to canonical snake_case names.
2. **Filter** — restrict to the 2000–2024 window.
3. **Merge climate** — join DS1/DS6/DS7/DS8 on (Year, Country); aggregate by mean.
4. **Merge soil** — join DS2/DS3/DS4 on (Year, Country, Crop_Type); aggregate by mean.
5. **Merge environmental** — join DS9/DS10 on (Year, Country); aggregate annual means.
6. **Final join** — left-join climate + soil + environmental on the composite key.
7. **Handle missing** — median imputation (numerical), mode (categorical); drop rows >30% missing.
8. **Engineer features** — `decade`, `temperature_range`, `rainfall_category`.

The output DataFrame is saved to `data/processed/merged.parquet`.

### Stage 3: Tabular Training (`src/models.py`)

Splits 80/20 → trains four regressors and four classifiers → saves `best_*.pkl` and `all_*.pkl` → writes evaluation CSVs to `outputs/`.

Algorithms:
- **Linear Regression** (`sklearn.linear_model.LinearRegression`)
- **Random Forest** (`sklearn.ensemble.RandomForestRegressor` / `Classifier`)
- **XGBoost** (`xgboost.XGBRegressor` / `XGBClassifier`)
- **KNN** (`sklearn.neighbors.KNeighborsRegressor` / `Classifier`)
- **Naive Bayes** (classification only — `GaussianNB`)

### Stage 4: CNN Training (`src/cnn.py`)

A two-phase fine-tune of a ResNet-50 (or MobileNetV2) backbone.

**Phase 1 — frozen backbone (5 epochs)**
- All ResNet weights are frozen.
- Only the new dense head is trained at lr = 1e-4.

**Phase 2 — full fine-tune (10 epochs)**
- All weights unfrozen.
- Cosine annealing learning rate schedule, lr = 1e-4 → 1e-6.
- Early stopping on validation accuracy (not loss) to avoid mode-collapse.
- Augmentations: horizontal flip, rotation ±15°, color jitter.

The final model is saved as `models/plant_disease_cnn.keras`.

### Stage 5: Fusion (`src/fusion.py`)

Aggregates per-image features (disease class, confidence, RGB means, texture, severity, yellowing/browning indices) by crop, producing the **fusion crop profile** in `outputs/fusion_crop_profile.csv`.

This per-crop summary is concatenated to the tabular feature vector for downstream stakeholder reporting.

### Stage 6: Explainability (`src/explainability.py`)

- **Feature importance** — gain-based for tree models, magnitude-based for linear.
- **SHAP** — TreeSHAP for XGBoost; produces bar / beeswarm / waterfall plots.
- **Grad-CAM** — gradient-weighted class activation map overlaid on the input image.

---

## 10. Module Reference

### `src/loaders.py`

```python
load_all() -> Dict[str, pd.DataFrame]
    """Load every available raw dataset. Returns a dict keyed by dataset name."""

load_fao_yield() -> pd.DataFrame
load_fao_temp() -> pd.DataFrame
load_fao_rainfall() -> pd.DataFrame
load_crop_recommendation() -> pd.DataFrame
load_bd_agroclimatic() -> pd.DataFrame
load_environmental_sensor() -> pd.DataFrame
load_air_quality() -> pd.DataFrame
load_xls_soil() -> pd.DataFrame
```

### `src/merger.py`

```python
merge_all(loaded: Dict[str, pd.DataFrame],
          bangladesh_only: bool = False) -> pd.DataFrame
    """Run the 8-step merge pipeline on the loaded dict."""
```

### `src/models.py`

```python
train_regressors(X_train, y_train, X_test, y_test) -> Dict[str, Any]
    """Train Linear, RF, XGB, KNN regressors. Return dict of fitted models."""

train_classifiers(X_train, y_train, X_test, y_test) -> Dict[str, Any]
    """Train RF, XGB, KNN, NB classifiers."""
```

### `src/cnn.py`

```python
build_cnn(num_classes: int) -> tf.keras.Model
    """Build ResNet-50 with a new softmax head."""

train_cnn(model, train_ds, val_ds) -> tf.keras.callbacks.History
    """Two-phase training schedule: frozen → fine-tune."""

predict_image(model, img_path: str) -> dict
    """Return {disease_class, confidence, severity, RGB stats, etc.}."""
```

### `src/fusion.py`

```python
build_crop_profile(image_features: pd.DataFrame) -> pd.DataFrame
    """Aggregate per-image features by crop. Saved to fusion_crop_profile.csv."""
```

### `src/explainability.py`

```python
plot_feature_importance(model, feature_names, task: str) -> None
plot_shap_summary(model, X, plot_type: str)        # 'bar' / 'beeswarm'
plot_shap_waterfall(model, X, sample_idx: int)
plot_gradcam(model, img_path: str, class_idx: int)
```

### `src/utils.py`

```python
banner(title: str)                # Print formatted section header
section(title: str)               # Print sub-section header
ensure_dirs(*paths)               # mkdir -p
save_fig(fig, name: str)          # Save plt figure to outputs/
```

---

## 11. Trained Models

All trained artefacts are saved in `models/`:

| File | Size | Contents |
|------|------|----------|
| `best_regressor.pkl` | 863 KB | XGBoost regressor (best of 4) |
| `best_classifier.pkl` | 2.1 MB | XGBoost classifier (best of 4) |
| `all_regressors.pkl` | 167.5 MB | All 4 regressors (LR, RF, XGB, KNN) |
| `all_classifiers.pkl` | 128.4 MB | All 4 classifiers (RF, XGB, KNN, NB) |
| `plant_disease_cnn.keras` | 25.1 MB | Fine-tuned CNN (38-class PlantVillage) |
| `preprocessor.pkl` | 2.1 KB | `ColumnTransformer` (one-hot + scaler) |
| `feature_names.pkl` | 298 B | Saved feature names list |

### Loading a saved model

```python
import joblib

reg = joblib.load("models/best_regressor.pkl")
cls = joblib.load("models/best_classifier.pkl")

# CNN
import tensorflow as tf
cnn = tf.keras.models.load_model("models/plant_disease_cnn.keras")
```

---

## 12. Output Files

The `outputs/` directory contains all generated figures and result tables.

### 12.1 EDA figures

| File | Description |
|------|-------------|
| `eda_yield_distribution.png` | Histogram of yield (right-skewed) |
| `eda_yield_trend.png` | Mean yield over years (1961–2016) |
| `eda_yield_categories.png` | Class proportions after Low/Med/High binning |
| `eda_correlation_heatmap.png` | Pearson correlation matrix |
| `eda_scatter_plots.png` | Pairwise feature × yield scatter |

### 12.2 Regression results

| File | Description |
|------|-------------|
| `regression_results.csv` | RMSE / MAE / R² / CV-R² for all 4 models |
| `regression_comparison.png` | Bar chart comparing the 4 regressors |
| `actual_vs_pred_Linear_Regression.png` | Diagonal scatter plot |
| `actual_vs_pred_Random_Forest.png` | Diagonal scatter plot |
| `actual_vs_pred_XGBoost.png` | Diagonal scatter plot |
| `feature_importance_Regression-Random_Forest.png` | RF feature ranking |
| `feature_importance_Regression-XGBoost.png` | XGBoost feature ranking |

### 12.3 Classification results

| File | Description |
|------|-------------|
| `classification_results.csv` | Accuracy / F1 for all 4 classifiers |
| `classification_comparison.png` | Bar chart of classifier metrics |
| `confusion_matrix_XGBoost.png` | Best classifier confusion matrix |
| `confusion_matrix.png` | CNN 38-class confusion matrix |
| `feature_importance_Classification-XGBoost.png` | XGBoost classifier feature ranking |

### 12.4 CNN results

| File | Description |
|------|-------------|
| `classification_report.txt` | Per-class precision / recall / F1 (38 classes) |
| `history_phase_1_frozen.png` | Phase 1 training curves |
| `history_phase_2_fine-tune.png` | Phase 2 training curves |
| `gradcam_sample.jpg` | Grad-CAM overlay on a Potato Early Blight leaf |

### 12.5 SHAP results

| File | Description |
|------|-------------|
| `shap_regression_bar.png` | Global SHAP importance (regression) |
| `shap_classification_bar.png` | Global SHAP importance (classification) |
| `shap_regression_beeswarm.png` | Per-sample SHAP distribution |
| `shap_regression_waterfall.png` | Single-prediction additive explanation |

### 12.6 Fusion outputs

| File | Description |
|------|-------------|
| `image_features_sample.csv` | Single-image feature row (preview) |
| `image_features_multi.csv` | Batch-extracted image features |
| `disease_result.csv` | Disease prediction for one test image |
| `test_disease_results.csv` | Disease predictions for the test set |
| `fusion_crop_profile.csv` | Per-crop aggregated image profile |

---

## 13. Empirical Results

### 13.1 Regression — held-out test set

| Model | RMSE | MAE | R² | CV R² |
|-------|------|-----|----|----|
| Linear Regression | 72,718.80 | 52,931.93 | 0.1522 | 0.1569 |
| Random Forest | 81,172.66 | 53,700.77 | −0.0564 | −0.0333 |
| **XGBoost** ✅ | **67,602.21** | **46,719.14** | **0.2673** | **0.2608** |
| KNN | 73,553.93 | 51,466.65 | 0.1326 | 0.1268 |

XGBoost is the best regressor. RF surprisingly under-performs without aggressive hyper-parameter tuning — indicative of noisy categorical encodings (Country, Crop_Type).

### 13.2 Classification — Low / Medium / High yield

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| Random Forest | 0.5292 | 0.5260 |
| **XGBoost** ✅ | **0.6298** | **0.6199** |
| KNN | 0.5455 | 0.5296 |
| Naive Bayes | 0.4702 | 0.4393 |

### 13.3 CNN — PlantVillage 38-class

| Metric | Value |
|--------|-------|
| Test accuracy | **0.98** |
| Macro precision | 0.98 |
| Macro recall | 0.98 |
| Macro F1 | 0.98 |
| Test images | 17,572 |

Per-class scores are in `outputs/classification_report.txt`. Tomato classes (Target_Spot, healthy) show the most confusion due to within-species visual similarity.

### 13.4 SHAP global importance (regression)

Top predictive features for yield:

1. **Pesticide application**
2. **Country**
3. **Crop_Type**
4. **Average rainfall**
5. **Average temperature**

This ranking matches both the gain-based and SHAP-based importance plots.

---

## 14. Reproducing the Report

Three report formats are auto-generated from the trained outputs:

### 14.1 PDF / DOCX (Python-generated)

```bash
python generate_report.py        # PDF + DOCX in report/
python make_report_doc.py        # Alternative DOCX generator
```

### 14.2 LaTeX (Overleaf-ready)

The `report.tex` file is fully self-contained.
Upload to Overleaf with `icon.png` and the entire `outputs/` folder, then compile with `pdflatex`.

```
report.tex          ← LaTeX source
icon.png            ← UAP logo
outputs/*.png       ← Figures (auto-located via \graphicspath)
```

### 14.3 Generated artefacts

After running the full pipeline + report generation, you will have:

```
report/
├── Crop_Yield_PlantDisease_Report_MahinurAkhter.pdf  (1.5 MB)
└── report.docx                                        (62.9 KB)

report_latex/
└── report.pdf                                         (compiled from report.tex)
```

---

## 15. Troubleshooting

### Issue — `ModuleNotFoundError: No module named 'src'`
**Fix:** Run from the project root, or set `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:./src
```

### Issue — `kaggle.json` permission error
**Fix:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Issue — TensorFlow does not detect GPU
**Fix:** Verify CUDA / cuDNN compatibility with your TF version:
```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

### Issue — Out-of-memory during CNN training
**Fix:** Reduce `BATCH_SIZE` in `src/config.py` from 32 → 16 → 8.

### Issue — Pandas FutureWarning about deprecated dtypes
**Fix:** Harmless. Suppress with:
```python
import warnings; warnings.filterwarnings("ignore")
```
(Already applied in `main.py`.)

### Issue — XGBoost “label encoder is deprecated”
**Fix:** Pass `enable_categorical=True` or pre-encode categoricals (already handled via `ColumnTransformer`).

### Issue — LaTeX compile timeout on Overleaf free
**Fix:** Already mitigated in `report.tex` by using simple TikZ figures and including pre-rendered PNGs from `outputs/` instead of computing in-document.

---

## 16. FAQ

**Q. Why XGBoost instead of deep tabular models (TabNet, FT-Transformer)?**
A. The merged DataFrame is small (after filtering, ~10–50k rows depending on Bangladesh filter). Tree-based ensembles dominate in this regime; deep tabular models would over-fit without strong regularisation and orders-of-magnitude more data.

**Q. Why is RF regression negative?**
A. Without hyper-parameter tuning, RF over-fits the noisy categorical features (especially Country, Crop_Type) and generalises worse than even Linear Regression. Aggressive Bayesian optimisation typically recovers RF to R² ≈ 0.20–0.25.

**Q. Why two-phase CNN training?**
A. Phase 1 lets the new head converge without disrupting pretrained ResNet features. Phase 2 then fine-tunes the entire network — but starting from a sane head minimises catastrophic forgetting.

**Q. Why early-stop on val accuracy, not val loss?**
A. Cross-entropy can converge to ln 2 ≈ 0.693 if the model collapses to uniform predictions — a state with only 50% accuracy. A loss-based selector would pick this collapsed checkpoint. Tracking val accuracy avoids this pitfall (lesson learned from related project [11] in the references).

**Q. Can I add a new dataset?**
A. Yes:
1. Add a `load_<name>()` function in `src/loaders.py`.
2. Update `load_all()` to call it.
3. Update `src/merger.py` to merge the new columns into the appropriate domain.
4. Add the new column names to `FEATURE_COLS` in `src/config.py`.

**Q. How do I retrain only one model?**
A. Edit `train_regressors()` / `train_classifiers()` in `src/models.py` to toggle individual algorithms. Or call them directly:

```python
from src.models import train_regressors
results = train_regressors(X_train, y_train, X_test, y_test, models=['xgboost'])
```

**Q. Where is the actual SQL/data-flow code?**
A. The 8-step merge is implemented in `src/merger.py:merge_all()`. Each step is a clearly delimited section in that function.

**Q. Can I deploy this as a web service?**
A. Yes — wrap `predict.py` in a Flask / FastAPI endpoint. Load the saved `.pkl` and `.keras` models on startup; expose `/predict_yield` and `/predict_disease` routes.

---

## 17. License & Acknowledgements

### License

This project is academic-coursework software, released for educational use only. Third-party datasets retain their original licenses (FAO, Kaggle, PlantVillage); please consult each source for redistribution terms.

### Acknowledgements

- **University of Asia Pacific**, Department of Computer Science and Engineering — for academic support.
- **Faria Zarin Subah**, Assistant Professor — for project supervision.
- **Open-source community** — FAO, Kaggle contributors, PlantVillage, scikit-learn, TensorFlow, XGBoost, SHAP.

### Authors

| Name | ID | Role |
|------|----|----|
| Mahinur Akhter | 22201100 | Pipeline architecture, CNN, fusion |
| Halima Akter Shorna | 22201101 | Data merging, EDA, classification |
| Mehedi Hasan | 22201109 | Regression models, SHAP, reporting |

---

*Document version 1.0 — last updated May 2026.*

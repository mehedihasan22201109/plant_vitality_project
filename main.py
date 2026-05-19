"""
Crop Yield Prediction & Plant Disease Detection — Full Pipeline


Usage:
    python main.py                  # full pipeline, real data
    python main.py --bangladesh     # Bangladesh rows only
    python main.py --skip-cnn       # skip CNN training
    python main.py --synthetic      # use synthetic tabular data
"""

import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _section(title: str):
    print(f"\n── {title} ──")


# ── parts ─────────────────────────────────────────────────────────────────────

def part1_load_and_merge(bangladesh_only: bool, use_synthetic: bool) -> pd.DataFrame:
    _banner("PART 1 — DATA MERGING (7-Step Strategy)")

    if use_synthetic:
        _section("Using synthetic data")
        from config import (CROP_MAP, CROP_SEASON, PROCESSED_DIR,
                             YIELD_BINS, YIELD_LABELS)
        rng = np.random.default_rng(42)
        crops  = list(set(CROP_MAP.values()))[:8]
        n      = 5000
        years  = rng.integers(2000, 2021, n)
        cr     = rng.choice(crops, n)
        season_map = {c: CROP_SEASON.get(c, "Kharif") for c in crops}
        df = pd.DataFrame({
            "Year": years, "Country": rng.choice(["Bangladesh","India","Pakistan"], n),
            "Crop_Type": cr,
            "yield_hgha":      rng.uniform(5000, 80000, n).round(0),
            "avg_temp":        rng.uniform(20, 35, n).round(2),
            "min_temp":        rng.uniform(10, 22, n).round(2),
            "max_temp":        rng.uniform(30, 42, n).round(2),
            "rainfall_mm":     rng.uniform(300, 2500, n).round(1),
            "humidity_pct":    rng.uniform(50, 90, n).round(1),
            "wind_speed_kmh":  rng.uniform(5, 25, n).round(1),
            "sunshine_hours":  rng.uniform(4, 10, n).round(1),
            "season":          [season_map.get(c, "Kharif") for c in cr],
            "nitrogen_N":      rng.uniform(10, 140, n).round(1),
            "phosphorous_P":   rng.uniform(5, 145, n).round(1),
            "potassium_K":     rng.uniform(5, 205, n).round(1),
            "soil_pH":         rng.uniform(5.0, 8.5, n).round(2),
            "soil_moisture_pct": rng.uniform(20, 70, n).round(1),
            "soil_type":       rng.choice(["Loam","Clay","Sandy","Alluvial"], n),
            "fertilizer_kgha": rng.uniform(80, 400, n).round(1),
            "AQI":             rng.uniform(50, 300, n).round(1),
            "CO2_ppm":         rng.uniform(390, 420, n).round(1),
            "PM25_ugm3":       rng.uniform(10, 150, n).round(1),
            "NO2_ppb":         rng.uniform(5, 60, n).round(1),
            "decade":          (years // 10) * 10,
            "temperature_range": rng.uniform(8, 20, n).round(2),
        })
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / "merged_dataset.csv", index=False)
        print(f"  Synthetic data: {df.shape}")
        return df

    from loaders import load_all, print_status
    from merger  import run_pipeline

    print("\nLoading datasets...")
    datasets = load_all()
    print_status(datasets)

    print("\nRunning merge pipeline...")
    return run_pipeline(datasets, bangladesh_only=bangladesh_only)


def part2_eda(df: pd.DataFrame) -> pd.DataFrame:
    _banner("PART 2 — EXPLORATORY DATA ANALYSIS")
    print(f"\n  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    print(f"\n  Sample:\n{df.head(3).to_string()}")
    print(f"\n  Stats:\n{df.describe().round(2).to_string()}")

    try:
        from utils import plot_eda
        plot_eda(df)
    except Exception as e:
        print(f"  [WARN] EDA plots failed: {e}")

    return df


def part3_regression(X_tr, X_te, yr_tr, yr_te, feature_names: list):
    _banner("PART 3 — REGRESSION (Predict yield_hgha)")

    from models import run_regression
    from utils  import plot_regression_results, plot_actual_vs_predicted, plot_feature_importance

    results_df, best_model = run_regression(X_tr, X_te, yr_tr, yr_te, feature_names)

    try:
        plot_regression_results(results_df)
        best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
        from sklearn.metrics import r2_score
        from models import prepare_features as _pf
        # predictions already done inside run_regression, re-predict for plots
        y_pred = best_model.predict(X_te)
        plot_actual_vs_predicted(yr_te, y_pred, best_name)
        plot_feature_importance(best_model, feature_names, f"Regression-{best_name}")
    except Exception as e:
        print(f"  [WARN] Regression plots failed: {e}")

    best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
    print(f"\n  Best Regressor : {best_name}  R2={results_df['R2'].max():.4f}")
    return results_df, best_model, best_name


def part4_classification(X_tr, X_te, yc_tr, yc_te, feature_names: list):
    _banner("PART 4 — CLASSIFICATION (High / Medium / Low Yield)")

    from models import run_classification
    from utils  import (plot_classification_results, plot_confusion_matrix,
                         plot_feature_importance)
    from config import YIELD_LABELS

    results_df, best_model = run_classification(X_tr, X_te, yc_tr, yc_te, feature_names)

    try:
        plot_classification_results(results_df)
        best_name = results_df.loc[results_df["F1_weighted"].idxmax(), "Model"]
        y_pred = best_model.predict(X_te)
        # XGBoost returns encoded ints — decode if needed
        if hasattr(y_pred[0], 'item') and isinstance(y_pred[0].item(), int):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder(); le.fit(YIELD_LABELS)
            y_pred = le.inverse_transform(y_pred)
        plot_confusion_matrix(yc_te, y_pred, best_name, YIELD_LABELS)
        plot_feature_importance(best_model, feature_names, f"Classification-{best_name}")
    except Exception as e:
        print(f"  [WARN] Classification plots failed: {e}")

    best_name = results_df.loc[results_df["F1_weighted"].idxmax(), "Model"]
    print(f"\n  Best Classifier : {best_name}  F1={results_df['F1_weighted'].max():.4f}")
    return results_df, best_model, best_name


def part5_cnn():
    _banner("PART 5 — CNN: Plant Disease Detection")
    print("""
  Architecture : MobileNetV2 (ImageNet) + Custom Head
  Phase 1      : Feature extraction (frozen base, lr=1e-3)
  Phase 2      : Fine-tuning (last 30 layers, lr=1e-4)
  XAI          : Grad-CAM + Disease Severity %
""")
    try:
        from cnn import run_cnn_pipeline
        return run_cnn_pipeline()
    except ImportError:
        print("  [SKIP] TensorFlow not installed.")
        return {}
    except Exception as e:
        print(f"  [ERR] CNN failed: {e}")
        return {}


def print_summary(df, reg_df, cls_df, best_reg, best_cls, cnn_info: dict):
    _banner("FINAL SUMMARY")

    sep = "─" * 42

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n  Dataset\n  {sep}")
    print(f"  Rows         : {len(df):,}")
    print(f"  Features     : {df.shape[1]}")
    print(f"  Target       : yield_hgha  (crop yield in hg/ha)")
    print(f"  Countries    : {df['Country'].nunique() if 'Country' in df.columns else 'N/A'}")
    print(f"  Crop types   : {df['Crop_Type'].nunique() if 'Crop_Type' in df.columns else 'N/A'}")
    print(f"  Year range   : {int(df['Year'].min())} – {int(df['Year'].max())}")

    # ── Regression ───────────────────────────────────────────────────────────
    print(f"\n  Regression  (predict exact yield_hgha)\n  {sep}")
    print(reg_df.to_string(index=False))
    print(f"\n  ★ Best Regressor  : {best_reg}  "
          f"R2={reg_df['R2'].max():.4f}  "
          f"RMSE={reg_df.loc[reg_df['R2'].idxmax(),'RMSE']:,.1f}")

    # ── Classification ────────────────────────────────────────────────────────
    print(f"\n  Classification  (High / Medium / Low yield)\n  {sep}")
    print(cls_df.to_string(index=False))
    print(f"\n  ★ Best Classifier : {best_cls}  "
          f"F1={cls_df['F1_weighted'].max():.4f}  "
          f"Acc={cls_df.loc[cls_df['F1_weighted'].idxmax(),'Accuracy']:.4f}")

    # ── CNN ───────────────────────────────────────────────────────────────────
    if cnn_info:
        print(f"\n  Plant Disease Detection (CNN + XAI)\n  {sep}")
        print(f"  Architecture  : MobileNetV2 (ImageNet) + Custom Head")
        print(f"  Classes       : {cnn_info.get('num_classes', '?')}")
        print(f"  Val Accuracy  : {cnn_info.get('val_acc', 0):.4f}")
        print(f"  Val Loss      : {cnn_info.get('val_loss', 0):.4f}")
        print(f"  XAI Method    : Grad-CAM heatmap + Disease Severity %")

        feats = cnn_info.get("image_features", {})
        if feats:
            print(f"\n  Sample Image — Extracted Features (XAI)\n  {sep}")
            rows = [
                ("Disease Class",    feats.get("Disease_Class", "—")),
                ("Confidence %",     feats.get("Confidence_Pct", "—")),
                ("R mean",           feats.get("R_mean", "—")),
                ("G mean",           feats.get("G_mean", "—")),
                ("B mean",           feats.get("B_mean", "—")),
                ("Texture Score",    feats.get("Texture_Score", "—")),
                ("Yellowing Index",  feats.get("Yellowing_Index", "—")),
                ("Browning Index",   feats.get("Browning_Index", "—")),
                ("Disease Severity", f"{feats.get('Disease_Severity_Pct', 0):.1f}%"),
            ]
            for label, val in rows:
                print(f"  {label:<22}: {val}")

    # ── Output files ──────────────────────────────────────────────────────────
    print(f"\n  Output files  → outputs/")
    print(f"  Saved models  → models/")
    print(f"    best_regressor.pkl")
    print(f"    best_classifier.pkl")
    if cnn_info:
        print(f"    plant_disease_cnn.keras")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic",  action="store_true", help="Use synthetic tabular data")
    parser.add_argument("--bangladesh", action="store_true", help="Bangladesh rows only")
    parser.add_argument("--skip-cnn",   action="store_true", help="Skip CNN training")
    args = parser.parse_args()

    _banner("CROP YIELD PREDICTION & PLANT DISEASE DETECTION")
    print("  Mahinur Akhter  |  ID: 22201100")

    # Part 1 — Data
    df = part1_load_and_merge(bangladesh_only=args.bangladesh,
                               use_synthetic=args.synthetic)

    # Part 2 — EDA
    part2_eda(df)

    # Prepare ML features
    _banner("PREPARING ML FEATURES")
    from models import prepare_features
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, feature_names = prepare_features(df)
    print(f"\n  Train : {len(X_tr):,} samples")
    print(f"  Test  : {len(X_te):,} samples")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    import joblib
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    # save preprocessor info
    joblib.dump(feature_names, "models/feature_names.pkl")

    # Part 3 — Regression
    reg_df, _, best_reg = part3_regression(X_tr, X_te, yr_tr, yr_te, feature_names)

    # Part 4 — Classification
    cls_df, _, best_cls = part4_classification(X_tr, X_te, yc_tr, yc_te, feature_names)

    # Part 5 — CNN
    cnn_info = {}
    if not args.skip_cnn:
        cnn_info = part5_cnn()
    else:
        print("\n  [INFO] CNN skipped.")

    # Part 6 — Image-Tabular Fusion
    _banner("PART 6 — IMAGE-TABULAR FUSION")
    try:
        from fusion import print_fusion_summary, fuse
        print_fusion_summary()
        df_fused = fuse(df)
        if df_fused.shape[1] > df.shape[1]:
            print(f"\n  Fused dataset: {df_fused.shape[1]} features "
                  f"(+{df_fused.shape[1]-df.shape[1]} image features)")
            df_fused.to_csv("data/processed/merged_fused.csv", index=False)
            print("  Saved → data/processed/merged_fused.csv")
        else:
            print("  No image features to fuse yet — run CNN first.")
    except Exception as e:
        print(f"  [INFO] Fusion skipped: {e}")

    # Summary
    print_summary(df, reg_df, cls_df, best_reg, best_cls, cnn_info)


if __name__ == "__main__":
    main()

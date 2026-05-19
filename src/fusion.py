"""
fusion.py — Image-Tabular Fusion Module
Links CNN image features to the tabular crop yield dataset.

How it works:
  1. Load saved image features (from outputs/image_features_multi.csv)
  2. Group by Crop_Type → compute mean CNN stats per crop
  3. Merge into the tabular DataFrame as additional features
  4. Optionally retrain the yield model with these extra features

"""

from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

OUTPUTS = Path("outputs")
MODELS  = Path("models")

IMAGE_FEAT_FILE = OUTPUTS / "image_features_multi.csv"


# ── Build crop-level image feature lookup ─────────────────────────────────────

def build_crop_image_profile(img_feat_path: Path = IMAGE_FEAT_FILE) -> pd.DataFrame:
    """
    Aggregates image features by Crop_Type.

    Returns a DataFrame with one row per crop type and columns:
        img_avg_confidence, img_avg_severity, img_avg_yellowing,
        img_avg_browning,   img_avg_texture,  img_disease_rate
    """
    if not img_feat_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(img_feat_path)

    if "Crop_Type" not in df.columns:
        # Try to extract crop type from Disease_Class column
        if "Disease_Class" in df.columns:
            df["Crop_Type"] = df["Disease_Class"].apply(
                lambda x: x.split("___")[0].replace("_", " ").strip()
                          if "___" in str(x) else str(x)
            )
        else:
            return pd.DataFrame()

    df["is_diseased"] = (~df["Disease_Class"].str.contains("healthy",
                          case=False, na=False)).astype(int)

    agg = df.groupby("Crop_Type").agg(
        img_avg_confidence = ("Confidence_Pct",       "mean"),
        img_avg_severity   = ("Disease_Severity_Pct", "mean"),
        img_avg_yellowing  = ("Yellowing_Index",       "mean"),
        img_avg_browning   = ("Browning_Index",        "mean"),
        img_avg_texture    = ("Texture_Score",         "mean"),
        img_disease_rate   = ("is_diseased",           "mean"),
        img_sample_count   = ("Disease_Class",         "count"),
    ).reset_index()

    # Round for readability
    num_cols = [c for c in agg.columns if c != "Crop_Type"]
    agg[num_cols] = agg[num_cols].round(4)

    return agg


def fuse(tabular_df: pd.DataFrame,
         img_feat_path: Path = IMAGE_FEAT_FILE) -> pd.DataFrame:
    """
    Merges CNN image features into the tabular DataFrame.

    Joins on Crop_Type (normalised to lower-case, stripped).
    Rows with no matching image profile get 0-filled image features.

    Returns the augmented DataFrame.
    """
    profile = build_crop_image_profile(img_feat_path)
    if profile.empty:
        print("  [Fusion] No image features available — skipping fusion.")
        return tabular_df

    df = tabular_df.copy()

    # PlantVillage → FAO crop name bridge
    PV_TO_FAO = {
        "corn maize": "maize", "corn": "maize",
        "potato":     "potato",
        "tomato":     "tomato",
        "apple":      "apple",
        "grape":      "grape",
        "peach":      "peach",
        "cherry including sour": "cherry",
        "pepper bell": "pepper",
        "soybean":    "soybean",
        "raspberry":  "raspberry",
        "strawberry": "strawberry",
        "blueberry":  "blueberry",
        "squash":     "squash",
        "orange":     "orange",
    }

    def _norm(s):
        import re
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower().strip())

    def _bridge(s):
        n = _norm(s)
        return PV_TO_FAO.get(n, n)

    df["_crop_norm"]      = df["Crop_Type"].apply(_bridge)
    profile["_crop_norm"] = profile["Crop_Type"].apply(_bridge)

    img_cols = [c for c in profile.columns if c.startswith("img_")]
    profile_slim = profile[["_crop_norm"] + img_cols]

    df = df.merge(profile_slim, on="_crop_norm", how="left")
    df = df.drop(columns=["_crop_norm"])

    # Fill unmatched rows with 0
    for col in img_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    matched = df[img_cols[0]].astype(bool).sum() if img_cols else 0
    print(f"  [Fusion] Merged {len(img_cols)} image features. "
          f"Matched rows: {matched:,} / {len(df):,}")

    return df


def print_fusion_summary(img_feat_path: Path = IMAGE_FEAT_FILE):
    """Print a summary table of image features per crop type."""
    profile = build_crop_image_profile(img_feat_path)
    if profile.empty:
        print("  [Fusion] No image feature profile available.")
        return

    print("\n  Image-Tabular Fusion — Crop Disease Profile")
    print("  " + "─" * 75)
    print(f"  {'Crop Type':<20} {'Conf%':>7} {'Severity%':>10} "
          f"{'Yellow':>8} {'Brown':>8} {'Disease Rate':>13}")
    print("  " + "─" * 75)
    for _, row in profile.iterrows():
        print(f"  {str(row['Crop_Type']):<20} "
              f"{row['img_avg_confidence']:>7.1f} "
              f"{row['img_avg_severity']:>10.1f} "
              f"{row['img_avg_yellowing']:>8.4f} "
              f"{row['img_avg_browning']:>8.4f} "
              f"{row['img_disease_rate']*100:>12.1f}%")
    print("  " + "─" * 75)

    # Save
    out = OUTPUTS / "fusion_crop_profile.csv"
    profile.to_csv(out, index=False)
    print(f"  Saved → {out}")

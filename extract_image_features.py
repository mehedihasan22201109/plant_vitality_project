"""
extract_image_features.py
Loads the saved CNN model and extracts 1 image per class → image_features_multi.csv
Run: python3 extract_image_features.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from cnn import _find_data_dir, _gradcam, extract_image_features

OUTPUTS   = Path("outputs")
MODEL_PATH = Path("models/plant_disease_cnn.keras")
IMG_SIZE   = (224, 224)

def main():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("Loading CNN model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    data_dir = _find_data_dir()
    val_dir  = data_dir / "valid" if (data_dir / "valid").exists() else data_dir / "val"
    class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    print(f"Classes: {len(class_names)}")

    # Last conv layer name
    last_conv = next(
        (l.name for l in reversed(model.layers) if "conv" in l.name.lower()),
        "out_relu"
    )

    # Build val generator (no shuffle)
    val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(val_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode="categorical", shuffle=False
    )

    seen, results = set(), []
    print("Extracting 1 image per class...")
    for x_batch, y_batch in val_gen:
        for j in range(len(x_batch)):
            cls_idx = int(np.argmax(y_batch[j]))
            if cls_idx in seen:
                continue
            seen.add(cls_idx)
            img_j = x_batch[j:j+1]
            try:
                hmap = _gradcam(img_j, model, last_conv)
                hmap = cv2.resize(hmap, IMG_SIZE)
            except Exception:
                hmap = None
            f = extract_image_features(model, img_j, class_names, heatmap=hmap)
            f["Sample_Index"] = cls_idx
            results.append(f)
            print(f"  [{len(seen):>2}/{len(class_names)}] {f['Disease_Class']:<50} "
                  f"severity={f['Disease_Severity_Pct']:.1f}%  "
                  f"stage={f['Growth_Stage']}")
        if len(seen) >= len(class_names):
            break

    df = pd.DataFrame(results)
    out = OUTPUTS / "image_features_multi.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    print(df[["Crop_Type","Disease_Class","Confidence_Pct",
              "Yellowing_Index","Disease_Severity_Pct","Growth_Stage"]].to_string(index=False))

if __name__ == "__main__":
    main()



import matplotlib
matplotlib.use("Agg")

import os
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 5
EPOCHS_FINE = 10
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# 1. Auto-detect data directory
# ---------------------------------------------------------------------------
def _find_data_dir() -> Path:
    """
    Auto-detects the PlantVillage train/valid root at any depth under
    data/plant_images/ using rglob. Returns the parent of the first
    train/ directory found (sorted by depth, shallowest first).
    """
    base = Path("data/plant_images")
    if not base.exists():
        return base
    # Find all train/ dirs, pick the shallowest one
    train_dirs = sorted(
        [p for p in base.rglob("train") if p.is_dir()],
        key=lambda p: len(p.parts),
    )
    if train_dirs:
        return train_dirs[0].parent
    return base


# ---------------------------------------------------------------------------
# 2. Clean corrupted images
# ---------------------------------------------------------------------------
def _clean_corrupted(data_dir: Path) -> int:
    """
    Scans all images in data_dir once. On subsequent runs, skips the scan
    using a .cleaned marker file — avoids re-scanning 70k+ images every run.
    """
    marker = data_dir / ".cleaned"
    if marker.exists():
        print("  Corruption scan already done — skipping.")
        return 0

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        print("  [WARN] PIL not available — skipping corruption scan.")
        return 0

    removed = 0
    extensions = {".jpg", ".jpeg", ".png"}
    all_files = [p for p in data_dir.rglob("*") if p.suffix.lower() in extensions]
    total = len(all_files)
    print(f"  Scanning {total:,} images for corruption...")

    for fpath in all_files:
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception:
            fpath.unlink(missing_ok=True)
            removed += 1

    marker.touch()
    print(f"  Done — removed {removed} corrupted files. Marker saved.")
    return removed


# ---------------------------------------------------------------------------
# 3. Build data generators
# ---------------------------------------------------------------------------
def _build_generators(data_dir: Path) -> tuple:
    """
    Creates ImageDataGenerator instances for train and validation sets.
    Returns: (train_gen, val_gen, num_classes, class_names)
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = data_dir / "train"

    # Prefer valid/ but fall back to val/
    val_dir = data_dir / "valid"
    if not val_dir.exists():
        val_dir = data_dir / "val"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Could not find valid/ or val/ directory in {data_dir}"
        )

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    return train_gen, val_gen, num_classes, class_names


# ---------------------------------------------------------------------------
# 4. Build MobileNetV2 model
# ---------------------------------------------------------------------------
def _build_model(num_classes: int, freeze_base: bool) -> tuple:
    """
    Builds a MobileNetV2-based transfer learning model.
    Returns: (model, base_model)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import (
            GlobalAveragePooling2D, BatchNormalization,
            Dense, Dropout, Input,
        )
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        from keras.applications import MobileNetV2
        from keras import Model
        from keras.layers import (
            GlobalAveragePooling2D, BatchNormalization,
            Dense, Dropout, Input,
        )
        from keras.optimizers import Adam

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


# ---------------------------------------------------------------------------
# 5. Grad-CAM
# ---------------------------------------------------------------------------
def _gradcam(img_array: np.ndarray, model, last_conv_name: str) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for the first image in the batch.
    Returns normalized heatmap array (values 0-1).
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("TensorFlow is required for Grad-CAM.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    # Use only the first image
    img = img_array[0:1]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Normalize to 0-1
    heatmap = np.maximum(heatmap, 0)
    denom = heatmap.max()
    if denom > 0:
        heatmap /= denom

    return heatmap


# ---------------------------------------------------------------------------
# 6. Severity estimation
# ---------------------------------------------------------------------------
def _severity(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """Returns percentage of pixels in heatmap above the given threshold."""
    total = heatmap.size
    if total == 0:
        return 0.0
    return float(np.sum(heatmap > threshold) / total * 100)


# ---------------------------------------------------------------------------
# 6b. Image feature extraction (XAI output table)
# ---------------------------------------------------------------------------
def extract_image_features(model, img_array: np.ndarray,
                            class_names: list, heatmap: np.ndarray = None) -> dict:
    """
    Extracts interpretable features from a single image after CNN inference.

    Parameters
    ----------
    model      : trained Keras model
    img_array  : shape (1, H, W, 3), values 0-1
    class_names: list of class label strings
    heatmap    : optional Grad-CAM heatmap (H, W) for severity score

    Returns
    -------
    dict with keys:
        Crop_Type, Disease_Class, Confidence_Pct,
        R_mean, G_mean, B_mean, Texture_Score,
        Yellowing_Index, Browning_Index,
        Disease_Severity_Pct, Growth_Stage
    """
    preds = model.predict(img_array, verbose=0)
    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx]) * 100.0

    rgb = img_array[0]
    r_mean = float(rgb[:, :, 0].mean())
    g_mean = float(rgb[:, :, 1].mean())
    b_mean = float(rgb[:, :, 2].mean())

    texture_score   = float(rgb.std())
    yellowing_index = round(r_mean - g_mean, 4)
    browning_index  = round(r_mean - b_mean, 4)
    severity        = _severity(heatmap) if heatmap is not None else 0.0

    disease_class = (class_names[pred_idx]
                     if pred_idx < len(class_names) else str(pred_idx))

    # ── Crop Type from class name (e.g. "Tomato___Early_blight" → "Tomato") ──
    if "___" in disease_class:
        crop_type = disease_class.split("___")[0].replace("_", " ").strip()
    else:
        crop_type = disease_class.replace("_", " ").strip()

    # ── Growth Stage heuristic ────────────────────────────────────────────────
    # Estimated from leaf colour and disease severity as a proxy.
    # (Requires Plant Seedlings dataset for ML-based classification.)
    if "healthy" in disease_class.lower():
        if texture_score < 0.12:
            growth_stage = "Seedling"
        elif texture_score < 0.18:
            growth_stage = "Mature"
        else:
            growth_stage = "Harvest-Ready"
    else:
        if severity < 10.0:
            growth_stage = "Early Infection"
        elif severity < 30.0:
            growth_stage = "Moderate Infection"
        else:
            growth_stage = "Severe Infection"

    return {
        "Crop_Type":            crop_type,
        "Disease_Class":        disease_class,
        "Confidence_Pct":       round(confidence, 2),
        "R_mean":               round(r_mean, 4),
        "G_mean":               round(g_mean, 4),
        "B_mean":               round(b_mean, 4),
        "Texture_Score":        round(texture_score, 4),
        "Yellowing_Index":      yellowing_index,
        "Browning_Index":       browning_index,
        "Disease_Severity_Pct": round(severity, 2),
        "Growth_Stage":         growth_stage,
    }


def _print_feature_table(features: dict) -> None:
    """Pretty-prints the extracted image feature dict as a table."""
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │        Extracted Image Features (XAI)       │")
    print("  ├──────────────────────────┬──────────────────┤")
    for k, v in features.items():
        label = k.replace("_", " ")
        print(f"  │ {label:<26}│ {str(v):>16} │")
    print("  └──────────────────────────┴──────────────────┘")


# ---------------------------------------------------------------------------
# Helper: plot training history
# ---------------------------------------------------------------------------
def _plot_history(history, phase: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label="Train Accuracy")
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy")
    axes[0].set_title(f"{phase} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label="Train Loss")
    axes[1].plot(epochs_range, val_loss, label="Val Loss")
    axes[1].set_title(f"{phase} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    fname = output_dir / f"history_{phase.lower().replace(' ', '_')}.png"
    plt.savefig(fname)
    plt.close(fig)
    print(f"  History plot saved: {fname}")


# ---------------------------------------------------------------------------
# Helper: confusion matrix
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(
    val_gen, model, class_names: list, output_dir: Path
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    val_gen.reset()
    y_true = []
    y_pred = []

    steps = len(val_gen)
    for i, (x_batch, y_batch) in enumerate(val_gen):
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        if i + 1 >= steps:
            break

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(
        figsize=(max(10, len(class_names) // 2), max(8, len(class_names) // 2))
    )
    sns.heatmap(
        cm, annot=len(class_names) <= 20,
        fmt="d", cmap="Blues",
        xticklabels=class_names if len(class_names) <= 30 else False,
        yticklabels=class_names if len(class_names) <= 30 else False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fname = output_dir / "confusion_matrix.png"
    plt.savefig(fname)
    plt.close(fig)
    print(f"  Confusion matrix saved: {fname}")


# ---------------------------------------------------------------------------
# Helper: synthetic data fallback
# ---------------------------------------------------------------------------
def _generate_synthetic(data_dir: Path, n_per_class: int = 50) -> None:
    """
    Generates synthetic RGB images organised in train/ and val/ subdirectories
    so that downstream generators work without real data.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("  [WARN] PIL unavailable — cannot generate synthetic data.")
        return

    classes = [
        "Apple___healthy",
        "Apple___Apple_scab",
        "Tomato___Early_blight",
        "Tomato___healthy",
        "Corn_(maize)___healthy",
    ]

    rng = np.random.default_rng(42)

    for split, n in [("train", n_per_class), ("val", max(10, n_per_class // 5))]:
        for cls in classes:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                fpath = cls_dir / f"synth_{i:04d}.jpg"
                if fpath.exists():
                    continue
                arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                PILImage.fromarray(arr).save(str(fpath))

    print(
        f"  Synthetic data generated in {data_dir} "
        f"({len(classes)} classes × {n_per_class} train images)."
    )


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def run_cnn_pipeline() -> dict:
    """
    Full CNN training pipeline for plant disease detection.
    Returns dict with val_acc, val_loss, num_classes, class_names.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        import cv2
    except ImportError as exc:
        print(f"[ERROR] Missing required dependency: {exc}")
        raise

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Find data directory
    # ------------------------------------------------------------------
    print("[INFO] Locating dataset...")
    data_dir = _find_data_dir()
    print(f"  Detected data root: {data_dir}")

    # ------------------------------------------------------------------
    # Step 2: Fall back to synthetic data if train/ is missing
    # ------------------------------------------------------------------
    if not (data_dir / "train").exists():
        print("[INFO] No real images found, using synthetic data")
        _generate_synthetic(data_dir)

    # ------------------------------------------------------------------
    # Step 3: Remove corrupted images
    # ------------------------------------------------------------------
    print("[INFO] Cleaning corrupted images...")
    _clean_corrupted(data_dir)

    # ------------------------------------------------------------------
    # Step 4: Build generators
    # ------------------------------------------------------------------
    print("[INFO] Building data generators...")
    train_gen, val_gen, num_classes, class_names = _build_generators(data_dir)
    print(f"  Classes: {num_classes}  |  Train samples: {train_gen.n}  |  Val samples: {val_gen.n}")

    # ------------------------------------------------------------------
    # Step 5: Phase 1 — frozen base training
    # ------------------------------------------------------------------
    print("\n[INFO] Phase 1: Training with frozen base...")
    model, base_model = _build_model(num_classes, freeze_base=True)

    callbacks_p1 = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    history_p1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Step 6: Plot Phase 1 history
    # ------------------------------------------------------------------
    _plot_history(history_p1, "Phase 1 Frozen", OUTPUTS_DIR)

    # ------------------------------------------------------------------
    # Step 7: Phase 2 — fine-tune last 30 layers
    # ------------------------------------------------------------------
    print("\n[INFO] Phase 2: Fine-tuning last 30 layers of base...")
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    history_p2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Step 8: Plot Phase 2 history
    # ------------------------------------------------------------------
    _plot_history(history_p2, "Phase 2 Fine-tune", OUTPUTS_DIR)

    # ------------------------------------------------------------------
    # Step 9: Evaluate on validation set
    # ------------------------------------------------------------------
    print("\n[INFO] Evaluating on validation set...")
    val_gen.reset()
    results = model.evaluate(val_gen, verbose=1)
    val_loss = results[0]
    val_acc = results[1]
    print(f"  val_loss    : {val_loss:.4f}")
    print(f"  val_accuracy: {val_acc:.4f}")

    # ------------------------------------------------------------------
    # Step 10: Classification report
    # ------------------------------------------------------------------
    print("\n[INFO] Generating classification report...")
    val_gen.reset()
    y_true, y_pred = [], []
    steps = len(val_gen)
    for i, (x_batch, y_batch) in enumerate(val_gen):
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        if i + 1 >= steps:
            break

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    # Save report to file
    report_path = OUTPUTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    print(f"  Classification report saved: {report_path}")

    # ------------------------------------------------------------------
    # Step 11: Confusion matrix
    # ------------------------------------------------------------------
    print("\n[INFO] Plotting confusion matrix...")
    try:
        _plot_confusion_matrix(val_gen, model, class_names, OUTPUTS_DIR)
    except Exception as e:
        print(f"  [WARN] Could not plot confusion matrix: {e}")

    # ------------------------------------------------------------------
    # Step 12: Grad-CAM on first validation image
    # ------------------------------------------------------------------
    print("\n[INFO] Running Grad-CAM on first validation image...")
    try:
        val_gen.reset()
        x_sample, y_sample = next(iter(val_gen))

        # Find last Conv2D layer name in base model
        last_conv_name = None
        for layer in reversed(base_model.layers):
            if "conv" in layer.name.lower():
                last_conv_name = layer.name
                break

        if last_conv_name is None:
            # MobileNetV2 fallback
            last_conv_name = "out_relu"

        heatmap = _gradcam(x_sample, model, last_conv_name)

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay on original image
        orig_img = (x_sample[0] * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        gradcam_path = OUTPUTS_DIR / "gradcam_sample.jpg"
        cv2.imwrite(str(gradcam_path), superimposed)
        print(f"  Grad-CAM saved: {gradcam_path}")

        severity_pct = _severity(heatmap_resized, threshold=0.5)
        print(f"  Disease severity: {severity_pct:.1f}% of image area above threshold")

        # ── Extract full image feature table ─────────────────────────
        print("\n[INFO] Extracting image features (XAI output)...")
        image_features = extract_image_features(
            model, x_sample[0:1], class_names, heatmap=heatmap_resized
        )
        _print_feature_table(image_features)

        # Save features to CSV
        import pandas as pd
        feat_df = pd.DataFrame([image_features])
        feat_path = OUTPUTS_DIR / "image_features_sample.csv"
        feat_df.to_csv(feat_path, index=False)
        print(f"  Image features saved: {feat_path}")

        # Save multi-sample features — 1 image per class for fusion
        print("\n[INFO] Extracting features from 1 image per class (for fusion)...")
        val_gen.reset()
        class_to_idx = {v: k for k, v in val_gen.class_indices.items()}
        seen_classes, all_feats, sample_idx = set(), [], 0
        for x_batch_f, y_batch_f in val_gen:
            for j in range(len(x_batch_f)):
                cls_idx = int(np.argmax(y_batch_f[j]))
                if cls_idx in seen_classes:
                    continue
                seen_classes.add(cls_idx)
                img_j = x_batch_f[j:j+1]
                try:
                    hmap_j = _gradcam(img_j, model, last_conv_name)
                    hmap_j = cv2.resize(hmap_j, IMG_SIZE)
                except Exception:
                    hmap_j = None
                f = extract_image_features(model, img_j, class_names, heatmap=hmap_j)
                f["Sample_Index"] = sample_idx
                all_feats.append(f)
                sample_idx += 1
            if len(seen_classes) >= len(class_names):
                break
        multi_df = pd.DataFrame(all_feats)
        multi_path = OUTPUTS_DIR / "image_features_multi.csv"
        multi_df.to_csv(multi_path, index=False)
        print(f"  Multi-sample features saved: {multi_path}")
        print("\n  Sample batch features:")
        print(multi_df[["Sample_Index","Disease_Class","Confidence_Pct",
                         "Yellowing_Index","Browning_Index",
                         "Disease_Severity_Pct"]].to_string(index=False))

    except Exception as e:
        print(f"  [WARN] Grad-CAM / feature extraction failed: {e}")
        image_features = {}

    # ------------------------------------------------------------------
    # Step 13: Save model
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "plant_disease_cnn.keras"
    print(f"\n[INFO] Saving model to {model_path}...")
    model.save(str(model_path))
    print(f"  Model saved: {model_path}")

    # ------------------------------------------------------------------
    # Step 14: Return results
    # ------------------------------------------------------------------
    return {
        "val_acc":        val_acc,
        "val_loss":       val_loss,
        "num_classes":    num_classes,
        "class_names":    class_names,
        "image_features": image_features,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        results = run_cnn_pipeline()
        print("\n[DONE] Pipeline complete.")
        print(f"  val_accuracy : {results['val_acc']:.4f}")
        print(f"  val_loss     : {results['val_loss']:.4f}")
        print(f"  num_classes  : {results['num_classes']}")
    except Exception as exc:
        print(f"\n[FATAL] CNN pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

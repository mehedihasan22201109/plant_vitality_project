"""
PART 5b — Explainability
  - SHAP summary plot  (global feature importance)
  - SHAP waterfall     (single prediction explanation)
  - GradCAM++          (CNN — better than plain GradCAM)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

OUTPUTS = Path("outputs")


# ═════════════════════════════════════════════════════════════════════════════
# 1. SHAP — Tabular models
# ═════════════════════════════════════════════════════════════════════════════

def run_shap(best_reg, best_cls, X_tr, X_te, yr_te, yc_te, feature_names):
    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed.")
        return

    print("\n  Running SHAP explainability...")

    feat_labels = [f.replace("_", " ") for f in feature_names]

    X_te_arr = X_te.values if hasattr(X_te, "values") else np.array(X_te)
    X_tr_arr = X_tr.values if hasattr(X_tr, "values") else np.array(X_tr)
    feat_arr = np.array(feature_names)

    # background sample for KernelExplainer (100 rows)
    bg = shap.sample(X_tr_arr, 100, random_state=42)

    # ── 1a. Regression SHAP ──────────────────────────────────────────────────
    try:
        def reg_predict(x):
            return best_reg.predict(x.astype(np.float32))
        explainer_reg = shap.KernelExplainer(reg_predict, bg)
        # Use 200 test samples for speed
        sample_te = X_te_arr[:200]
        shap_vals_reg = explainer_reg.shap_values(sample_te, silent=True)
        importance_reg = np.abs(shap_vals_reg).mean(axis=0)
        order_reg = np.argsort(importance_reg)[::-1][:15]

        # Global bar plot
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(order_reg)))[::-1]
        bars = ax.barh(feat_arr[order_reg][::-1], importance_reg[order_reg][::-1],
                       color=colors, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|  (impact on predicted yield hg/ha)", fontsize=11)
        ax.set_title("SHAP Feature Importance — Yield Regression (XGBoost)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, importance_reg[order_reg][::-1]):
            ax.text(val + importance_reg.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(OUTPUTS / "shap_regression_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/shap_regression_bar.png")

        # Beeswarm-style: SHAP value distribution per feature
        fig, ax = plt.subplots(figsize=(9, 7))
        top_vals = shap_vals_reg[:, order_reg[:12]]
        top_names = feat_arr[order_reg[:12]]
        bp = ax.boxplot(top_vals, vert=False, labels=top_names,
                        patch_artist=True,
                        boxprops=dict(facecolor="#AED6F1", color="#2874A6"),
                        medianprops=dict(color="#E74C3C", linewidth=2))
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
        ax.set_title("SHAP Distribution — Yield Regression (Top 12 Features)",
                     fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUTPUTS / "shap_regression_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/shap_regression_beeswarm.png")

        # Waterfall for single prediction (highest yield)
        preds_sample = best_reg.predict(sample_te)
        best_idx = int(np.argmax(preds_sample))
        sv = shap_vals_reg[best_idx]
        base = explainer_reg.expected_value
        order_w = np.argsort(np.abs(sv))[::-1][:12]

        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative = base
        y_pos = list(range(len(order_w)))
        for i, fi in enumerate(order_w[::-1]):
            color = "#E74C3C" if sv[fi] > 0 else "#2ECC71"
            ax.barh(i, sv[fi], left=cumulative, color=color,
                    edgecolor="white", height=0.6)
            ax.text(cumulative + sv[fi] + (importance_reg.max() * 0.01 * np.sign(sv[fi])),
                    i, f"{feat_arr[fi]}: {sv[fi]:+.0f}", va="center", fontsize=9)
            cumulative += sv[fi]
        ax.axvline(base, color="gray", linestyle="--", linewidth=1, label=f"Base: {base:.0f}")
        ax.set_yticks([])
        ax.set_xlabel("Predicted Yield (hg/ha)", fontsize=11)
        ax.set_title("SHAP Waterfall — Highest Yield Prediction", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUTPUTS / "shap_regression_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/shap_regression_waterfall.png")

    except Exception as e:
        print(f"  [WARN] Regression SHAP failed: {e}")

    # ── 1b. Classification SHAP ──────────────────────────────────────────────
    try:
        def cls_predict_proba(x):
            from sklearn.preprocessing import LabelEncoder
            raw = best_cls.predict(x)
            if hasattr(raw[0], 'item') and isinstance(raw[0].item(), int):
                le = LabelEncoder()
                le.classes_ = np.array(['High', 'Low', 'Medium'])
                labels = le.inverse_transform(raw.astype(int))
            else:
                labels = raw
            proba = np.zeros((len(labels), 3))
            for i, l in enumerate(labels):
                proba[i, ['High','Low','Medium'].index(str(l))] = 1.0
            return proba

        explainer_cls = shap.KernelExplainer(cls_predict_proba, bg)
        shap_vals_cls = explainer_cls.shap_values(X_te_arr[:100], silent=True)
        # shap_vals_cls is list of 3 arrays (one per class)
        importance_cls = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_cls], axis=0)
        order_cls = np.argsort(importance_cls)[::-1][:15]

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(order_cls)))[::-1]
        bars = ax.barh(feat_arr[order_cls][::-1], importance_cls[order_cls][::-1],
                       color=colors, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value| (avg across High/Low/Medium classes)", fontsize=11)
        ax.set_title("SHAP Feature Importance — Yield Classification (XGBoost)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, importance_cls[order_cls][::-1]):
            ax.text(val + importance_cls.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(OUTPUTS / "shap_classification_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/shap_classification_bar.png")

    except Exception as e:
        print(f"  [WARN] Classification SHAP failed: {e}")

    print("  SHAP explainability done.")


# ═════════════════════════════════════════════════════════════════════════════
# 2. GradCAM++ — CNN
# ═════════════════════════════════════════════════════════════════════════════

def _make_gradcampp(model, img_array, class_idx, layer_name="Conv_1"):
    """Compute GradCAM++ heatmap."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_out, preds = grad_model(img_array, training=False)
            loss = preds[:, class_idx]
        grads  = tape1.gradient(loss, conv_out)
    grads2 = tape2.gradient(grads, conv_out)          # second-order

    grads_val  = grads.numpy()[0]
    grads2_val = grads2.numpy()[0]
    conv_val   = conv_out.numpy()[0]

    # GradCAM++ weights
    denom      = 2 * grads2_val + conv_val * (grads2_val ** 2 + 1e-8)
    alpha      = np.where(denom != 0, grads2_val / denom, 0)
    weights    = np.sum(alpha * np.maximum(grads_val, 0), axis=(0, 1))

    cam = np.sum(conv_val * weights, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def run_gradcampp(cnn_model, val_gen=None, sample_img_path=None):
    """
    Generate GradCAM++ heatmaps.
    Uses sample_img_path if provided, else picks first validation image.
    """
    try:
        import tensorflow as tf
        from PIL import Image
    except ImportError:
        print("  [SKIP] TensorFlow/PIL not available for GradCAM++.")
        return

    print("\n  Running GradCAM++...")

    # ── Load image ────────────────────────────────────────────────────────────
    if sample_img_path and Path(sample_img_path).exists():
        img_path = Path(sample_img_path)
    else:
        # Pick first test image available
        test_dir = Path("data/test_images")
        candidates = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
        if not candidates:
            print("  [SKIP] No test images in data/test_images/")
            return
        img_path = candidates[0]

    img   = Image.open(img_path).convert("RGB").resize((224, 224))
    arr   = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)

    # ── Predict ───────────────────────────────────────────────────────────────
    preds     = cnn_model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    conf      = preds[class_idx] * 100

    CLASS_NAMES = [
        'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
        'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
        'Grape___Black_rot','Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
        'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
        'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
        'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    pred_label = CLASS_NAMES[class_idx].replace("___", "\n").replace("_", " ")

    # ── GradCAM++ heatmap ─────────────────────────────────────────────────────
    try:
        # Find last conv layer
        layer_name = None
        for layer in reversed(cnn_model.layers):
            if len(layer.output_shape) == 4:
                layer_name = layer.name
                break
        if layer_name is None:
            layer_name = "Conv_1"

        cam = _make_gradcampp(cnn_model, batch, class_idx, layer_name)
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
        ) / 255.0

        # ── Plot: original | heatmap | overlay ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle(
            f"GradCAM++  |  {pred_label}  ({conf:.1f}%)",
            fontsize=12, fontweight="bold", y=1.02
        )

        axes[0].imshow(arr)
        axes[0].set_title("Original Image", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title("GradCAM++ Heatmap", fontsize=11)
        axes[1].axis("off")

        heatmap_color = cm.jet(cam_resized)[..., :3]
        overlay       = 0.55 * arr + 0.45 * heatmap_color
        overlay       = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(OUTPUTS / "gradcampp_sample.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → outputs/gradcampp_sample.png")
        print(f"  Prediction : {CLASS_NAMES[class_idx]}  ({conf:.1f}%)")

    except Exception as e:
        print(f"  [WARN] GradCAM++ heatmap failed: {e}")

    # ── Multi-class comparison: top-5 GradCAM++ side by side ─────────────────
    try:
        top5    = np.argsort(preds)[::-1][:5]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.suptitle("GradCAM++ — Top-5 Predicted Classes", fontsize=13, fontweight="bold")

        for ax, idx in zip(axes, top5):
            cam_i = _make_gradcampp(cnn_model, batch, idx, layer_name)
            cam_r = np.array(
                Image.fromarray((cam_i * 255).astype(np.uint8)).resize((224, 224))
            ) / 255.0
            heat  = cm.jet(cam_r)[..., :3]
            ov    = np.clip(0.55 * arr + 0.45 * heat, 0, 1)
            name  = CLASS_NAMES[idx].replace("___", "\n").replace("_", " ")
            ax.imshow(ov)
            ax.set_title(f"{name}\n{preds[idx]*100:.1f}%", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(OUTPUTS / "gradcampp_top5.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/gradcampp_top5.png")

    except Exception as e:
        print(f"  [WARN] Top-5 GradCAM++ failed: {e}")

    print("  GradCAM++ done.")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Entry point called from main.py
# ═════════════════════════════════════════════════════════════════════════════

def run_explainability(best_reg, best_cls, X_tr, X_te, yr_te, yc_te,
                       feature_names, cnn_model=None):
    print("\n" + "=" * 60)
    print("  PART 5b — EXPLAINABILITY (SHAP + GradCAM++)")
    print("=" * 60)

    # SHAP for tabular models
    run_shap(best_reg, best_cls, X_tr, X_te, yr_te, yc_te, feature_names)

    # GradCAM++ for CNN
    if cnn_model is not None:
        run_gradcampp(cnn_model)
    else:
        print("\n  [SKIP] GradCAM++ — CNN model not provided.")

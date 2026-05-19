import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_eda(df: pd.DataFrame) -> None:
    # Plot 1: Yield distribution histogram + median yield by Crop_Type bar chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "yield_hgha" in df.columns:
            axes[0].hist(df["yield_hgha"].dropna(), bins=30, color="steelblue", edgecolor="white")
            axes[0].axvline(df["yield_hgha"].median(), color="red", linestyle="--", label=f"Median: {df['yield_hgha'].median():.0f}")
            axes[0].set_title("Yield Distribution")
            axes[0].set_xlabel("Yield (hg/ha)")
            axes[0].set_ylabel("Count")
            axes[0].legend()
        else:
            axes[0].set_title("yield_hgha column not found")

        if "Crop_Type" in df.columns and "yield_hgha" in df.columns:
            median_yield = df.groupby("Crop_Type")["yield_hgha"].median().sort_values(ascending=False)
            axes[1].bar(median_yield.index, median_yield.values, color="coral", edgecolor="white")
            axes[1].set_title("Median Yield by Crop Type")
            axes[1].set_xlabel("Crop Type")
            axes[1].set_ylabel("Median Yield (hg/ha)")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[1].set_title("Crop_Type or yield_hgha column not found")

        plt.tight_layout()
        out = OUTPUT_DIR / "eda_yield_distribution.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()

    # Plot 2: Correlation heatmap of numeric columns
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(12, 9))
            corr = numeric_df.corr()
            sns.heatmap(
                corr,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Correlation Heatmap (Numeric Features)")
            plt.tight_layout()
            out = OUTPUT_DIR / "eda_correlation_heatmap.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved -> {out}")
        else:
            plt.close()
    except Exception:
        plt.close()

    # Plot 3: Scatter plots of avg_temp, rainfall_mm, nitrogen_N vs yield_hgha
    try:
        scatter_cols = ["avg_temp", "rainfall_mm", "nitrogen_N"]
        available = [c for c in scatter_cols if c in df.columns]
        if available and "yield_hgha" in df.columns:
            n = len(available)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, available):
                ax.scatter(df[col], df["yield_hgha"], alpha=0.4, s=10, color="teal")
                ax.set_xlabel(col)
                ax.set_ylabel("yield_hgha")
                ax.set_title(f"{col} vs yield_hgha")
            plt.tight_layout()
            out = OUTPUT_DIR / "eda_scatter_plots.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved -> {out}")
        else:
            plt.close()
    except Exception:
        plt.close()

    # Plot 4: Pie chart of yield category (Low/Medium/High)
    try:
        if "yield_hgha" in df.columns:
            bins = [0, 20000, 50000, float("inf")]
            labels = ["Low", "Medium", "High"]
            cats = pd.cut(df["yield_hgha"].dropna(), bins=bins, labels=labels)
            counts = cats.value_counts().reindex(labels, fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=["#ff9999", "#66b3ff", "#99ff99"],
                startangle=140,
            )
            ax.set_title("Yield Category Distribution")
            plt.tight_layout()
            out = OUTPUT_DIR / "eda_yield_categories.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved -> {out}")
        else:
            plt.close()
    except Exception:
        plt.close()

    # Plot 5: Line chart of avg yield by Year
    try:
        if "Year" in df.columns and "yield_hgha" in df.columns:
            avg_by_year = df.groupby("Year")["yield_hgha"].mean().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(avg_by_year.index, avg_by_year.values, marker="o", color="darkorange", linewidth=2)
            ax.set_title("Average Yield Trend by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg Yield (hg/ha)")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out = OUTPUT_DIR / "eda_yield_trend.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved -> {out}")
        else:
            plt.close()
    except Exception:
        plt.close()


def plot_regression_results(df: pd.DataFrame) -> None:
    try:
        metrics = ["RMSE", "MAE", "R2"]
        available = [m for m in metrics if m in df.columns]
        if not available or "Model" not in df.columns:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        colors = ["steelblue", "coral", "mediumseagreen"]
        for ax, metric, color in zip(axes, available, colors):
            ax.bar(df["Model"], df[metric], color=color, edgecolor="white")
            ax.set_title(f"{metric} Comparison")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)

        plt.suptitle("Regression Model Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "regression_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()


def plot_classification_results(df: pd.DataFrame) -> None:
    try:
        metrics = ["Accuracy", "F1_weighted"]
        available = [m for m in metrics if m in df.columns]
        if not available or "Model" not in df.columns:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        colors = ["royalblue", "tomato"]
        for ax, metric, color in zip(axes, available, colors):
            ax.bar(df["Model"], df[metric], color=color, edgecolor="white")
            ax.set_title(f"{metric} Comparison")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)

        plt.suptitle("Classification Model Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "classification_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str, labels: list) -> None:
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_")
        out = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()


def plot_feature_importance(model, feature_names: list, title: str) -> None:
    try:
        if not hasattr(model, "feature_importances_"):
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(top_features)), top_importances[::-1], color="mediumslateblue", edgecolor="white")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 15 Feature Importances — {title}")
        plt.tight_layout()
        safe_title = title.replace(" ", "_")
        out = OUTPUT_DIR / f"feature_importance_{safe_title}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> None:
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue", label="Predictions")
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {model_name}")
        ax.legend()
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_")
        out = OUTPUT_DIR / f"actual_vs_pred_{safe_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()


def plot_training_history(history, title: str) -> None:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy subplot
        ax = axes[0]
        if hasattr(history, "history"):
            hist_dict = history.history
        elif isinstance(history, dict):
            hist_dict = history
        else:
            hist_dict = {}

        if "accuracy" in hist_dict:
            ax.plot(hist_dict["accuracy"], label="Train Accuracy", color="steelblue")
        if "val_accuracy" in hist_dict:
            ax.plot(hist_dict["val_accuracy"], label="Val Accuracy", color="coral", linestyle="--")
        ax.set_title("Model Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # Loss subplot
        ax = axes[1]
        if "loss" in hist_dict:
            ax.plot(hist_dict["loss"], label="Train Loss", color="steelblue")
        if "val_loss" in hist_dict:
            ax.plot(hist_dict["val_loss"], label="Val Loss", color="coral", linestyle="--")
        ax.set_title("Model Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(f"Training History — {title}", fontsize=14)
        plt.tight_layout()
        safe_title = title.replace(" ", "_")
        out = OUTPUT_DIR / f"cnn_{safe_title}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")
    except Exception:
        plt.close()

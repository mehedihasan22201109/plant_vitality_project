
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Local config constants (mirrors src/config.py so this module is standalone)
# ---------------------------------------------------------------------------
TARGET = "yield_hgha"
FEATURE_COLS = [
    "Year",
    "avg_temp",
    "rainfall_mm",
    "humidity_pct",
    "min_temp",
    "max_temp",
    "wind_speed_kmh",
    "sunshine_hours",
    "season",
    "nitrogen_N",
    "phosphorous_P",
    "potassium_K",
    "soil_pH",
    "soil_moisture_pct",
    "soil_type",
    "fertilizer_kgha",
    "AQI",
    "CO2_ppm",
    "PM25_ugm3",
    "NO2_ppb",
    "decade",
    "temperature_range",
]
YIELD_BINS = [0, 20_000, 50_000, float("inf")]
YIELD_LABELS = ["Low", "Medium", "High"]
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

CATEGORICAL_COLS = ["season", "soil_type"]


# ---------------------------------------------------------------------------
# 1. Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vectors for modelling.

    Returns
    -------
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, feature_names
    """
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET]).copy()

    # Regression target
    y_reg = df[TARGET].astype(float)

    # Classification target — drop rows where cut produces NaN (out-of-bin values)
    y_cls_raw = pd.cut(df[TARGET], bins=YIELD_BINS, labels=YIELD_LABELS)
    valid_mask = y_cls_raw.notna()
    df = df[valid_mask].copy()
    y_reg = y_reg[valid_mask]
    y_cls = y_cls_raw[valid_mask].astype(str)  # convert Categorical → str

    # Keep only FEATURE_COLS that exist in the dataframe
    used_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[used_cols].copy()

    # Encode categorical columns with LabelEncoder
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            le = LabelEncoder()
            # Fill NaN with a placeholder string before encoding
            X[col] = X[col].fillna("__missing__")
            X[col] = le.fit_transform(X[col].astype(str))

    # Impute remaining missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    feature_names = used_cols

    # Align indices before split
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    y_reg = y_reg.reset_index(drop=True)
    y_cls = y_cls.reset_index(drop=True)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X_df, y_reg, y_cls, test_size=0.20, random_state=42
    )

    return X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, feature_names


# ---------------------------------------------------------------------------
# 2. Regression
# ---------------------------------------------------------------------------

def run_regression(
    X_tr, X_te, yr_tr, yr_te, feature_names: list
) -> tuple[pd.DataFrame, object]:
    """
    Train regression models, evaluate, persist the best one.

    Returns
    -------
    results_df : pd.DataFrame  — columns [Model, RMSE, MAE, R2, CV_R2]
    best_model : fitted estimator with the highest R2 on the test set
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        ),
        "KNeighbors": KNeighborsRegressor(n_neighbors=7),
    }

    records = []
    best_r2 = -np.inf
    best_model = None

    print("\n=== Regression Results ===")
    print(f"{'Model':<22} {'RMSE':>12} {'MAE':>12} {'R2':>8} {'CV_R2':>8}")
    print("-" * 66)

    for name, model in models.items():
        model.fit(X_tr, yr_tr)
        preds = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(yr_te, preds)))
        mae = float(mean_absolute_error(yr_te, preds))
        r2 = float(r2_score(yr_te, preds))

        try:
            cv_scores = cross_val_score(
                model, X_tr, yr_tr, scoring="r2", cv=5, n_jobs=-1
            )
            cv_r2 = float(np.mean(cv_scores))
        except Exception:
            cv_r2 = np.nan

        print(
            f"{name:<22} {rmse:>12.2f} {mae:>12.2f} {r2:>8.4f} {cv_r2:>8.4f}"
        )
        records.append(
            {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "CV_R2": cv_r2}
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    print("-" * 66)
    results_df = pd.DataFrame(records, columns=["Model", "RMSE", "MAE", "R2", "CV_R2"])

    # Persist best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_regressor.pkl"
    joblib.dump(best_model, save_path)
    best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
    print(f"\nBest regressor: {best_name}  (R2={best_r2:.4f})  saved → {save_path}\n")

    return results_df, best_model


# ---------------------------------------------------------------------------
# 3. Classification
# ---------------------------------------------------------------------------

def run_classification(
    X_tr, X_te, yc_tr, yc_te, feature_names: list
) -> tuple[pd.DataFrame, object]:
    """
    Train classification models, evaluate, persist the best one.

    XGBClassifier requires integer labels — handled internally via LabelEncoder.

    Returns
    -------
    results_df : pd.DataFrame  — columns [Model, Accuracy, F1_weighted]
    best_model : fitted estimator with the highest weighted F1 on the test set
    """
    # Label encoder reserved for XGBoost
    xgb_le = LabelEncoder()
    xgb_le.fit(yc_tr)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss",
        ),
        "KNeighbors": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
    }

    records = []
    best_f1 = -np.inf
    best_model = None
    best_preds = None

    print("\n=== Classification Results ===")
    print(f"{'Model':<22} {'Accuracy':>10} {'F1_weighted':>12}")
    print("-" * 48)

    for name, model in models.items():
        if name == "XGBoost":
            # Encode string labels → integers for XGBoost
            yc_tr_enc = xgb_le.transform(yc_tr)
            yc_te_enc = xgb_le.transform(yc_te)
            model.fit(X_tr, yc_tr_enc)
            preds_enc = model.predict(X_te)
            preds = xgb_le.inverse_transform(preds_enc)
        else:
            model.fit(X_tr, yc_tr)
            preds = model.predict(X_te)

        acc = float(accuracy_score(yc_te, preds))
        f1 = float(f1_score(yc_te, preds, average="weighted"))

        print(f"{name:<22} {acc:>10.4f} {f1:>12.4f}")
        records.append({"Model": name, "Accuracy": acc, "F1_weighted": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_preds = preds

    print("-" * 48)
    results_df = pd.DataFrame(
        records, columns=["Model", "Accuracy", "F1_weighted"]
    )

    # Print classification report for the best model
    best_name = results_df.loc[results_df["F1_weighted"].idxmax(), "Model"]
    print(f"\nClassification report for best model ({best_name}):")
    print(classification_report(yc_te, best_preds))

    # Persist best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_classifier.pkl"
    joblib.dump(best_model, save_path)
    print(f"Best classifier: {best_name}  (F1={best_f1:.4f})  saved → {save_path}\n")

    return results_df, best_model

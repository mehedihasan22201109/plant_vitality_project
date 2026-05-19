import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Load models ───────────────────────────────────────────────────────────────
print("\nLoading models...", end=" ", flush=True)

regressor  = joblib.load("models/best_regressor.pkl")
classifier = joblib.load("models/best_classifier.pkl")
feat_names = joblib.load("models/feature_names.pkl")

df_train = pd.read_csv("data/processed/merged_dataset.csv")
X_train  = df_train[feat_names].copy()

label_encoders = {}
for col in ["season", "soil_type"]:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = X_train[col].fillna("__missing__")
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
scaler.fit(imputer.fit_transform(X_train))

import tensorflow as tf
cnn = tf.keras.models.load_model("models/plant_disease_cnn.keras")

print("Done.\n")

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
LABEL_MAP = {0: 'High', 1: 'Low', 2: 'Medium'}


def get_float(prompt):
    while True:
        try:
            return float(input(f"  {prompt}: "))
        except ValueError:
            print("  Please enter a number.")

def get_choice(prompt, options):
    opts = [o.lower() for o in options]
    while True:
        val = input(f"  {prompt} ({'/'.join(options)}): ").strip().lower()
        if val in opts:
            return options[opts.index(val)]
        print(f"  Choose from: {', '.join(options)}")


# ═════════════════════════════════════════════════════════════════════════════
def predict_yield():
    print("\n" + "="*55)
    print("  CROP YIELD PREDICTION")
    print("="*55)
    print("  Enter the crop and field conditions:\n")

    year        = get_float("Year (e.g. 2024)")
    avg_temp    = get_float("Average Temperature °C")
    min_temp    = get_float("Min Temperature °C")
    max_temp    = get_float("Max Temperature °C")
    rainfall    = get_float("Annual Rainfall mm")
    humidity    = get_float("Humidity %")
    wind        = get_float("Wind Speed km/h")
    sunshine    = get_float("Sunshine Hours per day")
    season      = get_choice("Season", ["Kharif", "Rabi"])
    nitrogen    = get_float("Nitrogen N mg/kg")
    phosphorous = get_float("Phosphorous P mg/kg")
    potassium   = get_float("Potassium K mg/kg")
    soil_ph     = get_float("Soil pH")
    soil_moist  = get_float("Soil Moisture %")
    soil_type   = get_choice("Soil Type", ["Loam", "Sandy", "Clay", "Silt"])
    fertilizer  = get_float("Fertilizer kg/ha")
    aqi         = get_float("AQI")
    co2         = get_float("CO2 ppm")
    pm25        = get_float("PM2.5 µg/m³")
    no2         = get_float("NO2 ppb")

    row = {
        "Year": year, "avg_temp": avg_temp, "rainfall_mm": rainfall,
        "humidity_pct": humidity, "min_temp": min_temp, "max_temp": max_temp,
        "wind_speed_kmh": wind, "sunshine_hours": sunshine, "season": season,
        "nitrogen_N": nitrogen, "phosphorous_P": phosphorous, "potassium_K": potassium,
        "soil_pH": soil_ph, "soil_moisture_pct": soil_moist, "soil_type": soil_type,
        "fertilizer_kgha": fertilizer, "AQI": aqi, "CO2_ppm": co2,
        "PM25_ugm3": pm25, "NO2_ppb": no2,
        "decade": float(int(year) // 10 * 10),
        "temperature_range": max_temp - min_temp,
    }

    df = pd.DataFrame([row])[feat_names].copy()
    for col in ["season", "soil_type"]:
        if col in df.columns and col in label_encoders:
            le  = label_encoders[col]
            val = str(df[col].iloc[0])
            val = val if val in le.classes_ else le.classes_[0]
            df[col] = le.transform([val])[0]

    X = scaler.transform(imputer.transform(df))

    yield_pred = regressor.predict(X)[0]
    cls_raw    = classifier.predict(X)[0]
    cls_label  = LABEL_MAP.get(int(cls_raw), str(cls_raw)) if isinstance(cls_raw, (int, np.integer)) else str(cls_raw)

    print("\n" + "="*55)
    print("  RESULT")
    print("="*55)
    print(f"  Predicted Yield  :  {yield_pred:>10,.0f}  hg/ha")
    print(f"  In tonnes/ha     :  {yield_pred/10000:>10.2f}  t/ha")
    print(f"  Yield Category   :  {cls_label}")
    print("="*55)


# ═════════════════════════════════════════════════════════════════════════════
def predict_disease():
    from PIL import Image

    print("\n" + "="*55)
    print("  PLANT DISEASE DETECTION")
    print("="*55)

    img_path = input("  Leaf image path: ").strip().strip('"')

    # Windows path → WSL path
    if len(img_path) > 1 and img_path[1] == ':':
        drive    = img_path[0].lower()
        img_path = f"/mnt/{drive}/" + img_path[3:].replace("\\", "/")

    if not os.path.exists(img_path):
        print(f"  File not found: {img_path}")
        return

    img  = Image.open(img_path).convert("RGB").resize((224, 224))
    arr  = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    pred = cnn.predict(arr, verbose=0)[0]
    top3 = np.argsort(pred)[::-1][:3]

    print("\n" + "="*55)
    print("  RESULT")
    print("="*55)
    for rank, idx in enumerate(top3):
        name = CLASS_NAMES[idx].replace("___", " → ").replace("_", " ")
        conf = pred[idx] * 100
        bar  = "█" * int(conf / 5)
        mark = "▶" if rank == 0 else "  "
        print(f"  {mark} #{rank+1}  {conf:5.1f}%  {name}")

    top_name  = CLASS_NAMES[top3[0]]
    is_healthy = "healthy" in top_name.lower()
    print()
    if is_healthy:
        print("  Status: ✓  Plant is HEALTHY")
    else:
        print("  Status: ⚠  DISEASE DETECTED")
    print("="*55)


# ═════════════════════════════════════════════════════════════════════════════
while True:
    print("\n  What do you want to test?")
    print("  1 → Crop Yield Prediction")
    print("  2 → Plant Disease Detection")
    print("  q → Quit")
    choice = input("\n  Choice: ").strip().lower()

    if choice == '1':
        predict_yield()
    elif choice == '2':
        predict_disease()
    elif choice == 'q':
        print()
        break
    else:
        print("  Enter 1, 2, or q.")

# predict_batch.py
# Supports 4- or 5-column input files.
# If actual PartWeight column provided, compute MAE and %Average Prediction Error.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import sys

# ---------------------------
# Load model & scaler
# ---------------------------
print("Loading model & scaler...")
model = keras.models.load_model("injection_model.keras")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded successfully.")

# ---------------------------
# Load input data
# ---------------------------
input_file = "injection_data_new.csv"  # Change file name as needed
print(f"📂 Reading input file: {input_file}")

data = pd.read_csv(input_file)
n_cols = data.shape[1]

if n_cols < 4:
    sys.exit("❌ Error: Input file must have at least 4 columns.")
elif n_cols == 4:
    print("📄 File with 4 columns detected → prediction-only mode.")
    X = data.iloc[:, 0:4].values
    actual = None
elif n_cols >= 5:
    print("📄 File with 5+ columns detected → prediction + error analysis mode.")
    X = data.iloc[:, 0:4].values
    actual = data.iloc[:, 4].values
else:
    sys.exit("❌ Unexpected file structure.")

# ---------------------------
# Scale input and predict
# ---------------------------
X_scaled = scaler.transform(X)
Y_pred = model.predict(X_scaled, verbose=0).flatten()
data["PredictedPartWeight"] = np.round(Y_pred, 2)

# ---------------------------
# If actual column exists, compute MAE and %Average Prediction Error
# ---------------------------
if actual is not None:
    abs_error = np.abs(actual - Y_pred)
    pct_error = np.abs((actual - Y_pred) / actual) * 100

    MAE = np.mean(abs_error)
    avg_pct_err = np.mean(pct_error)

    data["%Error"] = np.round(((actual - Y_pred) / actual) * 100, 2)

    print(f"📊 Mean Absolute Error (MAE): {MAE:.4f} g")
    print(f"📊 %Average Prediction Error: {avg_pct_err:.2f}%")

else:
    print("✅ Predictions completed (no actual weights available).")

# ---------------------------
# Save predictions
# ---------------------------
output_file = "batch_predictions.csv"
data.to_csv(output_file, index=False)
print(f"💾 Results saved to {output_file}")
print("\nPreview:")
print(data.head(10))
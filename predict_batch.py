# predict_batch.py
# Batch prediction: load dataset CSV, predict weights, save results

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

# ---------------------------
# Load model & scaler
# ---------------------------
print("Loading model & scaler...")
model = keras.models.load_model("injection_model.keras")
scaler = joblib.load("scaler.pkl")
print("Ready for batch predictions!\n")

# ---------------------------
# Load dataset (must include process params)
# ---------------------------
data = pd.read_csv("injection_data.csv")  # can replace with new file
X = data[["InjectionPressure","MeltTemp","CoolingTime","PackingPressure"]].values

# Scale and predict
X_scaled = scaler.transform(X)
Y_pred = model.predict(X_scaled, verbose=0).flatten()

# Build results DataFrame
results = data.copy()
results["PredictedPartWeight"] = np.round(Y_pred, 2)

# Save results
results.to_csv("batch_predictions.csv", index=False)
print("✅ Batch predictions saved to batch_predictions.csv")
print(results.head(10))
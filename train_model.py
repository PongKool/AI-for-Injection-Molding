# train_model.py
# Train NN surrogate model for injection molding and save it
# Shows dataset counts, step-wise timings, and total duration

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib   # for saving scaler
import time

# ---------------------------
# Overall timer
# ---------------------------
t0 = time.time()

# ---------------------------
# Step 1: Load dataset
# ---------------------------
s1 = time.time()
data = pd.read_csv("injection_data.csv")
s2 = time.time()
print("✅ Dataset loaded.")
print(data.head())
print(f"⏱ Step 1 (Load dataset): {s2 - s1:.2f} sec")

X = data[["InjectionPressure","MeltTemp","CoolingTime","PackingPressure"]].values
Y = data["PartWeight"].values

# ---------------------------
# Step 2: Train/Test split
# ---------------------------
s1 = time.time()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
s2 = time.time()

n_total = len(data)
n_train = len(X_train)
n_test = len(X_test)
print(f"\n📊 Dataset summary: Total={n_total}, Train={n_train}, Test={n_test}")
print(f"⏱ Step 2 (Split dataset): {s2 - s1:.2f} sec")

# ---------------------------
# Step 3: Normalize features
# ---------------------------
s1 = time.time()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
s2 = time.time()
print(f"⏱ Step 3 (Normalize data): {s2 - s1:.2f} sec")

# ---------------------------
# Step 4: Build model
# ---------------------------
s1 = time.time()
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
s2 = time.time()
print(f"⏱ Step 4 (Build model): {s2 - s1:.2f} sec")

# ---------------------------
# Step 5: Training
# ---------------------------
s1 = time.time()
print("\n🚀 Training Neural Network...")
history = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)
s2 = time.time()
print(f"✅ Training complete: {n_train} training samples in {s2 - s1:.2f} sec")

# ---------------------------
# Step 6: Evaluation
# ---------------------------
s1 = time.time()
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
s2 = time.time()
print(f"\n📈 Evaluation on {n_test} test samples:")
print(f"   Test MSE: {loss:.4f}, Test MAE: {mae:.4f} g")
print(f"⏱ Step 6 (Evaluate model): {s2 - s1:.2f} sec")

# ---------------------------
# Step 7: Save artifacts
# ---------------------------
s1 = time.time()
model.save("injection_model.keras")
joblib.dump(scaler, "scaler.pkl")
s2 = time.time()
print(f"💾 Model saved as injection_model.keras")
print(f"💾 Scaler saved as scaler.pkl")
print(f"⏱ Step 7 (Save model & scaler): {s2 - s1:.2f} sec")

# ---------------------------
# End: Total time
# ---------------------------
t_total = time.time() - t0
print(f"\n⏱ Total time for training pipeline: {t_total:.2f} sec")
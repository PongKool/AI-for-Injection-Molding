# predict_manual.py
# Manual prediction loop with auto-suggestion
# If predicted Part Weight > 50 g, search for new inputs that yield < 50 g

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
print("Ready for manual predictions!\n")

# ---------------------------
# Helper: suggest new inputs aiming < 50 g
# ---------------------------
def suggest_inputs(current_inputs, target=50.0):
    inj_pressure, melt_temp, cooling_time, pack_pressure = current_inputs

    # define small search ranges (±50 for pressures, ±5 for temp, ±3 sec for cooling)
    inj_range = np.arange(inj_pressure-50, inj_pressure+1, 10)
    pack_range = np.arange(pack_pressure-50, pack_pressure+1, 10)
    temp_range = [melt_temp]  # keep fixed for simplicity
    cool_range = [cooling_time]

    best = None
    best_pred = None

    # brute force grid search
    for inj in inj_range:
        if inj < 500: continue
        for pack in pack_range:
            if pack < 300: continue
            for temp in temp_range:
                for cool in cool_range:
                    candidate = np.array([[inj, temp, cool, pack]])
                    cand_scaled = scaler.transform(candidate)
                    pred = model.predict(cand_scaled, verbose=0).flatten()[0]

                    if pred < target:  # found a candidate below 50
                        if best is None or pred > best_pred:  
                            # choose one CLOSEST to target but still under 50
                            best = (inj, temp, cool, pack)
                            best_pred = pred
    return best, best_pred

# ---------------------------
# Manual prediction loop
# ---------------------------
while True:
    user_input = input("\nEnter InjectionPressure MeltTemp CoolingTime PackingPressure (or 'x' to quit): ")

    if user_input.lower() == "x":
        break

    try:
        # Parse 4 values
        values = user_input.strip().split()
        if len(values) != 4:
            print("⚠️ Please enter exactly 4 numbers separated by spaces.")
            continue

        inj_pressure, melt_temp, cooling_time, pack_pressure = map(float, values)
        arr = np.array([[inj_pressure, melt_temp, cooling_time, pack_pressure]])

        # Predict
        arr_scaled = scaler.transform(arr)
        pred_weight = model.predict(arr_scaled, verbose=0).flatten()[0]
        print(f" 👉 Predicted Part Weight: {pred_weight:.2f} g")

        # ---------------------------
        # Suggestion if > 50 g
        # ---------------------------
        if pred_weight > 50:
            best_input, best_pred = suggest_inputs((inj_pressure, melt_temp, cooling_time, pack_pressure), target=50.0)
            if best_input is not None:
                print(f" 💡 Suggestion: Try InjectionPressure={best_input[0]}, MeltTemp={best_input[1]}, "
                      f"CoolingTime={best_input[2]}, PackingPressure={best_input[3]}")
                print(f"    → This gives ~{best_pred:.2f} g (below 50)")
            else:
                print(" ❌ No suitable lower-weight parameter set found nearby.")
    except ValueError:
        print("⚠️ Invalid input. Please enter 4 numbers separated by spaces or 'x' to quit.")

print("\nExited manual prediction loop. Goodbye 👋")
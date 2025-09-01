import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load CSV
data = pd.read_csv("injection_data.csv")

print(data.head())   # Peek at first 5 rows

X = data[["InjectionPressure","MeltTemp","CoolingTime","PackingPressure"]].values
Y = data["PartWeight"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)   # regression output (part weight)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, Y_train,
                    epochs=200,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=1)

test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print("Test MAE (g):", test_mae)

new_cycle = np.array([[820, 242, 21, 510]])   # Example process inputs
new_cycle_scaled = scaler.transform(new_cycle)

predicted_weight = model.predict(new_cycle_scaled)
print("Predicted part weight (g):", predicted_weight[0][0])
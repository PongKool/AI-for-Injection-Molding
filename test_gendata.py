import numpy as np
import pandas as pd

# number of samples
n_samples = 10

# Random values around realistic ranges
injection_pressure = np.random.randint(750, 900, n_samples)   # bar
melt_temp = np.random.randint(230, 260, n_samples)            # °C
cooling_time = np.random.randint(15, 25, n_samples)           # sec
packing_pressure = np.random.randint(450, 600, n_samples)     # bar

# Synthetic relationship for part weight
# base weight affected by parameters + noise
part_weight = (
    0.01 * (injection_pressure - 750)
    + 0.005 * (melt_temp - 230)
    + 0.02 * (cooling_time - 15)
    + 0.003 * (packing_pressure - 450)
    + 50
    + np.random.normal(0, 0.3, n_samples)   # random noise
)

# Build dataframe
df = pd.DataFrame({
    "InjectionPressure": injection_pressure,
    "MeltTemp": melt_temp,
    "CoolingTime": cooling_time,
    "PackingPressure": packing_pressure,
    "PartWeight": part_weight
})

# Save to CSV
# df.to_csv("injection_data.csv", index=False)
print(df)
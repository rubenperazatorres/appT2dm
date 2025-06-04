import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Carga tu dataset original
df = pd.read_csv("dataSet.csv")

# Selecciona solo las columnas que usas para escalar (sin la variable objetivo)
X = df.drop(columns=["P_DIABETICO"])

# Crea y ajusta un nuevo StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# Extrae los parámetros y guárdalos en JSON
params = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
    "var_": scaler.var_.tolist(),
    "n_features_in_": scaler.n_features_in_
}

with open("scaler_params.json", "w") as f:
    json.dump(params, f)

print("Archivo scaler_params.json creado correctamente.")
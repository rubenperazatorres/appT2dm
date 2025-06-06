# prediccion29_paraRender.py
# Ubicado en: C:\python\mi_proyecto\entrenamiento

import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1) Añadimos el directorio padre al sys.path, para que Python encuentre
#    el módulo anfis_model.py que está en C:\python\mi_proyecto
script_dir = os.path.dirname(__file__)                             # ...\mi_proyecto\entrenamiento
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))       # ...\mi_proyecto
sys.path.append(parent_dir)

from anfis_model import ANFIS   # Ahora sí podemos importar la clase ANFIS

# 2) Definimos la ruta correcta al CSV (dataSet_smote.csv) 
csv_path = os.path.join(script_dir, "dataSet_smote.csv")  # dataSet_smote.csv, no "dataSet.smot.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo de datos en:\n  {csv_path}")

# Cargar el dataset balanceado con SMOTE
data = pd.read_csv(csv_path)

# Separar características (X) y etiqueta (y)
features = [
    'SEXO', 'EDAD', 'IMC', 'NEUTROFILOS', 'HEMOGLOBINA',
    'ERITROCITOS', 'HEMATOCRITO', 'MCH', 'MCV', 'EOSINOFILOS',
    'RDW-CV', 'LINFOCITOS', 'MONOCITOS'
]
target = 'P_DIABETICO'

X = data[features].values
y = data[target].values.reshape(-1, 1)

# 3) Escalar los datos con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Guardar el escalador en .pkl, en la carpeta padre (mi_proyecto/)
scaler_pkl_path = os.path.join(parent_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_pkl_path)

# 5) Guardar también el escalador en JSON para Flask/Render (en la carpeta padre)
scaler_json = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'var': scaler.var_.tolist(),
    'n_features_in_': scaler.n_features_in_,
    'feature_names_in_': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else features
}
scaler_json_path = os.path.join(parent_dir, 'scaler_params.json')
with open(scaler_json_path, 'w') as f:
    json.dump(scaler_json, f, indent=4)

# 6) Convertir a tensores de PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 7) Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# 8) Crear y entrenar el modelo ANFIS
model = ANFIS(n_inputs=13, n_rules=300)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# 9) Guardar el modelo entrenado (solo pesos) en la carpeta padre
model_path = os.path.join(parent_dir, "anfis_state_dict_300.pth")
torch.save(model.state_dict(), model_path)

print("Modelo y escaladores guardados correctamente.")

from anfis_model import ANFIS  # Asegúrate que el nombre del archivo y clase coincidan
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Función para cargar el scaler desde JSON
def load_scaler(json_path):
    with open(json_path, "r") as f:
        params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["mean_"])
    scaler.scale_ = np.array(params["scale_"])
    scaler.var_ = np.array(params["var_"])
    scaler.n_features_in_ = params["n_features_in_"]
    return scaler

# Carga el scaler y modelo al iniciar la app
scaler = load_scaler("scaler_params.json")
# model = torch.load("anfis_modelo_27.pth", map_location=torch.device('cpu'))

model = ANFIS(n_inputs=13, n_rules=300)  # Usa los valores reales que usaste al entrenar
model.load_state_dict(torch.load("anfis_state_dict_27.pth", map_location=torch.device('cpu')))                

model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "features" not in data:
        return jsonify({"error": "No features provided"}), 400
    
    features = np.array(data["features"]).reshape(1, -1)
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        return jsonify({"error": f"Error in scaling features: {str(e)}"}), 400
    
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    prediction = output.numpy().tolist()
    return jsonify({"prediction": prediction})

#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=8000)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render asigna el puerto a la variable de entorno PORT
    app.run(host="0.0.0.0", port=port)
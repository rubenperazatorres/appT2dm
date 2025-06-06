from anfis_model import ANFIS
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
import torch
import os

no_entradas = 13
no_reglas = 300

app = Flask(__name__)

# Función para cargar el scaler desde JSON
def load_scaler(json_path):
    with open(json_path, "r") as f:
        params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["mean"])
    scaler.scale_ = np.array(params["scale"])
    scaler.var_ = np.array(params["var"])
    scaler.n_features_in_ = params["n_features_in_"]
    scaler.feature_names_in_ = np.array(params["feature_names_in_"])

    return scaler


# Carga el scaler y el modelo
scaler = load_scaler("scaler_params.json")

# Inicializa el modelo con los mismos parámetros usados durante el entrenamiento
model = ANFIS(n_inputs=no_entradas, n_rules=no_reglas)
#model.load_state_dict(torch.load("anfis_state_dict_27.pth", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("anfis_state_dict_300.pth", map_location=torch.device('cpu')))

model.eval()

# Ruta para servir la página index.html
@app.route('/')
def index():
    return render_template('index.html')



# Ruta para predicción
@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.json
        if "features" not in data:
            return jsonify({"error": "No features provided"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)

        prediction = output.numpy().tolist()
        prediction_value = float(prediction[0][0])

        if 0 <= prediction_value < 0.5:
            resultado = "No diabético"
        elif 0.5 <= prediction_value <= 1:
            resultado = "Diabético"
        else:
            resultado = "❓ Resultado desconocido."

        return jsonify({"prediction": prediction_value, "resultado": resultado})

    except Exception as e:
        # Esto devolverá un JSON con el error para evitar respuesta HTML
        return jsonify({"error": str(e)}), 500



# Para ejecución en Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

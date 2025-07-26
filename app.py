from flask import Flask, request, jsonify
import torch
import pandas as pd
import joblib
from anfis_model import ANFIS

app = Flask(__name__)

# Columnas esperadas, igual que en entrenamiento
expected_columns = [
    'SEXO', 'EDAD', 'IMC', 'NEUTROFILOS', 'HEMOGLOBINA',
    'ERITROCITOS', 'HEMATOCRITO', 'MCH', 'MCV', 'EOSINOFILOS',
    'RDW-CV', 'LINFOCITOS', 'MONOCITOS'
]

# Cargar scaler entrenado
scaler = joblib.load("scaler.pkl")

# Crear modelo e cargar pesos
model = ANFIS(n_inputs=13, n_rules=300)
model.load_state_dict(torch.load("entrenamiento/anfis_state_dict_300.pth", map_location=torch.device('cpu')))
model.eval()

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.json
        if not data or "features" not in data:
            return jsonify({"error": "No features provided"}), 400
        
        input_data = data["features"]
        
        # Convertir a DataFrame con columnas en orden esperado
        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_columns]

        # Escalar
        input_scaled = scaler.transform(input_df)
        
        # Convertir a tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Inferencia
        with torch.no_grad():
            output = model(input_tensor)
            prediction = float(output.item())
            resultado = "Diabético" if prediction >= 0.5 else "No diabético"

        return jsonify({"prediction": prediction, "resultado": resultado})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
    app.run(host="0.0.0.0", port=port)

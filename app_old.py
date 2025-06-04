from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np
from anfis_model import ANFIS  

app = Flask(__name__)

# Cargar modelo y scaler
model = torch.load("anfis_modelo_27.pth", map_location=torch.device('cpu'))
model.eval()
scaler = joblib.load("scaler27_render.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = [float(request.form[key]) for key in request.form]
        datos_entrada = np.array([datos])
        datos_entrada = scaler.transform(datos_entrada)

        input_tensor = torch.tensor(datos_entrada, dtype=torch.float32)
        output = model(input_tensor)
        probabilidad = torch.sigmoid(output).item() * 100

        return jsonify({"probabilidad": round(probabilidad, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

import torch
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Clase del modelo ANFIS (debe coincidir con la usada en el entrenamiento)
class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super(ANFIS, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.pesos = nn.Parameter(torch.randn(n_rules, n_inputs))  
        self.membresia = nn.Parameter(torch.randn(n_rules, 1))  

    def forward(self, x):
        reglas = torch.matmul(x, self.pesos.T)  
        reglas_activadas = torch.softmax(reglas, dim=1)  
        membresia_activada = torch.matmul(reglas_activadas, self.membresia)  
        return membresia_activada  

# Cargar el modelo ANFIS y el scaler
torch.serialization.add_safe_globals([ANFIS])
model = torch.load("anfis_modelo_27.pth", weights_only=False)
model.eval()
scaler = joblib.load("scaler27.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener datos del formulario
        datos = [float(request.form[key]) for key in request.form]
        datos_entrada = np.array([datos])  # Convertir a numpy array
        datos_entrada = scaler.transform(datos_entrada)  # Escalar datos

        # Convertir a tensor y predecir
        input_tensor = torch.tensor(datos_entrada, dtype=torch.float32)
        output = model(input_tensor)
        probabilidad = torch.sigmoid(output).item() * 100  # Convertir a porcentaje

        return jsonify({"probabilidad": round(probabilidad, 2)})  # Responder en JSON
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
from flask import Flask, request, jsonify
import torch
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el scaler
model = torch.load('model/modelo_entrenado.pth', map_location=torch.device('cpu'))
model.eval()
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return 'App T2DM funcionando'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    with torch.no_grad():
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        prediction = model(input_tensor)
        prediction = torch.sigmoid(prediction).item()
    return jsonify({'prediction': prediction})

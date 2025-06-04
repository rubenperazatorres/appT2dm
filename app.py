from flask import Flask, request, jsonify, render_template
import torch
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = torch.load('anfis_modelo_27.pth', map_location=torch.device('cpu'))
model.eval()
scaler = joblib.load('scaler27.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Usa tus plantillas

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

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar modelo y scaler
# model = torch.load("anfis_modelo_27.pth", map_location=torch.device('cpu'))
# model.eval()
# scaler = joblib.load("scaler27_render.pkl")


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

# Carga el scaler al iniciar la app
scaler = load_scaler("scaler_params.json")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    
    data = request.json  # Suponemos que recibes un JSON con datos en lista
    # Ejemplo: data = {"features": [val1, val2, val3, ...]}
    features = np.array(data["features"]).reshape(1, -1)    
    
    return jsonify({"scaled_features": features_scaled.tolist()})

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)
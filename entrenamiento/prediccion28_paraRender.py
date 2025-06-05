# Modelo ANFIS.28 para Render.com

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

##############################
# Parámetros y configuración #
##############################
entradas = 13
reglas = 300
epocas = 1000

tasa_inicial = 0.01
tasa_reduccion = 350
factor_reduccion = 0.45

print("Sistema ANFIS - Predicción de Diabetes Tipo 2")

##############################
# Funciones auxiliares
##############################

def cargar_dataset(ruta):
    df = pd.read_csv(ruta)
    return df

def aplicar_smote(df):
    X = df.drop(columns=['P_DIABETICO'])
    y = df['P_DIABETICO']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['P_DIABETICO'] = y_resampled
    return df_resampled

##############################
# Modelo ANFIS con PyTorch
##############################

class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super(ANFIS, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Pesos inicializados con Xavier
        self.pesos = nn.Parameter(torch.empty(n_rules, n_inputs))
        nn.init.xavier_uniform_(self.pesos)

        # Membresía inicializada con pequeña varianza
        self.membresia = nn.Parameter(torch.randn(n_rules, 1) * 0.1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        reglas = torch.matmul(x, self.pesos.T)
        reglas_activadas = F.softmax(reglas, dim=1)
        membresia_activada = torch.matmul(reglas_activadas, self.membresia)
        return self.dropout(membresia_activada)  # SIN sigmoid aquí

##############################
# Carga y preparación de datos
##############################

ruta = "dataSet.csv"
df_original = cargar_dataset(ruta)

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
df_original = pd.DataFrame(imputer.fit_transform(df_original), columns=df_original.columns)

# Balancear con SMOTE
df_smote = aplicar_smote(df_original)
print("Dataset balanceado con SMOTE. Tamaño:", df_smote.shape)
df = df_smote

# Separar X e y
X = df.drop("P_DIABETICO", axis=1)
y = df["P_DIABETICO"]

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##############################
# Preparar tensores
##############################
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

##############################
# Crear modelo, optimizador y pérdida
##############################
model = ANFIS(n_inputs=entradas, n_rules=reglas)
optimizer = optim.Adam(model.parameters(), lr=tasa_inicial)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tasa_reduccion, gamma=factor_reduccion)
loss_fn = nn.BCEWithLogitsLoss()  # IMPORTANTE: usar esta función

##############################
# Entrenamiento
##############################
train_losses = []

for epoch in range(epocas):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        y_pred_prob = torch.sigmoid(y_pred)
        y_pred_binary = (y_pred_prob >= 0.5).float()
        accuracy = (y_pred_binary.eq(y_train_tensor).sum().item()) / y_train_tensor.shape[0] * 100

    if epoch % 100 == 0 or epoch == epocas - 1:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

    train_losses.append(loss.item())

##############################
# Graficar pérdida
##############################
plt.plot(range(epocas), train_losses, label='Pérdida de Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

##############################
# Evaluación en conjunto de prueba
##############################
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_test_prob = torch.sigmoid(y_pred_test)
    y_pred_test_binary = (y_pred_test_prob >= 0.5).float()

print("Accuracy en prueba:", accuracy_score(y_test_tensor, y_pred_test_binary))
print("Classification Report:")
print(classification_report(y_test_tensor, y_pred_test_binary))
roc_auc = roc_auc_score(y_test_tensor, y_pred_test_prob)
print(f"AUC-ROC Score: {roc_auc:.4f}")

cm = confusion_matrix(y_test_tensor.numpy(), y_pred_test_binary.numpy())
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

##############################
# Predicción para un nuevo paciente
##############################
nuevo_paciente = pd.DataFrame({
    'SEXO': [0],
    'EDAD': [51],
    'IMC': [29],
    'NEUTROFILOS': [60],
    'HEMOGLOBINA': [13.5],
    'ERITROCITOS': [5.0],
    'HEMATOCRITO': [42.5],
    'MCH': [32.1],
    'MCV': [85],
    'EOSINOFILOS': [2],
    'RDW-CV': [13.5],
    'LINFOCITOS': [30],
    'MONOCITOS': [8]
})

nuevo_paciente_scaled = scaler.transform(nuevo_paciente)
nuevo_paciente_tensor = torch.tensor(nuevo_paciente_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred_nuevo = model(nuevo_paciente_tensor)
    pred_prob = torch.sigmoid(pred_nuevo).item()

print(f"Predicción del riesgo de diabetes para el nuevo paciente: {pred_prob:.4f}")

##############################
# Guardar modelo y scaler
##############################
#torch.save(model.state_dict(), "anfis_state_dict_27.pth")  # torch.save(model, "file.pth") guarda modelo completo y torch.save(model.state_dict(), "file.pth") guarda solo pesos
torch.save(model.state_dict(), "anfis_state_dict_300.pth")
print("Modelo ANFIS guardado exitosamente como: anfis_state_dict_300.pth")

joblib.dump(scaler, "scaler.pkl")
print("Scaler guardado exitosamente como: scaler.pkl")


##############################
# Guardar scaler_params.json
##############################
import json

scaler_params = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
    "var_": scaler.var_.tolist(),
    "n_features_in_": scaler.n_features_in_,
    "feature_names_in_": scaler.feature_names_in_.tolist()
}

with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f)

print("Scaler params guardados exitosamente como: scaler_params.json")

# Modelo ANFIS ver 24
# En el modelo ANFIS las funciones de membresía se adaptan automáticamente durante el proceso de entrenamiento.
# En esta prueba se incorpora la reducción progresivamente de la tasa de aprendizaje (LR) para ajustar los pesos de manera más fina.
# Se prbaron 3 métodos de balanceo: Undersampling (sub-muestreo), Oversampling (sobre-muestreo) y SMOTE (Synthetic Minority Over-sampling Technique). 
# para balancear las clases diabéticos con la no diabéticos. Se elige el último porque dio mejores resultados.
# Esto porque en la vesión anterior se ve que hay problemas para predecir los diabéticos dado este  desbalanceo 
# (No diabéticos (0): 25,155 registros contra Diabéticos (1): 5,930 registros)
# Se corrige el dataset original para variable sexo que estaba configurada en el dataSet.csv como M(mujer) H(hombre) lo he cambiado manualmente a 1(mujer) y 0(hombre).
#
# Se realizan modificaciones para evitar un sobre ajuste de pesos iniciales en el  entrenamiento ya que se detectó que la variable RDW-CV influye en el resultado de manera muy abrupta, cuando es <14 o >14
# Inicialización Mejorada de Pesos: Cambiar la inicialización de los pesos de las reglas a valores más pequeños y controlados para evitar que una variable tenga un impacto excesivo.
# Normalización de Pesos: Se uso nn.init.xavier_uniform_ para una mejor distribución de los pesos iniciales.
# Cambio en la Función de Pérdida: Utilizar nn.BCELoss() en lugar de nn.BCEWithLogitsLoss() para estabilizar la retropropagación.
# Pesos Diferenciales para Variables: Ajustar manualmente los pesos iniciales de las reglas, asignando menor peso a variables menos relevantes.
# Regulación con Dropout: Agregar nn.Dropout() para evitar sobreajuste.


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import resample #balanceo de clases
from imblearn.over_sampling import SMOTE #Librería SMOTE (Synthetic Minority Over-sampling Technique)
from sklearn.impute import SimpleImputer # Para rellenar valores faltantes con la media
import joblib #guardar modelos

##################################
# Variables para el modelo ANFIS #
##################################
entradas = 13
reglas = 200
epocas = 1000

#Optimizador Adam para reducir la tasa de aprendizaje
tasa_inicial = 0.01
tasa_reduccion = 350
factor_reduccion = 0.45 

print("Sistema ANFIS - Predicción de Diabetes Tipo 2")

def cargar_dataset(ruta):
    df = pd.read_csv(ruta)
    return df

def aplicar_smote(df):
    X = df.drop(columns=['P_DIABETICO'])  # Excluir P_DIABETICO que no es una entrada
    y = df['P_DIABETICO']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Crear el DataFrame balanceado
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['P_DIABETICO'] = y_resampled  # Restaurar la variable objetivo

    return df_resampled

def guardar_dataset(df, nombre_archivo="dataSet_balanceado.csv"):
    #df.to_csv(nombre_archivo, index=False) # Ya no lo guardamos de nuevo porque se realizó anteriormente
    print(f"Dataset balanceado guardado como {nombre_archivo}")

# Cargar el dataset
ruta = "dataSet.csv"
df_original = cargar_dataset(ruta)

# Verificar valores NaN antes de cualquier transformación
#print("Valores NaN antes de limpieza:")
#print(df_original.isnull().sum())

# Imputar valores de SEXO con la moda
#imputer_sexo = SimpleImputer(strategy="most_frequent")
#df_original['SEXO'] = imputer_sexo.fit_transform(df_original[['SEXO']]).ravel()

# Convertir M/H a 0/1 después de asegurarnos que no hay NaN
#df_original['SEXO'] = df_original['SEXO'].map({'M': 0, 'H': 1})

# Asegurar que no hay valores NaN en las demás columnas
imputer = SimpleImputer(strategy='mean')
df_original = pd.DataFrame(imputer.fit_transform(df_original), columns=df_original.columns)

#print("Suma de valores nulos:")
#print(df_original.isnull().sum())

# Verificar valores NaN después de la imputación
#print("Valores NaN después de limpieza:")
#print(df_original.isnull().sum())


#METODOS DE BALANCEO

# Método: SMOTE
df_smote = aplicar_smote(df_original)
guardar_dataset(df_smote, "dataSet_smote.csv")
df = df_smote

# Codificar la columna SEXO (categórica) usando LabelEncoder
# label_encoder = LabelEncoder()
# df['SEXO'] = label_encoder.fit_transform(df['SEXO'])

# Separar las características y la etiqueta
X = df.drop("P_DIABETICO", axis=1)
y = df["P_DIABETICO"]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo ANFIS con PyTorch
class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super(ANFIS, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Pesos de las reglas anteriores
        self.pesos = nn.Parameter(torch.randn(n_rules, n_inputs))  
        self.membresia = nn.Parameter(torch.randn(n_rules, 1))  

### Cambios
        
        # Pesos inicializados con Xavier para mejor estabilidad
        self.pesos = nn.Parameter(torch.empty(n_rules, n_inputs))
        nn.init.xavier_uniform_(self.pesos)  
                
        # Membresía inicializada de manera más estable
        self.membresia = nn.Parameter(torch.randn(n_rules, 1) * 0.1)

        # Dropout para evitar sobreajuste
        self.dropout = nn.Dropout(p=0.2)        
        
######## fin cambios

    def forward(self, x):
        reglas = torch.matmul(x, self.pesos.T)  
        reglas_activadas = F.softmax(reglas, dim=1)  
        membresia_activada = torch.matmul(reglas_activadas, self.membresia)  
        return self.dropout(membresia_activada)        
        # return membresia_activada  

# Inicializar modelo
model = ANFIS(n_inputs=13, n_rules=reglas)

# Definir optimizador y pérdida
lr_inicial = tasa_inicial #Tasa inicial
optimizer = optim.Adam(model.parameters(), lr=lr_inicial)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tasa_reduccion, gamma=factor_reduccion)  # Reducir la tasa de aprendizaje
loss_fn = nn.BCEWithLogitsLoss()

# Convertir datos a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_losses = []

# Entrenamiento
for epoch in range(epocas):
    model.train()
    optimizer.zero_grad()
    
    # Predicción y cálculo de pérdida
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    # Retropropagación
    loss.backward()
    optimizer.step()
    scheduler.step()  # Actualizar LR según el scheduler

    # Accuracy en entrenamiento
    with torch.no_grad():
        y_pred_binary = (torch.sigmoid(y_pred) >= 0.5).float()
        accuracy = (y_pred_binary.eq(y_train_tensor).sum().item()) / y_train_tensor.shape[0] * 100 

    # Reporte cada 10 épocas
    if epoch % 100 == 0 or epoch == epocas - 1:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

    train_losses.append(loss.item())


# Graficar las pérdidas de entrenamiento
plt.plot(range(epocas), train_losses, label='Pérdida de Entrenamiento')  
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


# Evaluación en conjunto de prueba
model.eval()  
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_test_prob = torch.sigmoid(y_pred_test)
    y_pred_test_binary = (y_pred_test_prob >= 0.5).float()

# Evaluar rendimiento
print("Accuracy en prueba:", accuracy_score(y_test_tensor, y_pred_test_binary))
print("Classification Report:")
print(classification_report(y_test_tensor, y_pred_test_binary))
roc_auc = roc_auc_score(y_test_tensor, y_pred_test)
print(f"AUC-ROC Score: {roc_auc:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test_tensor.numpy(), y_pred_test_binary.numpy())
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# Predicción para un nuevo paciente
nuevo_paciente = pd.DataFrame({
    'SEXO': [0],  # 0 = Masculino  1 = Femenino 
    'EDAD': [51],
    'IMC': [29],  # Altura=1.79 Peso=91kg
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

# Preprocesar los datos del nuevo paciente
nuevo_paciente_scaled = scaler.transform(nuevo_paciente)

# Convertir a tensor y hacer la predicción
nuevo_paciente_tensor = torch.tensor(nuevo_paciente_scaled, dtype=torch.float32)
prediccion_red_neuronal = model(nuevo_paciente_tensor)
prediccion_prob = torch.sigmoid(prediccion_red_neuronal).item()
60  
# Mostrar los resultados
print(f"Predicción del riesgo de diabetes para el nuevo paciente: {prediccion_prob:.2f}")

# Guardar el modelo ANFIS entrenado
#joblib.dump(model, "anfis_modelo.pkl")

# Guardar el modelo ANFIS con PyTorch
#torch.save(model.state_dict(), "anfis_modelo.pth")
#print("Modelo ANFIS guardado exitosamente como: anfis_modelo.pth")

torch.save(model, "anfis_modelo_completo.pth")
print("Modelo ANFIS guardado exitosamente como: anfis_modelo_completo.pth")



# Guardar el scaler
#joblib.dump(scaler, "scaler.pkl")

# Guardar el scaler con joblib
joblib.dump(scaler, "scaler.pkl")
print("Scaler guardado exitosamente como: scaler.pkl")
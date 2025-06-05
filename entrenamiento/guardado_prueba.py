import pandas as pd
import os

def guardar_dataset(df, nombre_archivo="dataSet_DM2_G1G2_public_balanceado.csv"):
    ruta_absoluta = os.path.abspath(nombre_archivo)
    df.to_csv(ruta_absoluta, index=False)
    print(f"Dataset balanceado guardado como {ruta_absoluta}")

# Crear un DataFrame de prueba
datos = {
    "SEXO": [0, 1],
    "EDAD": [30, 45],
    "IMC": [22.5, 27.3],
    "P_DIABETICO": [0, 1]
}

df_prueba = pd.DataFrame(datos)

# Guardar el DataFrame usando la función
guardar_dataset(df_prueba)

# Verificar que el archivo existe
ruta = os.path.abspath("dataSet_DM2_G1G2_public_balanceado.csv")
if os.path.exists(ruta):
    print("El archivo se guardó correctamente.")
else:
    print("Error: el archivo no se guardó.")

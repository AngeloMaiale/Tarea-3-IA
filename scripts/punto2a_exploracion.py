"""
Punto 2 (parte A): Análisis exploratorio y limpieza
===================================================
- Muestra el tamaño del dataset, los tipos de datos y estadísticas básicas.
- Calcula la proporción de ejemplos de cada clase (lo pide la tarea).
- Limpia los datos (ver limpiar_datos en datos.py).
- Grafica histogramas por clase y la matriz de correlación.

Ejecutar desde la carpeta /scripts:   python punto2a_exploracion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from datos import FEATURES, ETIQUETA, cargar_datos, limpiar_datos

CARPETA_FIG = Path(__file__).resolve().parent.parent / "figuras"
CARPETA_FIG.mkdir(exist_ok=True)

# 1) Cargar los datos crudos
df = cargar_datos()
print("Tamaño (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nPrimeras filas:\n", df.head())

# 2) Estadísticas descriptivas: media, desviación, mínimo, máximo, cuartiles.
#    Sirve para ver las ESCALAS: Area está en miles y Eccentricity entre 0 y 1
#    -> por eso habrá que estandarizar antes del SVC con kernel gaussiano.
print("\nEstadísticas descriptivas:\n", df.describe().T.round(3))

# 3) Proporción de cada clase
print("\nCantidad de ejemplos por clase:\n", df[ETIQUETA].value_counts())
print("\nProporción de cada clase:\n", df[ETIQUETA].value_counts(normalize=True).round(3))
# Resultado: ~57 % Osmancik y ~43 % Cammeo -> clases algo desbalanceadas pero no extremas.

# 4) Limpieza
print("\n--- Limpieza ---")
df = limpiar_datos(df)

# 5) Histogramas de cada feature separados por clase:
#    si las distribuciones de las dos clases se separan, esa feature ayuda a clasificar.
fig, ejes = plt.subplots(2, 4, figsize=(16, 7))
for eje, feature in zip(ejes.ravel(), FEATURES):
    sns.histplot(data=df, x=feature, hue=ETIQUETA, ax=eje, bins=30, element="step")
ejes.ravel()[-1].axis("off")
fig.suptitle("Distribución de cada feature por clase")
fig.tight_layout()
fig.savefig(CARPETA_FIG / "histogramas.png", dpi=110)

# 6) Matriz de correlación: Area, Perimeter y Convex_Area están casi perfectamente
#    correlacionadas (miden casi lo mismo: el tamaño del grano).
fig, eje = plt.subplots(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=eje)
eje.set_title("Correlación entre features")
fig.tight_layout()
fig.savefig(CARPETA_FIG / "correlacion.png", dpi=110)

print(f"\nFiguras guardadas en: {CARPETA_FIG}")
plt.show()

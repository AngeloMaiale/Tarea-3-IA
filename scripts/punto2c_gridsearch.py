"""
Punto 2 (parte C): Pipeline + GridSearchCV + mapa de calor + score de prueba
============================================================================
1) Pipeline = StandardScaler -> SVC(kernel="rbf"), usando las 7 features.
   ¿Por qué el scaler va DENTRO del Pipeline? Porque GridSearchCV hace validación
   cruzada: en cada fold el scaler se ajusta SOLO con los datos de entrenamiento
   de ese fold. Si se estandarizara antes con todos los datos, la media y la
   desviación incluirían información del fold de validación (fuga de datos) y el
   score de validación sería demasiado optimista.

2) GridSearchCV prueba todas las combinaciones de C y gamma (escala logarítmica)
   con validación cruzada de 5 folds y se queda con la de mejor exactitud media.
   - Nº de modelos entrenados = (nº de C) x (nº de gamma) x (nº de folds) + 1
     (el +1 es el reentrenamiento final del mejor modelo con todo el entrenamiento).
   - best_score_ = exactitud MEDIA de validación cruzada del mejor modelo (calculada
     con datos de entrenamiento). Es distinta del score de prueba, que se calcula con
     datos que NUNCA se usaron para elegir hiperparámetros.

3) Mapa de calor del score de validación cruzada para cada (C, gamma).

4) Score del mejor modelo en el conjunto de PRUEBA: una sola vez, al final.

Ejecutar desde la carpeta /scripts:   python punto2c_gridsearch.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datos import SEMILLA, obtener_datos

CARPETA_FIG = Path(__file__).resolve().parent.parent / "figuras"
CARPETA_FIG.mkdir(exist_ok=True)

# 1) Datos: misma separación de prueba que en los otros scripts.
X_train, X_test, y_train, y_test = obtener_datos()

# 2) Pipeline: los nombres "escalador" y "svc" se usan en la grilla como "svc__C".
pipeline = Pipeline([
    ("escalador", StandardScaler()),
    ("svc", SVC(kernel="rbf")),
])

# 3) Grilla en escala logarítmica: 10^-2 ... 10^4 para C y 10^-4 ... 10^1 para gamma.
#    Escala logarítmica porque el efecto de C y gamma depende de su ORDEN DE MAGNITUD.
#    Si el mejor valor queda en el borde de la grilla, conviene ampliarla.
grilla = {
    "svc__C": np.logspace(-2, 4, 7),       # [0.01, 0.1, 1, 10, 100, 1000, 10000]
    "svc__gamma": np.logspace(-4, 1, 6),   # [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

# 4) Validación cruzada estratificada de 5 folds (mantiene la proporción de clases).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
busqueda = GridSearchCV(pipeline, grilla, cv=cv, scoring="accuracy", n_jobs=-1)
busqueda.fit(X_train, y_train)

n_comb = len(grilla["svc__C"]) * len(grilla["svc__gamma"])
print(f"Combinaciones probadas: {n_comb}  ->  modelos entrenados: {n_comb} x 5 folds + 1 = {n_comb * 5 + 1}")
print("Mejores hiperparámetros:", busqueda.best_params_)
print(f"best_score_ (exactitud media de validación cruzada): {busqueda.best_score_:.4f}")

# 5) Mapa de calor: filas = C, columnas = gamma, valor = exactitud media de validación.
res = pd.DataFrame(busqueda.cv_results_)
tabla = res.pivot_table(index="param_svc__C", columns="param_svc__gamma", values="mean_test_score")
fig, eje = plt.subplots(figsize=(8, 6))
sns.heatmap(tabla, annot=True, fmt=".3f", cmap="viridis", ax=eje,
            xticklabels=[f"{g:g}" for g in tabla.columns],
            yticklabels=[f"{c:g}" for c in tabla.index])
eje.set_xlabel("gamma")
eje.set_ylabel("C")
eje.set_title("Exactitud media de validación cruzada (5 folds)")
fig.tight_layout()
fig.savefig(CARPETA_FIG / "mapa_calor_gridsearch.png", dpi=110)

# 6) Score de PRUEBA del mejor modelo: se calcula UNA sola vez, al final.
#    busqueda.score usa best_estimator_ (ya reentrenado con todo X_train).
score_prueba = busqueda.score(X_test, y_test)
print(f"\nExactitud en el conjunto de PRUEBA: {score_prueba:.4f}")
print(f"Figura guardada en: {CARPETA_FIG / 'mapa_calor_gridsearch.png'}")
plt.show()

"""
Punto 2 (parte B): SVC con kernel gaussiano (RBF) usando SOLO 2 features
========================================================================
Objetivo: VER cómo cambia la frontera de decisión al mover gamma y C.

Kernel gaussiano:  K(x, z) = exp(-gamma * ||x - z||^2),  con  gamma = 1 / (2 sigma^2)
  - gamma grande (sigma chico): cada punto influye solo MUY cerca -> frontera muy
    curva/irregular que se adapta a cada punto -> riesgo de OVERFITTING.
  - gamma chico (sigma grande): cada punto influye lejos -> frontera suave, casi recta.

C (SVM de margen suave): min  1/2 ||w||^2 + C * sum(xi_i)
  - xi_i = holgura: cuánto viola el ejemplo i el margen.
  - C chico: violar el margen sale barato -> margen ANCHO, muchos vectores de soporte.
  - C grande: violar el margen sale caro -> margen ANGOSTO, menos vectores de soporte,
    el modelo intenta clasificar bien todo el entrenamiento (riesgo de overfitting).

Ejecutar desde la carpeta /scripts:   python punto2b_svc_dos_features.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datos import SEMILLA, obtener_datos

CARPETA_FIG = Path(__file__).resolve().parent.parent / "figuras"
CARPETA_FIG.mkdir(exist_ok=True)

# Las 2 features elegidas: largo y ancho del grano (fáciles de interpretar y,
# según los histogramas, las dos separan bien las clases).
F1, F2 = "Major_Axis_Length", "Minor_Axis_Length"


def graficar_frontera(modelo, X, y, titulo, eje):
    """
    Dibuja:
      - las regiones de cada clase (color de fondo),
      - la frontera de decisión (línea continua, decision_function = 0),
      - los márgenes (líneas punteadas, decision_function = -1 y +1),
      - los puntos de entrenamiento,
      - los vectores de soporte (círculos negros sin relleno).
    X debe estar estandarizado (2 columnas).
    """
    # Malla de puntos que cubre todo el gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    malla = np.c_[xx.ravel(), yy.ravel()]

    # decision_function > 0 -> clase modelo.classes_[1] ; < 0 -> clase modelo.classes_[0]
    Z = modelo.decision_function(malla).reshape(xx.shape)
    eje.contourf(xx, yy, Z > 0, alpha=0.25, cmap="coolwarm")
    eje.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)

    # Puntos de entrenamiento coloreados por clase
    colores = (y == modelo.classes_[1]).astype(int)
    eje.scatter(X[:, 0], X[:, 1], c=colores, cmap="coolwarm", s=8, alpha=0.6)

    # Vectores de soporte
    sv = modelo.support_vectors_
    eje.scatter(sv[:, 0], sv[:, 1], s=30, facecolors="none", edgecolors="k", linewidths=0.6)

    eje.set_title(titulo, fontsize=10)
    eje.set_xlabel(F1 + " (estandarizada)")
    eje.set_ylabel(F2 + " (estandarizada)")


if __name__ == "__main__":
    # 1) Mismos datos y misma separación de prueba que el resto de la tarea.
    #    El conjunto de prueba NO se usa en este script.
    X_train, X_test, y_train, y_test = obtener_datos()

    # 2) Del conjunto de entrenamiento separamos un conjunto de VALIDACIÓN (25 %)
    #    para comparar los modelos sin tocar la prueba.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train[[F1, F2]], y_train, test_size=0.25, stratify=y_train, random_state=SEMILLA
    )

    # 3) Estandarizar: media 0 y desviación 1. El scaler se AJUSTA solo con X_tr
    #    y luego se aplica a validación (así no se filtra información de validación).
    escalador = StandardScaler().fit(X_tr)
    X_tr_s = escalador.transform(X_tr)
    X_val_s = escalador.transform(X_val)
    y_tr, y_val = y_tr.to_numpy(), y_val.to_numpy()

    # 4) Configuraciones: 3 gammas con C fijo y 3 valores de C con gamma fijo.
    configuraciones = [
        {"gamma": 0.01, "C": 1},
        {"gamma": 1,    "C": 1},
        {"gamma": 100,  "C": 1},
        {"gamma": 1,    "C": 0.01},
        {"gamma": 1,    "C": 1},
        {"gamma": 1,    "C": 100},
    ]

    fig, ejes = plt.subplots(2, 3, figsize=(17, 10))
    resultados = []
    for cfg, eje in zip(configuraciones, ejes.ravel()):
        modelo = SVC(kernel="rbf", gamma=cfg["gamma"], C=cfg["C"])
        modelo.fit(X_tr_s, y_tr)

        n_sv = int(modelo.n_support_.sum())          # nº total de vectores de soporte
        acc_tr = modelo.score(X_tr_s, y_tr)          # exactitud de entrenamiento
        acc_val = modelo.score(X_val_s, y_val)       # exactitud de validación
        resultados.append({**cfg, "vectores_soporte": n_sv,
                           "exactitud_entrenamiento": round(acc_tr, 4),
                           "exactitud_validacion": round(acc_val, 4)})

        titulo = (f"gamma={cfg['gamma']}, C={cfg['C']}\n"
                  f"VS={n_sv} | train={acc_tr:.3f} | val={acc_val:.3f}")
        graficar_frontera(modelo, X_tr_s, y_tr, titulo, eje)

    ejes[0, 0].annotate("Fila 1: varía gamma (C = 1)", (0, 1.15), xycoords="axes fraction", fontsize=12, weight="bold")
    ejes[1, 0].annotate("Fila 2: varía C (gamma = 1)", (0, 1.15), xycoords="axes fraction", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "fronteras_svc.png", dpi=110)

    tabla = pd.DataFrame(resultados)
    print(tabla.to_string(index=False))
    print(f"\nClase positiva (decision_function > 0): {modelo.classes_[1]}  (rojo)")
    print(f"Figura guardada en: {CARPETA_FIG / 'fronteras_svc.png'}")
    plt.show()

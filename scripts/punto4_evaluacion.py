"""
Punto 4: Evaluación y comparación
=================================
1) Matriz de confusión en PRUEBA de los tres modelos (SVC, árbol, random forest)
   + precisión y sensibilidad (recall) de cada clase.
2) Con class_weight="balanced": mismas matrices y métricas, para comparar con y sin.
3) Tablas resumen para escribir las explicaciones (exactitud vs otras métricas,
   SVC vs random forest, efecto de class_weight).

Los modelos se buscan con las MISMAS grillas y la MISMA validación cruzada de los
puntos 2 y 3 (se reutilizan las funciones de punto3_arbol_rf.py).

Ejecutar desde la carpeta /scripts:   python punto4_evaluacion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datos import obtener_datos, SEMILLA
from punto3_arbol_rf import CV, buscar_arbol, buscar_random_forest

CARPETA_FIG = Path(__file__).resolve().parent.parent / "figuras"
CARPETA_FIG.mkdir(exist_ok=True)


def buscar_svc(X_train, y_train, class_weight=None):
    """Misma búsqueda del Punto 2 (pipeline + grilla logarítmica), con class_weight opcional."""
    pipeline = Pipeline([
        ("escalador", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight=class_weight)),
    ])
    grilla = {"svc__C": np.logspace(-2, 4, 7), "svc__gamma": np.logspace(-4, 1, 6)}
    busqueda = GridSearchCV(pipeline, grilla, cv=CV, scoring="accuracy", n_jobs=-1)
    return busqueda.fit(X_train, y_train)


def buscar_arbol_cw(X_train, y_train, class_weight):
    """Árbol con class_weight: se vuelve a elegir max_depth con la misma grilla."""
    busqueda = GridSearchCV(
        DecisionTreeClassifier(random_state=SEMILLA, class_weight=class_weight),
        {"max_depth": list(range(1, 21))}, cv=CV, scoring="accuracy", n_jobs=-1)
    return busqueda.fit(X_train, y_train)


def metricas_por_clase(modelo, X_test, y_test, nombre):
    """Precisión y sensibilidad (recall) de cada clase, en filas de una tabla."""
    y_pred = modelo.predict(X_test)
    clases = list(modelo.classes_)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, labels=clases)
    filas = []
    for c, p, r in zip(clases, prec, rec):
        filas.append({"modelo": nombre, "clase": c, "precisión": round(p, 4), "sensibilidad": round(r, 4)})
    return filas, float((y_pred == y_test).mean())


def graficar_matrices(modelos, X_test, y_test, titulo, archivo):
    """Una matriz de confusión por modelo (conteos), en una sola figura."""
    fig, ejes = plt.subplots(1, len(modelos), figsize=(5.2 * len(modelos), 4.6))
    for eje, (nombre, modelo) in zip(np.atleast_1d(ejes), modelos.items()):
        ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, ax=eje, cmap="Blues", colorbar=False)
        eje.set_title(nombre)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / archivo, dpi=110)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = obtener_datos()

    # ---------------- Modelos SIN class_weight ----------------
    svc = buscar_svc(X_train, y_train).best_estimator_
    arbol = buscar_arbol(X_train, y_train).best_estimator_
    rf = buscar_random_forest(X_train, y_train).best_estimator_
    modelos = {"SVC": svc, "Árbol de decisión": arbol, "Random forest": rf}

    # 4a) Matrices de confusión + precisión y sensibilidad por clase
    graficar_matrices(modelos, X_test, y_test,
                      "Matrices de confusión en PRUEBA (filas = clase real, columnas = predicha)",
                      "matrices_confusion.png")
    filas, exactitudes = [], {}
    for nombre, m in modelos.items():
        f, acc = metricas_por_clase(m, X_test, y_test, nombre)
        filas += f
        exactitudes[nombre] = round(acc, 4)
    print("=== Sin class_weight: precisión y sensibilidad por clase (PRUEBA) ===")
    print(pd.DataFrame(filas).to_string(index=False))
    print("Exactitud:", exactitudes)
    print("\nReporte completo del SVC:\n", classification_report(y_test, svc.predict(X_test), digits=4))

    # 4d) Con class_weight="balanced"
    svc_cw = buscar_svc(X_train, y_train, class_weight="balanced").best_estimator_
    arbol_cw = buscar_arbol_cw(X_train, y_train, "balanced").best_estimator_
    rf_cw = buscar_random_forest(X_train, y_train, class_weight="balanced").best_estimator_
    modelos_cw = {"SVC (balanced)": svc_cw, "Árbol (balanced)": arbol_cw, "Random forest (balanced)": rf_cw}

    graficar_matrices(modelos_cw, X_test, y_test,
                      'Matrices de confusión en PRUEBA con class_weight="balanced"',
                      "matrices_confusion_class_weight.png")
    filas_cw, exactitudes_cw = [], {}
    for nombre, m in modelos_cw.items():
        f, acc = metricas_por_clase(m, X_test, y_test, nombre)
        filas_cw += f
        exactitudes_cw[nombre] = round(acc, 4)
    print('\n=== Con class_weight="balanced": precisión y sensibilidad por clase (PRUEBA) ===')
    print(pd.DataFrame(filas_cw).to_string(index=False))
    print("Exactitud:", exactitudes_cw)

    # Comparación lado a lado (sensibilidad de cada clase, con y sin class_weight)
    comp = pd.DataFrame(filas + filas_cw)
    print("\n=== Resumen con y sin class_weight ===")
    print(comp.pivot_table(index="modelo", columns="clase", values=["precisión", "sensibilidad"]).round(4))

    print(f"\nFiguras guardadas en: {CARPETA_FIG}")
    plt.show()

"""
Punto 3: Árbol de decisión y Random Forest
==========================================
Mismos datos y MISMA separación de prueba que el Punto 2 (se obtienen con datos.py).

1) DecisionTreeClassifier: max_depth elegido con GridSearchCV (+ gráfico de la
   exactitud de entrenamiento vs validación según la profundidad).
2) Árbol de profundidad 3 dibujado con nombres de features y de clases.
3) RandomForestClassifier: n_estimators, max_depth y max_features con GridSearchCV.
4) Score de PRUEBA del árbol y del random forest (una sola vez, al final).

NOTA: los árboles NO necesitan estandarizar las features (cada corte compara una
sola feature con un umbral, y eso no depende de la escala). Por eso aquí no hay
StandardScaler, a diferencia del SVC.

Ejecutar desde la carpeta /scripts:   python punto3_arbol_rf.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

from datos import FEATURES, SEMILLA, obtener_datos

CARPETA_FIG = Path(__file__).resolve().parent.parent / "figuras"
CARPETA_FIG.mkdir(exist_ok=True)

# Misma validación cruzada que en el Punto 2 (5 folds estratificados).
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def buscar_arbol(X_train, y_train):
    """GridSearchCV sobre max_depth de un árbol de decisión."""
    grilla = {"max_depth": list(range(1, 21))}
    busqueda = GridSearchCV(
        DecisionTreeClassifier(random_state=SEMILLA),
        grilla, cv=CV, scoring="accuracy", n_jobs=-1,
        return_train_score=True,   # para comparar entrenamiento vs validación
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def buscar_random_forest(X_train, y_train, class_weight=None):
    """GridSearchCV sobre n_estimators, max_depth y max_features de un random forest."""
    grilla = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, 12, None],           # None = sin límite de profundidad
        "max_features": ["sqrt", "log2", 0.5, None],  # None = usar las 7 features (= bagging puro)
    }
    busqueda = GridSearchCV(
        RandomForestClassifier(random_state=SEMILLA, class_weight=class_weight),
        grilla, cv=CV, scoring="accuracy", n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    return busqueda


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = obtener_datos()

    # ------------------------------------------------------------------
    # 3a) Árbol de decisión: elegir max_depth con GridSearchCV
    # ------------------------------------------------------------------
    busq_arbol = buscar_arbol(X_train, y_train)
    res = pd.DataFrame(busq_arbol.cv_results_)
    print("=== Árbol de decisión ===")
    print("Modelos entrenados: 20 profundidades x 5 folds + 1 =", 20 * 5 + 1)
    print("Mejor max_depth:", busq_arbol.best_params_["max_depth"])
    print(f"best_score_ (exactitud media de validación cruzada): {busq_arbol.best_score_:.4f}")

    # Curva: un árbol muy profundo memoriza el entrenamiento (train -> 1.0)
    # pero la validación deja de mejorar o empeora = OVERFITTING.
    fig, eje = plt.subplots(figsize=(8, 5))
    eje.plot(res["param_max_depth"], res["mean_train_score"], "o-", label="entrenamiento")
    eje.plot(res["param_max_depth"], res["mean_test_score"], "s-", label="validación cruzada")
    eje.axvline(busq_arbol.best_params_["max_depth"], color="gray", linestyle="--", label="mejor max_depth")
    eje.set_xlabel("max_depth")
    eje.set_ylabel("exactitud")
    eje.set_title("Árbol de decisión: exactitud según la profundidad")
    eje.legend()
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "arbol_profundidad.png", dpi=110)

    # ------------------------------------------------------------------
    # 3b) Árbol de profundidad 3 (para dibujar e interpretar)
    # ------------------------------------------------------------------
    arbol3 = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA).fit(X_train, y_train)
    fig, eje = plt.subplots(figsize=(20, 9))
    plot_tree(arbol3, feature_names=FEATURES, class_names=list(arbol3.classes_),
              filled=True, rounded=True, fontsize=10, ax=eje)
    eje.set_title("Árbol de decisión de profundidad 3")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "arbol_profundidad3.png", dpi=110)
    print(f"\nExactitud del árbol de profundidad 3 (solo informativa, en entrenamiento): "
          f"{arbol3.score(X_train, y_train):.4f}")

    # Ejemplo de recorrido: predict / predict_proba en la hoja donde cae un grano.
    ejemplo = X_train.iloc[[0]]
    print("\nEjemplo de recorrido (primer grano de entrenamiento):")
    print(ejemplo.T.round(3).to_string(header=False))
    print("  predict       :", arbol3.predict(ejemplo)[0])
    print("  predict_proba :", {c: round(float(p), 3) for c, p in zip(arbol3.classes_, arbol3.predict_proba(ejemplo)[0])})
    print("  hoja (nodo)   :", arbol3.apply(ejemplo)[0])

    # ------------------------------------------------------------------
    # 3c) Random forest: n_estimators, max_depth y max_features
    # ------------------------------------------------------------------
    busq_rf = buscar_random_forest(X_train, y_train)
    print("\n=== Random forest ===")
    print("Mejores hiperparámetros:", busq_rf.best_params_)
    print(f"best_score_ (exactitud media de validación cruzada): {busq_rf.best_score_:.4f}")

    importancias = pd.Series(busq_rf.best_estimator_.feature_importances_, index=FEATURES).sort_values()
    fig, eje = plt.subplots(figsize=(7, 4))
    importancias.plot.barh(ax=eje)
    eje.set_title("Importancia de cada feature (random forest)")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "rf_importancias.png", dpi=110)

    # ------------------------------------------------------------------
    # 3d) Score de PRUEBA (una sola vez, al final)
    # ------------------------------------------------------------------
    print("\n=== Score en el conjunto de PRUEBA ===")
    print(f"Árbol de decisión (max_depth={busq_arbol.best_params_['max_depth']}): "
          f"{busq_arbol.score(X_test, y_test):.4f}")
    print(f"Random forest: {busq_rf.score(X_test, y_test):.4f}")
    print(f"Figuras guardadas en: {CARPETA_FIG}")
    plt.show()

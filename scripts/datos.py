"""
datos.py
========
Funciones compartidas por los scripts del Punto 2:
  1) cargar el dataset de arroz desde el CSV,
  2) limpiarlo,
  3) separar el conjunto de prueba.

Los demás scripts hacen `from datos import ...` para que TODOS usen
exactamente los mismos datos limpios y la MISMA separación de prueba
(esto también lo exige el Punto 3 de la tarea).

Dataset: Rice Dataset Commeo and Osmancik (Kaggle)
https://www.kaggle.com/datasets/muratkokludataset/rice-dataset-commeo-and-osmancik
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta al CSV: está en la carpeta principal del proyecto (un nivel arriba de /scripts).
RUTA_CSV = Path(__file__).resolve().parent.parent / "Rice_Cammeo_Osmancik.csv"

# Las 7 features cuantitativas (todas medidas sobre la foto de cada grano).
FEATURES = [
    "Area",               # nº de píxeles dentro del grano (tamaño)
    "Perimeter",          # longitud del borde del grano, en píxeles
    "Major_Axis_Length",  # largo del grano (eje mayor de la elipse equivalente)
    "Minor_Axis_Length",  # ancho del grano (eje menor de la elipse equivalente)
    "Eccentricity",       # qué tan alargada es la elipse: 0 = círculo, cerca de 1 = muy alargada
    "Convex_Area",        # píxeles del menor polígono convexo que contiene al grano
    "Extent",             # área del grano / área del rectángulo que lo encierra
]
ETIQUETA = "Class"        # variedad de arroz: "Cammeo" u "Osmancik"

SEMILLA = 42              # semilla fija -> resultados reproducibles


def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV y devuelve un DataFrame de pandas."""
    return pd.read_csv(ruta)


def limpiar_datos(df, verbose=True):
    """
    Limpieza de datos. Revisa y corrige:
      - valores faltantes (NaN)      -> se eliminan esas filas
      - filas duplicadas             -> se eliminan
      - valores imposibles:
          * áreas, perímetros y ejes deben ser > 0
          * Eccentricity y Extent deben estar entre 0 y 1
          * el área del grano no puede ser mayor que su área convexa
      - espacios sobrantes en la etiqueta
    Devuelve una copia limpia del DataFrame.
    """
    df = df.copy()
    n_inicial = len(df)

    # 1) Quitar espacios en la etiqueta (" Cammeo" -> "Cammeo").
    df[ETIQUETA] = df[ETIQUETA].astype(str).str.strip()

    # 2) Valores faltantes.
    n_nan = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    # 3) Duplicados.
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 4) Valores físicamente imposibles.
    positivas = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Convex_Area"]
    validas = (
        (df[positivas] > 0).all(axis=1)
        & df["Eccentricity"].between(0, 1)
        & df["Extent"].between(0, 1)
        & (df["Area"] <= df["Convex_Area"])
    )
    n_invalidas = int((~validas).sum())
    df = df[validas].reset_index(drop=True)

    if verbose:
        print(f"Filas iniciales:            {n_inicial}")
        print(f"Filas con valores faltantes: {n_nan}")
        print(f"Filas duplicadas:            {n_dup}")
        print(f"Filas con valores inválidos: {n_invalidas}")
        print(f"Filas finales:              {len(df)}")
    return df


def separar_prueba(df, test_size=0.2):
    """
    Separa el 20 % de los datos como conjunto de PRUEBA.
    - stratify=y mantiene la misma proporción de clases en ambos conjuntos.
    - El conjunto de prueba NO se toca hasta el final (solo para el score final).
    """
    X = df[FEATURES]
    y = df[ETIQUETA]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEMILLA)


def obtener_datos(verbose=False):
    """Atajo: carga + limpia + separa. Devuelve X_train, X_test, y_train, y_test."""
    df = limpiar_datos(cargar_datos(), verbose=verbose)
    return separar_prueba(df)

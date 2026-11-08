# Tarea 3 IA — Puntos 1 y 2

## Dataset: Rice Dataset Commeo and Osmancik

**Kaggle:** https://www.kaggle.com/datasets/muratkokludataset/rice-dataset-commeo-and-osmancik

Fuente original: Cinar, I. y Koklu, M. (2019). *Classification of Rice Varieties Using Artificial
Intelligence Methods*. IJISAE, 7(3), 188-194. También está en el UCI ML Repository (id 545). Licencia CC0.

### Cumplimiento de los requisitos

| Requisito de la tarea | Este dataset | ¿Cumple? |
|---|---|---|
| Datos reales | Fotos reales de 3810 granos de arroz | ✅ |
| Entre 1000 y 10 000 ejemplos | **3810** | ✅ |
| Entre 5 y 10 features cuantitativas | **7**, todas numéricas continuas | ✅ |
| Sin variables indicadoras | Ninguna es 0/1 ni categórica | ✅ |
| Etiqueta categórica | `Class`: Cammeo / Osmancik | ✅ |
| Distinto a tareas y laboratorios anteriores | No es Iris, Titanic, MNIST, etc. | ✅ |

| Feature | Qué mide (en píxeles de la foto del grano) |
|---|---|
| `Area` | Número de píxeles dentro del grano (tamaño) |
| `Perimeter` | Longitud del borde del grano |
| `Major_Axis_Length` | Largo del grano (eje mayor de la elipse equivalente) |
| `Minor_Axis_Length` | Ancho del grano (eje menor de la elipse equivalente) |
| `Eccentricity` | Qué tan alargado es (0 = círculo, cerca de 1 = muy alargado) |
| `Convex_Area` | Píxeles del menor polígono convexo que contiene al grano |
| `Extent` | Área del grano / área del rectángulo que lo encierra |

## Archivos

| Archivo | Contenido |
|---|---|
| `Tarea3_Puntos1y2.ipynb` | Notebook con los puntos 1 y 2 (lo que se entrega) |
| `Rice_Cammeo_Osmancik.csv` | Dataset (el mismo de Kaggle, en formato CSV) |
| `scripts/datos.py` | Carga, limpieza y separación de prueba (lo usan los demás scripts) |
| `scripts/punto1_kernel_cuadratico.py` | Punto 1: phi, verificación de K y costos |
| `scripts/punto2a_exploracion.py` | Punto 2: análisis exploratorio, proporción de clases y limpieza |
| `scripts/punto2b_svc_dos_features.py` | Punto 2: SVC con 2 features, 3 gammas y 3 C, fronteras y vectores de soporte |
| `scripts/punto2c_gridsearch.py` | Punto 2: Pipeline + GridSearchCV, mapa de calor y score de prueba |
| `figuras/` | Gráficos que guardan los scripts |

## Cómo ejecutar

Se necesitan los paquetes: `numpy pandas matplotlib seaborn scikit-learn jupyter`

```powershell
# Notebook
.venv\Scripts\jupyter notebook Tarea3_Puntos1y2.ipynb

# Scripts (ejecutar desde la carpeta scripts)
cd scripts
..\.venv\Scripts\python punto1_kernel_cuadratico.py
..\.venv\Scripts\python punto2a_exploracion.py
..\.venv\Scripts\python punto2b_svc_dos_features.py
..\.venv\Scripts\python punto2c_gridsearch.py
```

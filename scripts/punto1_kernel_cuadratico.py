"""
Punto 1: Kernel cuadrático
==========================
Kernel:  K(x, z) = (x^T z + c)^2 = phi(x)^T phi(z),   con c = 9 y x, z en R^2.

Idea clave: K calcula el producto punto en un espacio de MÁS dimensiones
(el de phi) sin tener que construir phi explícitamente.

Desarrollo (caso general, d features):
    (x^T z + c)^2 = (sum_i x_i z_i + c)^2
                  = sum_i sum_j (x_i x_j)(z_i z_j)            <- términos cuadráticos
                  + sum_i (sqrt(2c) x_i)(sqrt(2c) z_i)          <- términos lineales
                  + c * c                                       <- término constante
Por lo tanto:
    phi(x) = [ x_i x_j  (para todo i, j),  sqrt(2c) x_i (para todo i),  c ]

Para d = 2:
    phi(x) = [x1*x1, x1*x2, x2*x1, x2*x2, sqrt(2c)*x1, sqrt(2c)*x2, c]
"""

import numpy as np

c = 9


def phi(x, c=c):
    """
    Transformación phi del kernel (x^T z + c)^2 para un vector x de CUALQUIER tamaño d.
    Devuelve un vector con d^2 + d + 1 componentes.
    """
    x = np.asarray(x, dtype=float)
    cuadraticos = np.outer(x, x).ravel()   # todos los productos x_i * x_j  (d^2 términos)
    lineales = np.sqrt(2 * c) * x          # sqrt(2c) * x_i                 (d términos)
    constante = [c]                        # c                              (1 término)
    return np.concatenate([cuadraticos, lineales, constante])


def phi_compacta(x, c=c):
    """
    Versión SIN repetidos: como x1*x2 = x2*x1, se juntan en sqrt(2)*x_i*x_j (i < j).
    Da el mismo producto punto pero con (d+1)(d+2)/2 componentes.
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    cuadrados = x ** 2                                                    # x_i^2
    cruzados = [np.sqrt(2) * x[i] * x[j] for i in range(d) for j in range(i + 1, d)]  # sqrt(2) x_i x_j
    lineales = np.sqrt(2 * c) * x
    return np.concatenate([cuadrados, cruzados, lineales, [c]])


def K(x, z, c=c):
    """El kernel: solo un producto punto en R^d, una suma y un cuadrado."""
    return (np.dot(x, z) + c) ** 2


if __name__ == "__main__":
    x = np.array([2, -3])
    z = np.array([1, 4])

    # --- a) phi([2, -3]) ---
    print("a) phi(x) para x = [2, -3]:")
    print("   ", np.round(phi(x), 4))
    print("    componentes: [x1x1, x1x2, x2x1, x2x2, sqrt(18)x1, sqrt(18)x2, c]")
    print("    versión compacta:", np.round(phi_compacta(x), 4))

    # --- b) Verificar phi(x)^T phi(z) = K(x, z) ---
    producto_phi = phi(x) @ phi(z)
    valor_kernel = K(x, z)
    print("\nb) z = [1, 4]")
    print("    phi(z)          =", np.round(phi(z), 4))
    print("    phi(x)^T phi(z) =", round(producto_phi, 6))
    print("    K(x, z)         =", valor_kernel, "  ->  (x^T z + c)^2 = (-10 + 9)^2 = 1")
    print("    phi_compacta(x)^T phi_compacta(z) =", round(phi_compacta(x) @ phi_compacta(z), 6))
    print("    ¿Coinciden?", np.isclose(producto_phi, valor_kernel))

    # --- c) Número de componentes y costo ---
    print("\nc) Componentes de phi y costo, para d features:")
    print("    componentes de phi(x): d^2 + d + 1  (versión compacta: (d+1)(d+2)/2) -> O(d^2)")
    print("    costo de phi(x)^T phi(z): O(d^2) (construir phi y luego sumar d^2+d+1 productos)")
    print("    costo de K(x, z):         O(d)   (un producto punto de d términos + suma + cuadrado)")
    print(f"\n    {'d':>6} {'comp. phi':>14} {'oper. phi^T phi':>16} {'oper. K':>9}")
    for d in [2, 10, 100, 1000]:
        n_phi = d**2 + d + 1
        print(f"    {d:>6} {n_phi:>14} {'~' + str(n_phi):>16} {'~' + str(d):>9}")

    # Comprobación con vectores aleatorios de otro tamaño (phi sirve para cualquier d)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5), rng.normal(size=5)
    print("\n    Prueba con d = 5 al azar: phi(a)^T phi(b) =", round(phi(a) @ phi(b), 6),
          "| K(a, b) =", round(K(a, b), 6))

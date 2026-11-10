"""Figuras del proyecto: curva precisión-recall y matrices de confusión."""

import os

import matplotlib
matplotlib.use("Agg")   # guarda las figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraude import config


def _ruta_figura(nombre_archivo):
    os.makedirs(config.CARPETA_FIGURAS, exist_ok=True)
    return os.path.join(config.CARPETA_FIGURAS, nombre_archivo)


def guardar_curvas_pr(y_test, prob_mlp, prob_regresion, nombre_mlp):
    """Compara la curva precisión-recall de la MLP y de la regresión logística."""
    prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, prob_mlp)
    prec_rl, rec_rl, _ = precision_recall_curve(y_test, prob_regresion)
    ap_mlp = average_precision_score(y_test, prob_mlp)
    ap_rl = average_precision_score(y_test, prob_regresion)

    plt.figure(figsize=(6, 5))
    plt.plot(rec_mlp, prec_mlp, label=f"MLP {nombre_mlp} (PR-AUC = {ap_mlp:.3f})")
    plt.plot(rec_rl, prec_rl, label=f"Regresión logística (PR-AUC = {ap_rl:.3f})")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", label="Azar")
    plt.xlabel("Recall (exhaustividad)")
    plt.ylabel("Precisión")
    plt.title("Curva precisión-recall en prueba")
    plt.legend()
    plt.tight_layout()

    ruta = _ruta_figura("curva_pr_prueba.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print("Figura guardada:", ruta)


def guardar_matriz_confusion(metricas, titulo, nombre_archivo):
    """Dibuja la matriz de confusión a partir del diccionario de calcular_metricas()."""
    matriz = np.array([[metricas["VN"], metricas["FP"]],
                       [metricas["FN"], metricas["VP"]]])

    plt.figure(figsize=(4.5, 4))
    plt.imshow(matriz, cmap="Blues")
    for fila in range(2):
        for columna in range(2):
            plt.text(columna, fila, str(int(matriz[fila][columna])),
                     ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Legítima", "Fraude"])
    plt.yticks([0, 1], ["Legítima", "Fraude"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()

    ruta = _ruta_figura(nombre_archivo)
    plt.savefig(ruta, dpi=150)
    plt.close()
    print("Figura guardada:", ruta)

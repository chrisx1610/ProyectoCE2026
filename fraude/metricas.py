"""Selección del umbral (máximo F2) y cálculo de métricas."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from fraude import config


def calcular_f_beta(precision, recall):
    """F-beta = (1 + b²)·P·R / (b²·P + R). Con b = 2 se obtiene F2."""
    b2 = config.BETA * config.BETA
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)   # 1e-12 evita dividir por 0


def elegir_umbral(y_val, prob_val):
    """Busca el umbral que da el mejor F2 en validación."""
    precision, recall, umbrales = precision_recall_curve(y_val, prob_val)

    # precision y recall tienen un elemento más que umbrales: quitamos el último
    f2 = calcular_f_beta(precision[:-1], recall[:-1])
    return float(umbrales[np.argmax(f2)])


def calcular_metricas(y_real, probabilidades, umbral):
    """Métricas del anteproyecto: PR-AUC, recall, precisión, F2 y matriz de confusión."""
    predicciones = probabilidades >= umbral   # por encima del umbral = fraude

    matriz = confusion_matrix(y_real, predicciones, labels=[0, 1])
    verdaderos_negativos = matriz[0][0]
    falsos_positivos = matriz[0][1]      # falsas alarmas
    falsos_negativos = matriz[1][0]      # fraudes que se escaparon
    verdaderos_positivos = matriz[1][1]  # fraudes detectados

    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
    precision = verdaderos_positivos / max(verdaderos_positivos + falsos_positivos, 1)

    total_normales = verdaderos_negativos + falsos_positivos
    compras_por_falsa_alarma = total_normales / max(falsos_positivos, 1)

    return {
        "pr_auc": average_precision_score(y_real, probabilidades),
        "recall": recall,
        "precision": precision,
        "f2": calcular_f_beta(precision, recall),
        "VN": verdaderos_negativos,
        "FP": falsos_positivos,
        "FN": falsos_negativos,
        "VP": verdaderos_positivos,
        "compras_por_falsa_alarma": compras_por_falsa_alarma,
        "umbral": umbral,
    }

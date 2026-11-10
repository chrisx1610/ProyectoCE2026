"""Comparación de estrategias en validación y evaluación final en prueba."""

import numpy as np
import pandas as pd

from fraude import config
from fraude.entrenamiento import texto_hiper
from fraude.metricas import calcular_metricas, elegir_umbral
from fraude.modelo import obtener_probabilidades


def imprimir_tabla_validacion(resultados_val, mejores_hiper, pr_auc_val_regresion):
    """Tabla comparativa de PR-AUC en validación (objetivo 3)."""
    print("PR-AUC en validación, media ± desv. de", len(config.SEMILLAS), "semillas (más alto es mejor):")
    for estrategia in config.ESTRATEGIAS:
        media = np.mean(resultados_val[estrategia])
        desviacion = np.std(resultados_val[estrategia], ddof=1)
        print(f"   MLP {estrategia:16s} {media:.3f} ± {desviacion:.3f}   "
              f"({texto_hiper(mejores_hiper[estrategia])})")
    print(f"   Regresión logística  {pr_auc_val_regresion:.3f}")
    print()


def elegir_mejor_estrategia(resultados_val):
    """Devuelve la estrategia con mayor PR-AUC media en validación."""
    mejor_estrategia = None
    mejor_media = -1.0
    for estrategia in config.ESTRATEGIAS:
        media = np.mean(resultados_val[estrategia])
        if media > mejor_media:
            mejor_media = media
            mejor_estrategia = estrategia
    return mejor_estrategia


def evaluar_mlp_en_prueba(modelos, X_val, y_val, X_test, y_test):
    """Evalúa cada semilla en prueba con el umbral elegido en validación.

    Devuelve una tabla (una fila por semilla) y las probabilidades de prueba por semilla.
    """
    filas = []
    probabilidades_test = {}
    for semilla, modelo in modelos.items():
        # El umbral se elige en validación (máximo F2), NUNCA en prueba
        umbral = elegir_umbral(y_val, obtener_probabilidades(modelo, X_val))

        prob_test = obtener_probabilidades(modelo, X_test)
        probabilidades_test[semilla] = prob_test

        metricas = calcular_metricas(y_test, prob_test, umbral)
        metricas["semilla"] = semilla
        filas.append(metricas)

    tabla = pd.DataFrame(filas).set_index("semilla")
    return tabla, probabilidades_test


def evaluar_regresion_en_prueba(regresion, X_val, y_val, X_test, y_test):
    """Mismo procedimiento de umbral para la regresión logística."""
    umbral = elegir_umbral(y_val, regresion.predict_proba(X_val)[:, 1])
    prob_test = regresion.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, prob_test, umbral)
    return metricas, prob_test


def imprimir_resultados_prueba(tabla_mlp, metricas_regresion):
    """Imprime las métricas en prueba y responde el objetivo 5."""
    print("MLP en prueba (una fila por semilla):")
    print(tabla_mlp.round(3).to_string())
    print()

    print("MLP en prueba, media ± desviación:")
    resumen = pd.DataFrame({"media": tabla_mlp.mean(), "desv": tabla_mlp.std()})
    print(resumen.round(3).to_string())
    print()

    print("Regresión logística en prueba:")
    print(pd.Series(metricas_regresion).round(3).to_string())
    print()

    # Objetivo 5: ¿la MLP supera a la regresión logística en PR-AUC de prueba?
    pr_auc_mlp = tabla_mlp["pr_auc"].mean()
    pr_auc_regresion = metricas_regresion["pr_auc"]
    print(f"PR-AUC en prueba -> MLP: {pr_auc_mlp:.3f} ± {tabla_mlp['pr_auc'].std():.3f} | "
          f"Regresión logística: {pr_auc_regresion:.3f}")
    if pr_auc_mlp > pr_auc_regresion:
        print("Resultado: la MLP SUPERA a la regresión logística.")
    else:
        print("Resultado: la MLP NO supera a la regresión logística (también es un hallazgo válido).")
    print()

"""Punto de entrada del proyecto: Detección de fraude con redes neuronales.

Grupo 1-G - Universidad Metropolitana

Uso:
    python main.py                      (usa creditcard.csv en esta carpeta)
    python main.py ruta/a/creditcard.csv

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import argparse
import os
import sys

from fraude import config
from fraude.datos import cargar_datos, dividir_y_escalar
from fraude.entrenamiento import buscar_hiperparametros, entrenar_con_semillas, texto_hiper
from fraude.evaluacion import (
    elegir_mejor_estrategia,
    evaluar_mlp_en_prueba,
    evaluar_regresion_en_prueba,
    imprimir_resultados_prueba,
    imprimir_tabla_validacion,
)
from fraude.graficos import guardar_curvas_pr, guardar_matriz_confusion
from fraude.referencia import entrenar_regresion_logistica


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Detección de fraude con una MLP.")
    parser.add_argument("ruta_datos", nargs="?", default=config.RUTA_DATOS_POR_DEFECTO,
                        help="ruta al archivo creditcard.csv")
    return parser.parse_args()


def main():
    argumentos = leer_argumentos()

    if not os.path.isfile(argumentos.ruta_datos):
        print(f"No se encontró el archivo de datos: {argumentos.ruta_datos}")
        print("Descárgalo de https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud,")
        print("descomprímelo y pon creditcard.csv en la carpeta del proyecto,")
        print("o indica la ruta: python main.py ruta/a/creditcard.csv")
        sys.exit(1)

    # 1. Datos: cargar, quitar duplicados, crear variables, dividir y escalar
    X, y = cargar_datos(argumentos.ruta_datos)
    X_train, y_train, X_val, y_val, X_test, y_test = dividir_y_escalar(X, y)

    # 2. Modelo de referencia: regresión logística con ponderación de clases
    regresion, pr_auc_val_regresion = entrenar_regresion_logistica(X_train, y_train, X_val, y_val)

    # 3. MLP: para cada estrategia, rejilla (1 semilla) y luego varias semillas
    mejores_hiper = {}
    modelos = {}
    resultados_val = {}
    for estrategia in config.ESTRATEGIAS:
        print("==========", estrategia, "==========")
        mejores_hiper[estrategia] = buscar_hiperparametros(X_train, y_train, X_val, y_val, estrategia)
        modelos[estrategia], resultados_val[estrategia] = entrenar_con_semillas(
            X_train, y_train, X_val, y_val, estrategia, mejores_hiper[estrategia]
        )
        print()

    # 4. Comparar estrategias en validación y elegir la mejor
    imprimir_tabla_validacion(resultados_val, mejores_hiper, pr_auc_val_regresion)
    mejor_estrategia = elegir_mejor_estrategia(resultados_val)
    print("Estrategia ganadora:", mejor_estrategia,
          "con", texto_hiper(mejores_hiper[mejor_estrategia]))
    print()

    # ========================================================
    # A PARTIR DE AQUÍ SE USA EL CONJUNTO DE PRUEBA (una sola vez)
    # ========================================================

    # 5. Evaluar en prueba la MLP ganadora y la regresión logística
    tabla_mlp, prob_test_mlp = evaluar_mlp_en_prueba(
        modelos[mejor_estrategia], X_val, y_val, X_test, y_test
    )
    metricas_regresion, prob_test_regresion = evaluar_regresion_en_prueba(
        regresion, X_val, y_val, X_test, y_test
    )
    imprimir_resultados_prueba(tabla_mlp, metricas_regresion)

    # 6. Figuras (se usa la primera semilla de la MLP)
    primera_semilla = config.SEMILLAS[0]
    guardar_curvas_pr(y_test, prob_test_mlp[primera_semilla], prob_test_regresion, mejor_estrategia)
    guardar_matriz_confusion(tabla_mlp.loc[primera_semilla],
                             f"MLP {mejor_estrategia} (semilla {primera_semilla})",
                             "matriz_confusion_mlp.png")
    guardar_matriz_confusion(metricas_regresion, "Regresión logística",
                             "matriz_confusion_regresion.png")


if __name__ == "__main__":
    main()

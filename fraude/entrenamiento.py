"""Entrenamiento de la MLP con parada temprana y búsqueda en la rejilla."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from fraude import config
from fraude.balanceo import balancear_datos
from fraude.modelo import RedFraude, obtener_probabilidades


def crear_rejilla():
    """Devuelve la lista con todas las combinaciones de hiperparámetros."""
    rejilla = []
    for capas in config.REJILLA_CAPAS:
        for dropout in config.REJILLA_DROPOUT:
            for lr in config.REJILLA_TASA_APRENDIZAJE:
                rejilla.append({"capas": capas, "dropout": dropout, "lr": lr})
    return rejilla


def texto_hiper(hiper):
    """Texto corto para imprimir una combinación de hiperparámetros."""
    return f"capas={hiper['capas']}, dropout={hiper['dropout']}, lr={hiper['lr']}"


def crear_funcion_perdida(y_train, estrategia):
    """Entropía cruzada binaria; con ponderación, los fraudes pesan más."""
    if estrategia == config.PONDERACION:
        # Peso inversamente proporcional a la frecuencia de cada clase
        cantidad_normales = (y_train == 0).sum()
        cantidad_fraudes = (y_train == 1).sum()
        peso_fraude = torch.tensor(cantidad_normales / cantidad_fraudes, dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=peso_fraude)
    return nn.BCEWithLogitsLoss()


def entrenar_una_epoca(modelo, X, y, funcion_perdida, optimizador, generador):
    """Recorre todos los datos una vez, en lotes barajados."""
    modelo.train()   # modo entrenamiento: activa el dropout
    orden = torch.randperm(len(X), generator=generador)

    for inicio in range(0, len(X), config.TAMANO_LOTE):
        indices = orden[inicio:inicio + config.TAMANO_LOTE]

        optimizador.zero_grad()                       # borrar gradientes anteriores
        prediccion = modelo(X[indices])               # pasar el lote por la red
        perdida = funcion_perdida(prediccion, y[indices])
        perdida.backward()                            # calcular gradientes
        optimizador.step()                            # actualizar pesos


def entrenar_red(X_train, y_train, X_val, y_val, estrategia, hiper, semilla):
    """Entrena una MLP y devuelve (modelo con los mejores pesos, mejor PR-AUC en validación)."""
    torch.manual_seed(semilla)

    # Balancear el entrenamiento según la estrategia
    X_bal, y_bal = balancear_datos(X_train, y_train, estrategia, semilla)
    X_bal = torch.tensor(np.asarray(X_bal), dtype=torch.float32)
    y_bal = torch.tensor(np.asarray(y_bal), dtype=torch.float32)

    funcion_perdida = crear_funcion_perdida(y_train, estrategia)

    neuronas_1, neuronas_2 = hiper["capas"]
    modelo = RedFraude(X_train.shape[1], neuronas_1, neuronas_2, hiper["dropout"])
    optimizador = torch.optim.Adam(modelo.parameters(), lr=hiper["lr"],
                                   weight_decay=config.WEIGHT_DECAY)
    generador = torch.Generator().manual_seed(semilla)   # barajado reproducible

    # Parada temprana según la PR-AUC de validación
    mejor_pr_auc = -1.0
    mejores_pesos = None
    epocas_sin_mejorar = 0

    for _ in range(config.MAX_EPOCAS):
        entrenar_una_epoca(modelo, X_bal, y_bal, funcion_perdida, optimizador, generador)

        # La validación NUNCA se balancea: es como los datos reales
        prob_val = obtener_probabilidades(modelo, X_val)
        pr_auc = average_precision_score(y_val, prob_val)

        if pr_auc > mejor_pr_auc:
            mejor_pr_auc = pr_auc
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejorar = 0
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= config.PACIENCIA:
                break

    # Nos quedamos con los pesos de la mejor época, no con los de la última
    modelo.load_state_dict(mejores_pesos)
    return modelo, mejor_pr_auc


def buscar_hiperparametros(X_train, y_train, X_val, y_val, estrategia):
    """Prueba toda la rejilla con una sola semilla y devuelve la mejor combinación."""
    rejilla = crear_rejilla()
    print(f"Buscando hiperparámetros ({len(rejilla)} combinaciones):")

    mejor_hiper = None
    mejor_pr_auc = -1.0
    for hiper in rejilla:
        _, pr_auc = entrenar_red(X_train, y_train, X_val, y_val,
                                 estrategia, hiper, config.SEMILLA_REJILLA)
        print(f"   {texto_hiper(hiper)} -> PR-AUC val = {pr_auc:.3f}")
        if pr_auc > mejor_pr_auc:
            mejor_pr_auc = pr_auc
            mejor_hiper = hiper

    print("Mejor combinación:", texto_hiper(mejor_hiper))
    return mejor_hiper


def entrenar_con_semillas(X_train, y_train, X_val, y_val, estrategia, hiper):
    """Entrena la misma configuración con varias semillas.

    Devuelve un diccionario {semilla: modelo} y la lista de PR-AUC en validación.
    """
    modelos = {}
    pr_aucs = []
    for semilla in config.SEMILLAS:
        modelo, pr_auc = entrenar_red(X_train, y_train, X_val, y_val, estrategia, hiper, semilla)
        modelos[semilla] = modelo
        pr_aucs.append(pr_auc)
        print(f"   semilla {semilla}: PR-AUC val = {pr_auc:.3f}")
    return modelos, pr_aucs

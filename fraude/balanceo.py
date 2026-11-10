"""Estrategias de balanceo. Se aplican SOLO al conjunto de entrenamiento."""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraude import config


def balancear_datos(X_train, y_train, estrategia, semilla):
    """Devuelve el entrenamiento balanceado según la estrategia.

    "sin_tratamiento" y "ponderacion" no cambian los datos
    (la ponderación se aplica en la función de pérdida).
    """
    if estrategia == config.SMOTE:
        # Crea fraudes "sintéticos" hasta igualar a las compras normales
        smote = SMOTE(random_state=semilla)
        return smote.fit_resample(X_train, y_train)

    if estrategia == config.SUBMUESTREO:
        # Quita compras normales al azar hasta que haya tantas como fraudes
        submuestreo = RandomUnderSampler(random_state=semilla)
        return submuestreo.fit_resample(X_train, y_train)

    if estrategia in (config.SIN_TRATAMIENTO, config.PONDERACION):
        return X_train, y_train

    raise ValueError(f"Estrategia desconocida: {estrategia}")

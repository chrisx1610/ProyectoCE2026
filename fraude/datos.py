"""Carga, limpieza, ingeniería de variables, división y escalado de los datos."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from fraude import config

SEGUNDOS_EN_UN_DIA = 86400
SEGUNDOS_EN_UNA_HORA = 3600


def cargar_datos(ruta):
    """Lee el CSV, quita duplicados y crea las variables.

    Devuelve X (31 columnas) e y (0 = legítima, 1 = fraude).
    """
    print("Cargando datos desde:", ruta)
    df = pd.read_csv(ruta)
    print("Filas leídas:", len(df))

    # Eliminar filas duplicadas (ANTES de dividir)
    df = df.drop_duplicates()
    print("Filas después de quitar duplicados:", len(df))

    df = crear_variables(df)

    # X = V1..V28 + log_amount + hora_sin + hora_cos = 31 columnas
    X = df.drop(columns="Class").values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    print("Total de fraudes:", y.sum())
    print("Número de entradas (columnas):", X.shape[1])
    return X, y


def crear_variables(df):
    """Transforma Amount y Time en variables más útiles para la red."""
    df = df.copy()

    # Monto: log(1 + x) porque hay montos muy grandes
    df["log_amount"] = np.log1p(df["Amount"])

    # Hora del día (0 a 24) a partir de Time (segundos desde la primera compra)
    hora = (df["Time"] % SEGUNDOS_EN_UN_DIA) / SEGUNDOS_EN_UNA_HORA

    # La hora es "circular" (las 23:59 están pegadas a las 00:00),
    # por eso la codificamos con seno y coseno
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    # Quitamos las columnas originales que ya transformamos
    df = df.drop(columns=["Time", "Amount"])
    return df


def dividir_y_escalar(X, y):
    """Divide 60/20/20 de forma estratificada y escala con RobustScaler.

    El escalador se ajusta SOLO con entrenamiento para evitar fuga de datos.
    """
    # 60% entrenamiento, 40% resto. stratify=y mantiene el % de fraudes.
    X_train, X_resto, y_train, y_resto = train_test_split(
        X, y,
        test_size=config.PROPORCION_RESTO,
        stratify=y,
        random_state=config.SEMILLA_DIVISION,
    )

    # El 40% restante se parte a la mitad: 20% validación y 20% prueba
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto, y_resto,
        test_size=config.PROPORCION_PRUEBA_EN_RESTO,
        stratify=y_resto,
        random_state=config.SEMILLA_DIVISION,
    )

    # RobustScaler usa la mediana y el rango intercuartil,
    # así los montos atípicos no lo afectan tanto.
    escalador = RobustScaler()
    escalador.fit(X_train)

    X_train = escalador.transform(X_train).astype(np.float32)
    X_val = escalador.transform(X_val).astype(np.float32)
    X_test = escalador.transform(X_test).astype(np.float32)

    print("Entrenamiento:", len(y_train), "compras,", y_train.sum(), "fraudes")
    print("Validación:   ", len(y_val), "compras,", y_val.sum(), "fraudes")
    print("Prueba:       ", len(y_test), "compras,", y_test.sum(), "fraudes")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test

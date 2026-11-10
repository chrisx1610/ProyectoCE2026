# ============================================================
# Proyecto: Detección de fraude con redes neuronales (MLP)
# Grupo 1-G - Universidad Metropolitana
#
# Qué hace este programa (sigue el anteproyecto):
#   1. Lee el dataset de Kaggle y elimina las filas duplicadas
#   2. Crea las variables: log(1 + Amount) y la hora del día con seno y coseno
#   3. Divide en entrenamiento (60%), validación (20%) y prueba (20%)
#   4. Escala con RobustScaler ajustado SOLO con entrenamiento
#   5. Entrena una regresión logística con ponderación de clases (referencia)
#   6. Para cada una de las 4 estrategias de balanceo de la MLP:
#        a) busca los mejores hiperparámetros en una rejilla pequeña (1 semilla)
#        b) entrena esa configuración con 5 semillas
#   7. Escoge la mejor estrategia usando SOLO validación
#   8. Elige el umbral que maximiza F2 en validación
#   9. Evalúa en prueba (una sola vez): PR-AUC, recall, precisión, F2 y matriz de confusión
#  10. Compara la MLP con la regresión logística en prueba (objetivo 5)
#  11. Guarda las figuras en la carpeta "figuras"
#
# Cómo correrlo:
#   pip install pandas scikit-learn imbalanced-learn torch matplotlib
#   python fraude_mlp.py creditcard.csv
#
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Nota: el anteproyecto decía TensorFlow/Keras; usamos PyTorch (mismo modelo).
# ============================================================

import os
import sys
import copy

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # para guardar las figuras sin abrir ventanas
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ------------------------------------------------------------
# Configuración (todo lo que se puede cambiar está aquí arriba)
# ------------------------------------------------------------
RUTA_POR_DEFECTO = "creditcard.csv"
CARPETA_FIGURAS = "figuras"

SEMILLA_DIVISION = 42          # semilla para dividir los datos (siempre la misma)
SEMILLA_REJILLA = 0            # semilla usada para buscar hiperparámetros
SEMILLAS = [0, 1, 2, 3, 4]     # la configuración final se entrena 5 veces

ESTRATEGIAS = ["sin_tratamiento", "ponderacion", "smote", "submuestreo"]

# Rejilla de hiperparámetros (anteproyecto): 2 x 2 x 2 = 8 combinaciones
REJILLA_CAPAS = [(32, 16), (64, 32)]
REJILLA_DROPOUT = [0.2, 0.3]
REJILLA_TASA_APRENDIZAJE = [0.001, 0.0005]

WEIGHT_DECAY = 0.00001
MAX_EPOCAS = 60
PACIENCIA = 6                  # épocas sin mejorar antes de parar (parada temprana)
TAMANO_LOTE = 512

BETA = 2                       # F2: le da más importancia al recall que a la precisión


# ============================================================
# PASO 1: Cargar los datos, quitar duplicados y crear variables
# ============================================================
def cargar_datos(ruta):
    print("Cargando datos desde:", ruta)
    df = pd.read_csv(ruta)
    print("Filas leídas:", len(df))

    # 1.1 Eliminar filas duplicadas (ANTES de dividir)
    df = df.drop_duplicates()
    print("Filas después de quitar duplicados:", len(df))

    # 1.2 Monto: aplicamos log(1 + x) porque hay montos muy grandes
    df["log_amount"] = np.log1p(df["Amount"])

    # 1.3 Hora del día a partir de Time (segundos desde la primera compra)
    segundos_en_un_dia = 86400
    segundos_en_una_hora = 3600
    hora = (df["Time"] % segundos_en_un_dia) / segundos_en_una_hora   # de 0 a 24

    # La hora es "circular" (las 23:59 están pegadas a las 00:00),
    # por eso la codificamos con seno y coseno
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    # 1.4 Quitamos las columnas originales que ya transformamos
    df = df.drop(columns=["Time", "Amount"])

    # X = V1..V28 + log_amount + hora_sin + hora_cos = 31 columnas
    # y = Class (0 = compra normal, 1 = fraude)
    X = df.drop(columns="Class").values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    print("Total de fraudes:", y.sum())
    print("Número de entradas (columnas):", X.shape[1])
    return X, y


# ============================================================
# PASO 2: Dividir los datos y escalar
# (si se escala antes de dividir, información de prueba se "fuga")
# ============================================================
def dividir_datos(X, y):
    # Primero separamos 60% para entrenamiento y 40% para el resto.
    # stratify=y hace que cada parte tenga el mismo % de fraudes.
    X_train, X_resto, y_train, y_resto = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEMILLA_DIVISION
    )

    # El 40% restante lo partimos a la mitad: 20% validación y 20% prueba
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto, y_resto, test_size=0.5, stratify=y_resto, random_state=SEMILLA_DIVISION
    )

    # RobustScaler usa la mediana y el rango intercuartil,
    # así los montos atípicos no lo afectan tanto.
    # IMPORTANTE: el escalador solo "aprende" con entrenamiento.
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


# ============================================================
# PASO 3: Balancear (SOLO el entrenamiento, nunca validación ni prueba)
# ============================================================
def balancear_datos(X_train, y_train, estrategia, semilla):
    if estrategia == "smote":
        # SMOTE crea fraudes "sintéticos" nuevos hasta igualar a las compras normales
        smote = SMOTE(random_state=semilla)
        X_nuevo, y_nuevo = smote.fit_resample(X_train, y_train)
        return X_nuevo, y_nuevo

    elif estrategia == "submuestreo":
        # Quitamos compras normales al azar hasta que haya tantas como fraudes
        submuestreo = RandomUnderSampler(random_state=semilla)
        X_nuevo, y_nuevo = submuestreo.fit_resample(X_train, y_train)
        return X_nuevo, y_nuevo

    else:
        # "sin_tratamiento" y "ponderacion" usan los datos tal cual.
        # (la ponderación se hace después, en la función de pérdida)
        return X_train, y_train


# ============================================================
# PASO 4: La red neuronal (MLP)
#   31 entradas -> capa oculta 1 -> capa oculta 2 -> 1 salida
# ============================================================
class RedFraude(nn.Module):
    def __init__(self, numero_entradas, neuronas_1, neuronas_2, dropout):
        super().__init__()

        self.capa1 = nn.Linear(numero_entradas, neuronas_1)
        self.capa2 = nn.Linear(neuronas_1, neuronas_2)
        self.salida = nn.Linear(neuronas_2, 1)   # 1 número: qué tan probable es que sea fraude

        self.activacion = nn.ReLU()
        self.dropout = nn.Dropout(dropout)       # apaga neuronas al azar al entrenar

    def forward(self, x):
        x = self.capa1(x)
        x = self.activacion(x)
        x = self.dropout(x)

        x = self.capa2(x)
        x = self.activacion(x)
        x = self.dropout(x)

        x = self.salida(x)
        # Devolvemos el "logit" (sin sigmoide). La sigmoide la aplica
        # la función de pérdida BCEWithLogitsLoss por dentro, y
        # obtener_probabilidades() la aplica al predecir.
        x = x.squeeze(1)
        return x


def obtener_probabilidades(modelo, X):
    """Devuelve la probabilidad de fraude (0 a 1) de cada compra."""
    modelo.eval()   # modo evaluación: apaga el dropout
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = modelo(X_tensor)
        probabilidades = torch.sigmoid(logits)
    return probabilidades.numpy()


# ============================================================
# PASO 5: Entrenar la red con una estrategia, unos hiperparámetros y una semilla
# ============================================================
def entrenar_red(X_train, y_train, X_val, y_val, estrategia, hiper, semilla):
    torch.manual_seed(semilla)

    # 5.1 Balancear el entrenamiento según la estrategia
    X_bal, y_bal = balancear_datos(X_train, y_train, estrategia, semilla)
    X_bal = torch.tensor(np.asarray(X_bal), dtype=torch.float32)
    y_bal = torch.tensor(np.asarray(y_bal), dtype=torch.float32)

    # 5.2 Función de pérdida (entropía cruzada binaria)
    if estrategia == "ponderacion":
        # Peso inversamente proporcional a la frecuencia de cada clase:
        # cada fraude "pesa" tanto como (normales / fraudes) compras normales
        cantidad_normales = (y_train == 0).sum()
        cantidad_fraudes = (y_train == 1).sum()
        peso_fraude = torch.tensor(cantidad_normales / cantidad_fraudes, dtype=torch.float32)
        funcion_perdida = nn.BCEWithLogitsLoss(pos_weight=peso_fraude)
    else:
        funcion_perdida = nn.BCEWithLogitsLoss()

    # 5.3 Crear la red y el optimizador Adam
    neuronas_1, neuronas_2 = hiper["capas"]
    modelo = RedFraude(X_train.shape[1], neuronas_1, neuronas_2, hiper["dropout"])
    optimizador = torch.optim.Adam(modelo.parameters(), lr=hiper["lr"], weight_decay=WEIGHT_DECAY)
    generador = torch.Generator().manual_seed(semilla)   # para que el barajado sea reproducible

    # Variables para la parada temprana
    mejor_pr_auc = -1.0
    mejores_pesos = None
    epocas_sin_mejorar = 0

    # 5.4 Bucle de entrenamiento
    for epoca in range(MAX_EPOCAS):
        modelo.train()   # modo entrenamiento: activa el dropout

        # Barajamos los datos en cada época
        orden = torch.randperm(len(X_bal), generator=generador)

        # Recorremos los datos en lotes de TAMANO_LOTE
        for inicio in range(0, len(X_bal), TAMANO_LOTE):
            indices = orden[inicio:inicio + TAMANO_LOTE]
            X_lote = X_bal[indices]
            y_lote = y_bal[indices]

            optimizador.zero_grad()            # borrar gradientes anteriores
            prediccion = modelo(X_lote)        # pasar el lote por la red
            perdida = funcion_perdida(prediccion, y_lote)
            perdida.backward()                 # calcular gradientes
            optimizador.step()                 # actualizar pesos

        # 5.5 Al final de cada época medimos PR-AUC en validación
        # (la validación NUNCA se balancea, es como los datos reales)
        prob_val = obtener_probabilidades(modelo, X_val)
        pr_auc = average_precision_score(y_val, prob_val)

        if pr_auc > mejor_pr_auc:
            # Mejoró: guardamos una copia de los pesos
            mejor_pr_auc = pr_auc
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejorar = 0
        else:
            epocas_sin_mejorar = epocas_sin_mejorar + 1
            if epocas_sin_mejorar >= PACIENCIA:
                # Lleva PACIENCIA épocas sin mejorar: paramos
                break

    # Nos quedamos con los pesos de la mejor época, no con los de la última
    modelo.load_state_dict(mejores_pesos)
    return modelo, mejor_pr_auc


def crear_rejilla():
    """Devuelve la lista con las 8 combinaciones de hiperparámetros."""
    rejilla = []
    for capas in REJILLA_CAPAS:
        for dropout in REJILLA_DROPOUT:
            for lr in REJILLA_TASA_APRENDIZAJE:
                rejilla.append({"capas": capas, "dropout": dropout, "lr": lr})
    return rejilla


def texto_hiper(hiper):
    return f"capas={hiper['capas']}, dropout={hiper['dropout']}, lr={hiper['lr']}"


# ============================================================
# PASO 6: Elegir el umbral (máximo F2) y calcular métricas
# ============================================================
def calcular_f_beta(precision, recall):
    # F-beta = (1 + b²) * P * R / (b² * P + R)   con b = 2 -> F2
    b2 = BETA * BETA
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)   # 1e-12 evita dividir por 0


def elegir_umbral(y_val, prob_val):
    """Busca el umbral que da el mejor F2 en validación."""
    precision, recall, umbrales = precision_recall_curve(y_val, prob_val)

    # precision y recall tienen un elemento más que umbrales, por eso quitamos el último
    precision = precision[:-1]
    recall = recall[:-1]

    f2 = calcular_f_beta(precision, recall)
    posicion_mejor = np.argmax(f2)
    return float(umbrales[posicion_mejor])


def calcular_metricas(y_real, probabilidades, umbral):
    # Si la probabilidad supera el umbral, decimos que es fraude
    predicciones = probabilidades >= umbral

    matriz = confusion_matrix(y_real, predicciones, labels=[0, 1])
    verdaderos_negativos = matriz[0][0]
    falsos_positivos = matriz[0][1]     # falsas alarmas
    falsos_negativos = matriz[1][0]     # fraudes que se nos escaparon
    verdaderos_positivos = matriz[1][1] # fraudes detectados

    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
    precision = verdaderos_positivos / max(verdaderos_positivos + falsos_positivos, 1)
    f2 = calcular_f_beta(precision, recall)

    total_normales = verdaderos_negativos + falsos_positivos
    compras_por_falsa_alarma = total_normales / max(falsos_positivos, 1)

    return {
        "pr_auc": average_precision_score(y_real, probabilidades),
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "VN": verdaderos_negativos,
        "FP": falsos_positivos,
        "FN": falsos_negativos,
        "VP": verdaderos_positivos,
        "compras_por_falsa_alarma": compras_por_falsa_alarma,
        "umbral": umbral,
    }


# ============================================================
# PASO 7: Figuras
# ============================================================
def guardar_curvas_pr(y_test, prob_mlp, prob_regresion, nombre_mlp):
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
    ruta = os.path.join(CARPETA_FIGURAS, "curva_pr_prueba.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print("Figura guardada:", ruta)


def guardar_matriz_confusion(metricas, titulo, nombre_archivo):
    matriz = np.array([[metricas["VN"], metricas["FP"]],
                       [metricas["FN"], metricas["VP"]]])

    plt.figure(figsize=(4.5, 4))
    plt.imshow(matriz, cmap="Blues")
    for fila in range(2):
        for columna in range(2):
            plt.text(columna, fila, str(matriz[fila][columna]), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Legítima", "Fraude"])
    plt.yticks([0, 1], ["Legítima", "Fraude"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, nombre_archivo)
    plt.savefig(ruta, dpi=150)
    plt.close()
    print("Figura guardada:", ruta)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def main(ruta):
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    # --- Pasos 1 y 2: cargar, limpiar, dividir y escalar ---
    X, y = cargar_datos(ruta)
    X_train, y_train, X_val, y_val, X_test, y_test = dividir_datos(X, y)

    # --- Modelo de referencia: regresión logística con ponderación de clases ---
    print("Entrenando regresión logística con ponderación de clases (referencia)...")
    regresion = LogisticRegression(class_weight="balanced", max_iter=2000)
    regresion.fit(X_train, y_train)
    prob_val_regresion = regresion.predict_proba(X_val)[:, 1]
    pr_auc_val_regresion = average_precision_score(y_val, prob_val_regresion)
    print(f"   PR-AUC val = {pr_auc_val_regresion:.3f}")
    print()

    # --- MLP: para cada estrategia, rejilla (1 semilla) y luego 5 semillas ---
    rejilla = crear_rejilla()
    mejores_hiper = {}
    modelos = {}
    resultados_val = {}

    for estrategia in ESTRATEGIAS:
        print("==========", estrategia, "==========")

        # a) Búsqueda en la rejilla con una sola semilla
        print("Buscando hiperparámetros (", len(rejilla), "combinaciones ):")
        mejor_pr_auc = -1.0
        for hiper in rejilla:
            _, pr_auc = entrenar_red(X_train, y_train, X_val, y_val, estrategia, hiper, SEMILLA_REJILLA)
            print(f"   {texto_hiper(hiper)} -> PR-AUC val = {pr_auc:.3f}")
            if pr_auc > mejor_pr_auc:
                mejor_pr_auc = pr_auc
                mejores_hiper[estrategia] = hiper
        print("Mejor combinación:", texto_hiper(mejores_hiper[estrategia]))

        # b) Entrenar la mejor combinación con 5 semillas
        modelos[estrategia] = {}
        resultados_val[estrategia] = []
        for semilla in SEMILLAS:
            modelo, pr_auc = entrenar_red(X_train, y_train, X_val, y_val,
                                          estrategia, mejores_hiper[estrategia], semilla)
            modelos[estrategia][semilla] = modelo
            resultados_val[estrategia].append(pr_auc)
            print(f"   semilla {semilla}: PR-AUC val = {pr_auc:.3f}")
        print()

    # --- Tabla comparativa en validación (objetivo 3) ---
    print("PR-AUC en validación, media ± desv. de 5 semillas (más alto es mejor):")
    for estrategia in ESTRATEGIAS:
        media = np.mean(resultados_val[estrategia])
        desviacion = np.std(resultados_val[estrategia], ddof=1)
        print(f"   MLP {estrategia:16s} {media:.3f} ± {desviacion:.3f}   ({texto_hiper(mejores_hiper[estrategia])})")
    print(f"   Regresión logística  {pr_auc_val_regresion:.3f}")

    # --- Elegir la mejor estrategia (solo mirando validación) ---
    mejor_estrategia = None
    mejor_media = -1.0
    for estrategia in ESTRATEGIAS:
        media = np.mean(resultados_val[estrategia])
        if media > mejor_media:
            mejor_media = media
            mejor_estrategia = estrategia

    print()
    print("Estrategia ganadora:", mejor_estrategia, "con", texto_hiper(mejores_hiper[mejor_estrategia]))

    # ========================================================
    # A PARTIR DE AQUÍ SE USA EL CONJUNTO DE PRUEBA (una sola vez)
    # ========================================================

    # --- Evaluación de la MLP ganadora en prueba (objetivo 4) ---
    print()
    print("Evaluación final de la MLP en prueba (", len(SEMILLAS), "semillas ):")
    filas = []
    probabilidades_test = {}
    for semilla in SEMILLAS:
        modelo = modelos[mejor_estrategia][semilla]

        # El umbral se elige en validación (máximo F2), NO en prueba
        prob_val = obtener_probabilidades(modelo, X_val)
        umbral = elegir_umbral(y_val, prob_val)

        prob_test = obtener_probabilidades(modelo, X_test)
        probabilidades_test[semilla] = prob_test
        metricas = calcular_metricas(y_test, prob_test, umbral)
        metricas["semilla"] = semilla
        filas.append(metricas)

    tabla = pd.DataFrame(filas).set_index("semilla")
    print(tabla.round(3).to_string())

    print()
    print("Media ± desviación:")
    resumen = pd.DataFrame({"media": tabla.mean(), "desv": tabla.std()})
    print(resumen.round(3).to_string())

    # --- Regresión logística en prueba (mismo procedimiento de umbral) ---
    umbral_regresion = elegir_umbral(y_val, prob_val_regresion)
    prob_test_regresion = regresion.predict_proba(X_test)[:, 1]
    metricas_regresion = calcular_metricas(y_test, prob_test_regresion, umbral_regresion)

    print()
    print("Regresión logística en prueba:")
    print(pd.Series(metricas_regresion).round(3).to_string())

    # --- Objetivo 5: ¿la MLP supera a la regresión logística en PR-AUC de prueba? ---
    pr_auc_mlp = tabla["pr_auc"].mean()
    pr_auc_regresion = metricas_regresion["pr_auc"]
    print()
    print(f"PR-AUC en prueba -> MLP: {pr_auc_mlp:.3f} ± {tabla['pr_auc'].std():.3f} | "
          f"Regresión logística: {pr_auc_regresion:.3f}")
    if pr_auc_mlp > pr_auc_regresion:
        print("Resultado: la MLP SUPERA a la regresión logística.")
    else:
        print("Resultado: la MLP NO supera a la regresión logística (también es un hallazgo válido).")

    # --- Figuras ---
    print()
    primera_semilla = SEMILLAS[0]
    guardar_curvas_pr(y_test, probabilidades_test[primera_semilla], prob_test_regresion, mejor_estrategia)
    guardar_matriz_confusion(filas[0], f"MLP {mejor_estrategia} (semilla {primera_semilla})",
                             "matriz_confusion_mlp.png")
    guardar_matriz_confusion(metricas_regresion, "Regresión logística",
                             "matriz_confusion_regresion.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta_csv = sys.argv[1]
    else:
        ruta_csv = RUTA_POR_DEFECTO
    main(ruta_csv)

"""Configuración del proyecto.

Todos los valores que se pueden cambiar están aquí, así no hay
"números mágicos" repartidos por el código.
"""

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
RUTA_DATOS_POR_DEFECTO = "creditcard.csv"
CARPETA_FIGURAS = "figuras"

# ------------------------------------------------------------
# Semillas (reproducibilidad)
# ------------------------------------------------------------
SEMILLA_DIVISION = 42          # semilla para dividir los datos (siempre la misma)
SEMILLA_REJILLA = 0            # semilla usada para buscar hiperparámetros
SEMILLAS = [0, 1, 2, 3, 4]     # la configuración final se entrena 5 veces

# ------------------------------------------------------------
# División de los datos: 60% entrenamiento / 20% validación / 20% prueba
# ------------------------------------------------------------
PROPORCION_RESTO = 0.4         # 40% sale de entrenamiento...
PROPORCION_PRUEBA_EN_RESTO = 0.5   # ...y se parte a la mitad entre validación y prueba

# ------------------------------------------------------------
# Estrategias de balanceo
# ------------------------------------------------------------
SIN_TRATAMIENTO = "sin_tratamiento"
PONDERACION = "ponderacion"
SMOTE = "smote"
SUBMUESTREO = "submuestreo"

ESTRATEGIAS = [SIN_TRATAMIENTO, PONDERACION, SMOTE, SUBMUESTREO]

# ------------------------------------------------------------
# Rejilla de hiperparámetros (anteproyecto): 2 x 2 x 2 = 8 combinaciones
# ------------------------------------------------------------
REJILLA_CAPAS = [(32, 16), (64, 32)]
REJILLA_DROPOUT = [0.2, 0.3]
REJILLA_TASA_APRENDIZAJE = [0.001, 0.0005]

# ------------------------------------------------------------
# Entrenamiento de la red
# ------------------------------------------------------------
WEIGHT_DECAY = 0.00001
MAX_EPOCAS = 60
PACIENCIA = 6                  # épocas sin mejorar antes de parar (parada temprana)
TAMANO_LOTE = 512

# ------------------------------------------------------------
# Métricas
# ------------------------------------------------------------
BETA = 2                       # F2: le da más importancia al recall que a la precisión

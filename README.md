# Detección de fraude con redes neuronales

Grupo 1-G · Universidad Metropolitana — Tomas Paraco, Arturo Pulgar, Christian Goncalves

Perceptrón multicapa (MLP) para detectar fraude con tarjeta de crédito, comparando cuatro
estrategias de manejo del desbalance (sin tratamiento, ponderación, SMOTE y submuestreo)
con un protocolo sin fuga de datos.

## Cómo correrlo

1. Descargar `creditcard.csv` de [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   y ponerlo en esta carpeta.
2. Instalar las dependencias:
   ```
   python -m pip install -r requirements.txt
   ```
3. Ejecutar:
   ```
   python main.py
   ```
   o indicar otra ruta: `python main.py ruta/a/creditcard.csv`.

Las figuras se guardan en la carpeta `figuras/`.

## Estructura

```
main.py                  Punto de entrada: corre todo el experimento en orden
fraude/
  config.py              Constantes: semillas, rejilla, épocas, rutas
  datos.py               Carga, duplicados, variables, división 60/20/20 y escalado
  balanceo.py            SMOTE y submuestreo (solo entrenamiento)
  modelo.py              La red neuronal (MLP) y la función para predecir
  entrenamiento.py       Entrenamiento con parada temprana y búsqueda en la rejilla
  referencia.py          Regresión logística con ponderación de clases
  metricas.py            Umbral por F2 y métricas (PR-AUC, recall, precisión, F2, matriz)
  evaluacion.py          Comparación en validación y evaluación final en prueba
  graficos.py            Curva precisión-recall y matrices de confusión
versiones_anteriores/    Versiones del script en un solo archivo (referencia)
```

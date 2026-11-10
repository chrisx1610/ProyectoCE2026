"""Definición de la red neuronal (MLP) y función para predecir."""

import torch
import torch.nn as nn


class RedFraude(nn.Module):
    """MLP: entradas -> capa oculta 1 -> capa oculta 2 -> 1 salida."""

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
        # BCEWithLogitsLoss por dentro al entrenar, y
        # obtener_probabilidades() al predecir.
        return x.squeeze(1)


def obtener_probabilidades(modelo, X):
    """Devuelve la probabilidad de fraude (0 a 1) de cada compra."""
    modelo.eval()   # modo evaluación: apaga el dropout
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = modelo(X_tensor)
        probabilidades = torch.sigmoid(logits)
    return probabilidades.numpy()

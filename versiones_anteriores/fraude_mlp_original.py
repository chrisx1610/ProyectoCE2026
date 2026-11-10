"""Detección de fraude con una MLP: comparación de 4 estrategias de balanceo sin fuga de datos.

Uso:  python fraude_mlp.py creditcard.csv
Datos: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Requiere: pip install pandas scikit-learn imbalanced-learn torch
"""
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

SEMILLAS = [0, 1, 2, 3, 4]
ESTRATEGIAS = ["sin_tratamiento", "smote", "submuestreo", "ponderacion"]


# ---------- 1. Datos: dividir ANTES de todo ----------
def cargar(ruta):
    df = pd.read_csv(ruta)
    df["Hour"] = (df["Time"] % 86400) / 3600          # hora relativa del día -> 31 entradas
    X = df.drop(columns="Class").values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    return X, y


def dividir(X, y, semilla=42):
    """60 % entrenamiento / 20 % validación / 20 % prueba, estratificado."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=semilla)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=semilla)
    esc = StandardScaler().fit(X_tr)                   # el escalador solo ve entrenamiento
    return esc.transform(X_tr), y_tr, esc.transform(X_va), y_va, esc.transform(X_te), y_te


# ---------- 2. Balanceo: solo sobre entrenamiento ----------
def balancear(X, y, estrategia, semilla):
    if estrategia == "smote":
        return SMOTE(random_state=semilla).fit_resample(X, y)
    if estrategia == "submuestreo":
        return RandomUnderSampler(random_state=semilla).fit_resample(X, y)
    return X, y                                        # sin_tratamiento y ponderacion no tocan los datos


# ---------- 3. Modelo ----------
class MLP(nn.Module):
    def __init__(self, n_entradas, dropout=0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),                          # logit; la sigmoide va dentro de la pérdida
        )

    def forward(self, x):
        return self.red(x).squeeze(1)


def predecir(modelo, X):
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(torch.from_numpy(X.astype(np.float32)))).numpy()


def entrenar_mlp(X_tr, y_tr, X_va, y_va, estrategia, semilla, epocas=60, paciencia=6, lote=512):
    torch.manual_seed(semilla)
    Xb, yb = balancear(X_tr, y_tr, estrategia, semilla)
    Xb = torch.from_numpy(np.asarray(Xb, dtype=np.float32))
    yb = torch.from_numpy(np.asarray(yb, dtype=np.float32))

    peso = None
    if estrategia == "ponderacion":                    # un fraude pesa n_neg / n_pos veces más
        peso = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32)
    perdida = nn.BCEWithLogitsLoss(pos_weight=peso)

    modelo = MLP(X_tr.shape[1])
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(semilla)

    mejor_ap, mejor_estado, sin_mejora = -1.0, None, 0
    for _ in range(epocas):
        modelo.train()
        orden = torch.randperm(len(Xb), generator=gen)
        for i in range(0, len(Xb), lote):
            idx = orden[i:i + lote]
            opt.zero_grad()
            perdida(modelo(Xb[idx]), yb[idx]).backward()
            opt.step()
        # parada temprana con PR-AUC en validación (validación NUNCA se balancea)
        ap = average_precision_score(y_va, predecir(modelo, X_va))
        if ap > mejor_ap:
            mejor_ap, mejor_estado, sin_mejora = ap, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, mejor_ap


# ---------- 4. Umbral y métricas ----------
def elegir_umbral(y_va, p_va):
    """Umbral que maximiza F1 en validación."""
    prec, rec, umbrales = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(umbrales[np.argmax(f1)])


def informe(y, p, umbral):
    tn, fp, fn, tp = confusion_matrix(y, p >= umbral, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, p),
        "recall": tp / (tp + fn),                      # fraudes detectados
        "precision": tp / max(tp + fp, 1),
        "compras_por_falsa_alarma": (tn + fp) / max(fp, 1),
    }


# ---------- 5. Experimento ----------
def main(ruta):
    X, y = cargar(ruta)
    X_tr, y_tr, X_va, y_va, X_te, y_te = dividir(X, y)
    print(f"train {len(y_tr)} ({y_tr.sum()} fraudes) | val {len(y_va)} ({y_va.sum()}) | test {len(y_te)} ({y_te.sum()})")

    # Línea base: regresión logística
    rl = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    ap_rl = average_precision_score(y_va, rl.predict_proba(X_va)[:, 1])

    # 4 estrategias x 5 semillas, todo se decide en validación
    modelos, val = {}, {e: [] for e in ESTRATEGIAS}
    for e in ESTRATEGIAS:
        for s in SEMILLAS:
            modelos[e, s], ap = entrenar_mlp(X_tr, y_tr, X_va, y_va, e, s)
            val[e].append(ap)
        print(f"  MLP {e:16s} PR-AUC val = {np.mean(val[e]):.3f} ± {np.std(val[e]):.3f}")
    print(f"  Regresión logística  PR-AUC val = {ap_rl:.3f}")

    # Se elige la ganadora y su umbral en validación; el test se abre UNA sola vez
    ganadora = max(ESTRATEGIAS, key=lambda e: np.mean(val[e]))
    print(f"\nEstrategia elegida: {ganadora}. Evaluación final en test ({len(SEMILLAS)} semillas):")
    filas = []
    for s in SEMILLAS:
        m = modelos[ganadora, s]
        umbral = elegir_umbral(y_va, predecir(m, X_va))
        filas.append(informe(y_te, predecir(m, X_te), umbral))
    res = pd.DataFrame(filas)
    print(res.round(3).to_string())
    print("\nMedia ± desv.:")
    print(pd.DataFrame({"media": res.mean(), "desv": res.std()}).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv")

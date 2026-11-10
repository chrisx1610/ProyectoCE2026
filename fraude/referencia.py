"""Modelo de referencia: regresión logística con ponderación de clases."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def entrenar_regresion_logistica(X_train, y_train, X_val, y_val):
    """Entrena la regresión logística y devuelve (modelo, PR-AUC en validación)."""
    print("Entrenando regresión logística con ponderación de clases (referencia)...")
    regresion = LogisticRegression(class_weight="balanced", max_iter=2000)
    regresion.fit(X_train, y_train)

    prob_val = regresion.predict_proba(X_val)[:, 1]
    pr_auc_val = average_precision_score(y_val, prob_val)
    print(f"   PR-AUC val = {pr_auc_val:.3f}")
    print()
    return regresion, pr_auc_val

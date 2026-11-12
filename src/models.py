"""Torneo predictivo y calibración (Fase 3).

ESQUELETO - todavía sin implementación (sprint de diseño 12/11/2026).
Dueño: Backend & ML Engineer. Revisor: QA & Validation Engineer.

Protocolo inmutable:
    - Test (20%) se toca UNA sola vez, al final, para la matriz comparativa.
    - Toda búsqueda de hiperparámetros usa StratifiedKFold(n_splits=5, shuffle=True,
      random_state=RANDOM_STATE) SOLO dentro de Train.
    - Todo modelo es un Pipeline(preprocessor, estimator) para evitar leakage en CV.

Familias:
    1. Logística regularizada (L1/L2/ElasticNet) + Logit statsmodels para Odds Ratios con IC 95%.
    2. Naive Bayes (Gaussian/Bernoulli/Categorical) + análisis de independencia condicional.
    3. CART sin restricción (overfit) vs CART podado con ccp_alpha óptimo por CV.
    4. Random Forest + RandomizedSearchCV + oob_score=True (OOB vs CV).
"""

from __future__ import annotations


def make_cv(random_state: int = 42):
    """StratifiedKFold 5 folds. TODO."""
    raise NotImplementedError


def fit_logistic_family(X_train, y_train, preprocessor):
    """GridSearch sobre C y l1_ratio; devuelve mejor pipeline. TODO."""
    raise NotImplementedError


def logit_odds_ratios(df, formula: str):
    """statsmodels Logit -> OR = exp(beta) con IC 95%. TODO."""
    raise NotImplementedError


def fit_naive_bayes(X_train, y_train, preprocessor):
    """NB + diagnóstico de dependencia condicional (corr. intra-clase). TODO."""
    raise NotImplementedError


def fit_cart_with_pruning(X_train, y_train, preprocessor):
    """cost_complexity_pruning_path -> alpha_opt por CV (regla 1-SE opcional). TODO."""
    raise NotImplementedError


def fit_random_forest(X_train, y_train, preprocessor):
    """RandomizedSearchCV (n_estimators, max_depth, min_samples_split/leaf, max_features) + OOB. TODO."""
    raise NotImplementedError


def evaluate_on_test(models: dict, X_test, y_test):
    """Precision, Recall, F1 macro/ponderado, ROC-AUC, PR-AUC, Brier. TODO."""
    raise NotImplementedError


def reliability_curves(models: dict, X_test, y_test, n_bins: int = 10):
    """sklearn.calibration.calibration_curve por modelo (+ ECE). TODO."""
    raise NotImplementedError


def importance_mdi_vs_permutation(rf_pipeline, X_test, y_test):
    """MDI (Gini) vs permutation_importance en TEST (scoring='roc_auc', n_repeats>=20). TODO."""
    raise NotImplementedError

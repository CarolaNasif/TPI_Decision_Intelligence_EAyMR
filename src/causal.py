"""Inferencia causal y Causal ML (Fase 4).

ESQUELETO - todavía sin implementación (sprint de diseño 12/11/2026).
Dueño: Backend & ML / Causal Engineer. Revisor: QA & Validation Engineer.

Reglas de identificación (Pearl):
    - El set de ajuste = CONFOUNDERS (Z) pre-tratamiento.
    - MEDIATORS (M) y COLLIDERS (C) se declaran en el DAG pero NUNCA entran al ajuste.
    - Se reporta siempre el coeficiente ingenuo Y ~ T para cuantificar el sesgo de confusión.

Flujo DoWhy: Model -> Identify -> Estimate (linear_regression + propensity_score_weighting)
             -> Refute (random_common_cause, placebo_treatment_refuter, data_subset_refuter).
Causal ML : EconML LinearDML / CausalForestDML con RandomForest como nuisance (model_y, model_t).
"""

from __future__ import annotations


def build_dag(confounders, treatment: str, outcome: str, mediators=(), colliders=()):
    """Construye el DAG (networkx.DiGraph) y su serialización DOT/GML para DoWhy y la app. TODO."""
    raise NotImplementedError


def check_backdoor(dag, treatment: str, outcome: str, adjustment_set):
    """Verifica el criterio backdoor: ningún nodo descendiente de T y bloqueo de caminos T <- ... -> Y.

    TODO: networkx d-separation sobre el grafo con las aristas salientes de T removidas.
    """
    raise NotImplementedError


def naive_effect(df, treatment: str, outcome: str):
    """Regresión ingenua Y ~ T (coeficiente + IC 95%). TODO."""
    raise NotImplementedError


def dowhy_pipeline(df, dag_dot: str, treatment: str, outcome: str):
    """CausalModel -> identify_effect -> estimate_effect (2 métodos). TODO."""
    raise NotImplementedError


def run_refutations(model, identified_estimand, estimate):
    """3 refutadores + semáforo (verde/amarillo/rojo) serializable a JSON para la app. TODO."""
    raise NotImplementedError


def fit_dml_cate(df, treatment: str, outcome: str, effect_modifiers, confounders):
    """EconML LinearDML / CausalForestDML (RF nuisance, discrete_treatment=True) -> CATE + IC. TODO."""
    raise NotImplementedError

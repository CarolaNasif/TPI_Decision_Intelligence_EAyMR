"""Smoke test del stack causal (DoWhy + EconML) sobre datos sintéticos con ATE conocido.

Detecta incompatibilidades de versiones entre dowhy, pandas y networkx
(ver BITACORA.md, Entrada 5, Sesión IA-02).
"""

import numpy as np
import pandas as pd
import pytest

TRUE_ATE = 2.0


@pytest.fixture(scope="module")
def confounded_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2000
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(int)
    y = TRUE_ATE * t + 3 * z + rng.normal(size=n)
    return pd.DataFrame({"Z": z, "T": t, "Y": y})


def test_dowhy_backdoor_recovers_true_ate(confounded_df):
    dowhy = pytest.importorskip("dowhy")
    model = dowhy.CausalModel(data=confounded_df, treatment="T", outcome="Y", common_causes=["Z"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    naive = confounded_df.groupby("T").Y.mean().diff().iloc[-1]
    assert abs(estimate.value - TRUE_ATE) < 0.3
    assert abs(naive - TRUE_ATE) > 1.0  # el ingenuo está sesgado por Z


def test_econml_linear_dml_recovers_true_ate(confounded_df):
    dml = pytest.importorskip("econml.dml")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    est = dml.LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=0),
        model_t=RandomForestClassifier(n_estimators=100, random_state=0),
        discrete_treatment=True,
        random_state=0,
    )
    est.fit(confounded_df.Y, confounded_df["T"], X=confounded_df[["Z"]])
    assert abs(float(est.ate(confounded_df[["Z"]])) - TRUE_ATE) < 0.3

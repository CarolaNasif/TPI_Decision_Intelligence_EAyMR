"""Tests de Fase 2 (dueño: QA & Validation Engineer): MCO, Gauss-Markov, influencia, VIF y PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import RANDOM_STATE
from src import inference as inf
from src import preprocessing as prep


# --------------------------------------------------------------------------- fit_ols / ols_coef_table
def test_fit_ols_recovers_known_coefficients():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    res = inf.fit_ols(df, "y ~ x1 + x2")

    assert res.params["Intercept"] == pytest.approx(5.0, abs=0.1)
    assert res.params["x1"] == pytest.approx(2.0, abs=0.1)
    assert res.params["x2"] == pytest.approx(-1.5, abs=0.1)
    assert res.rsquared > 0.9
    assert res.f_pvalue < 1e-10

    table = inf.ols_coef_table(res)
    assert set(table["term"]) == {"Intercept", "x1", "x2"}
    row_x1 = table.loc[table["term"] == "x1"].iloc[0]
    assert row_x1["ci_low"] < 2.0 < row_x1["ci_high"]
    assert row_x1["decision"] == "Se rechaza H0"


def test_fit_ols_robust_hc3_changes_only_standard_errors():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 3000
    x = rng.normal(size=n)
    # heterocedasticidad fuerte: la varianza del error crece con x
    y = 1.0 + 3.0 * x + rng.normal(scale=0.2 + 2.0 * np.abs(x), size=n)
    df = pd.DataFrame({"y": y, "x": x})

    classic = inf.fit_ols(df, "y ~ x")
    robust = inf.fit_ols(df, "y ~ x", robust="HC3")

    np.testing.assert_allclose(classic.params.to_numpy(), robust.params.to_numpy(), rtol=1e-10)
    assert not np.allclose(classic.bse.to_numpy(), robust.bse.to_numpy())


# --------------------------------------------------------------------------- gauss_markov_diagnostics
def test_gauss_markov_flags_heteroscedasticity():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    x = rng.uniform(1.0, 10.0, size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=x, size=n)  # Var(error) crece con x
    res = inf.fit_ols(pd.DataFrame({"y": y, "x": x}), "y ~ x")

    diag = inf.gauss_markov_diagnostics(res)

    assert diag["breusch_pagan_p"] < 0.01
    assert diag["white_p"] < 0.01
    assert diag["homoscedastic"] is False
    assert diag["recommend_hc3"] is True
    assert np.isfinite(diag["durbin_watson"])
    assert len(diag["qq_theoretical"]) == len(diag["qq_sample"]) == n


def test_gauss_markov_does_not_flag_well_behaved_residuals():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=1.0, size=n)  # homocedástico y normal por diseño
    res = inf.fit_ols(pd.DataFrame({"y": y, "x": x}), "y ~ x")

    diag = inf.gauss_markov_diagnostics(res)

    assert diag["breusch_pagan_p"] > 0.01
    assert diag["homoscedastic"] is True
    assert diag["recommend_hc3"] is False


# --------------------------------------------------------------------------- influence_diagnostics
def test_influence_diagnostics_flags_planted_outlier_and_leverage():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 500
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)

    # Caso 1: outlier puro en Y (x típico, residuo enorme).
    x = np.append(x, 0.0)
    y = np.append(y, 1.0 + 2.0 * 0.0 + 20.0)
    # Caso 2: alto leverage (x extremo) consistente con la recta (poco residuo).
    x = np.append(x, 15.0)
    y = np.append(y, 1.0 + 2.0 * 15.0)

    res = inf.fit_ols(pd.DataFrame({"y": y, "x": x}), "y ~ x")
    infl = inf.influence_diagnostics(res)

    assert infl.loc[n, "is_outlier"], "El outlier plantado en Y no fue detectado"
    assert infl.loc[n + 1, "is_high_leverage"], "El punto de alto leverage no fue detectado"
    assert infl["leverage"].notna().all()
    assert infl.attrs["leverage_threshold"] > 0
    assert infl.attrs["cook_screen_threshold"] == pytest.approx(4.0 / (n + 2))
    assert set(infl["classification"]) != {"normal"}


# --------------------------------------------------------------------------- vif_table
def test_vif_table_flags_collinear_variables():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.05, size=n)  # casi colineal con x1
    x3 = rng.normal(size=n)  # independiente
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    table = inf.vif_table(X)

    vif_x1 = table.loc[table["variable"] == "x1", "VIF"].iloc[0]
    vif_x3 = table.loc[table["variable"] == "x3", "VIF"].iloc[0]
    assert vif_x1 > 10
    assert table.loc[table["variable"] == "x1", "flag"].iloc[0] == ">10"
    assert vif_x3 < 5
    assert table.loc[table["variable"] == "x3", "flag"].iloc[0] == "OK"
    assert "const" not in table["variable"].to_numpy()


# --------------------------------------------------------------------------- fit_pca_train_only
@pytest.fixture(scope="module")
def pca_split():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1000
    z = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": z + rng.normal(scale=0.1, size=n),
            "b": z + rng.normal(scale=0.1, size=n),
            "c": rng.normal(size=n),
        }
    )
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)


def test_fit_pca_train_only_fitted_only_on_train(pca_split):
    X_tr, X_te = pca_split
    result = prep.fit_pca_train_only(X_tr)

    scaler = result["pipeline"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().to_numpy(), rtol=1e-10)
    full_mean = pd.concat([X_tr, X_te]).mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean), "El scaler de la PCA vio datos de Test (leakage)"

    # Aplicar a ciegas a Test no debe re-ajustar nada.
    Z_test = result["pipeline"].transform(X_te)
    assert Z_test.shape == (len(X_te), 3)


def test_fit_pca_train_only_returns_loadings_and_variance(pca_split):
    X_tr, _ = pca_split
    result = prep.fit_pca_train_only(X_tr)

    assert list(result["loadings"].index) == list(X_tr.columns)
    assert list(result["loadings"].columns) == ["PC1", "PC2", "PC3"]
    assert result["explained_variance_ratio"].sum() == pytest.approx(1.0, abs=1e-8)
    # "a" y "b" comparten la señal z: deberían cargar fuerte y con el mismo signo en PC1.
    pc1 = result["loadings"]["PC1"]
    assert np.sign(pc1["a"]) == np.sign(pc1["b"])
    assert abs(pc1["a"]) > abs(pc1["c"])
    # Con z dominando 2 de las 3 variables, el criterio de Kaiser debería quedarse con 1 componente.
    assert result["kaiser_n_components"] >= 1


def test_fit_pca_train_only_n_components_limits_output(pca_split):
    X_tr, _ = pca_split
    result = prep.fit_pca_train_only(X_tr, n_components=2)
    assert result["loadings"].shape[1] == 2
    assert len(result["explained_variance_ratio"]) == 2

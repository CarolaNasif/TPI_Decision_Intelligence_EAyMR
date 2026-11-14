"""Tests de src/models.py (Fase 3) con datos sintéticos chicos + chequeo de artefactos reales."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import models as mdl
from src import preprocessing as prep


@pytest.fixture(scope="module")
def data():
    """N=600 con el esquema del dataset; churn depende de Tenure y Complain (señal aprendible)."""
    rng = np.random.default_rng(prep.RANDOM_STATE)
    n = 600
    df = pd.DataFrame({c: rng.lognormal(1.0, 0.8, n).round(2) for c in prep.NUMERIC_COLS})
    df["Complain"] = rng.integers(0, 2, n)
    for c, levels in {
        "PreferredLoginDevice": ["Mobile Phone", "Computer"],
        "CityTier": [1, 2, 3],
        "PreferredPaymentMode": ["Debit Card", "Credit Card", "UPI"],
        "Gender": ["Male", "Female"],
        "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Grocery"],
        "MaritalStatus": ["Single", "Married", "Divorced"],
    }.items():
        df[c] = pd.Categorical(rng.choice(levels, n))
    logit = -1.5 - 0.6 * (df["Tenure"] - df["Tenure"].mean()) + 1.2 * df["Complain"]
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.choice(n, 30, replace=False), "Tenure"] = np.nan
    X, y = df.drop(columns="Churn"), df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=prep.RANDOM_STATE)


@pytest.fixture(scope="module")
def rf(data):
    X_tr, _, y_tr, _ = data
    return mdl.fit_random_forest(X_tr, y_tr, n_iter=2)


def test_cv_is_stratified_5_fold():
    cv = mdl.make_cv()
    assert cv.get_n_splits() == 5 and cv.shuffle and cv.random_state == prep.RANDOM_STATE


def test_features_exclude_id_target_and_treatment():
    assert not {*prep.ID_COLS, prep.TARGET_COL, prep.TREATMENT_COL} & set(mdl.FEATURES)


def test_logistic_search_uses_cv_only_on_train(data):
    X_tr, _, y_tr, _ = data
    search = mdl.fit_logistic_family(X_tr, y_tr)
    assert search.n_splits_ == 5
    assert search.best_estimator_.named_steps["pre"].n_features_in_ == len(mdl.FEATURES)
    assert 0.5 < mdl.cv_summary(search)["cv_roc_auc_mean"] <= 1


def test_odds_ratios_have_ci_containing_point(data):
    X_tr, _, y_tr, _ = data
    table = mdl.logit_odds_ratios(pd.concat([X_tr, y_tr], axis=1), "Churn ~ Tenure + Complain + C(CityTier)")
    assert (table.IC95_inf <= table.OR).all() and (table.OR <= table.IC95_sup).all()
    assert table.loc["Complain", "OR"] > 1  # la señal sintética sube el churn con reclamo


def test_unpruned_cart_overfits_and_pruning_path(data):
    X_tr, _, y_tr, _ = data
    unpruned = mdl.fit_cart_unpruned(X_tr, y_tr)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y_tr, unpruned.predict_proba(X_tr[mdl.FEATURES])[:, 1]) == pytest.approx(1.0)
    search, table, alpha_1se = mdl.fit_cart_with_pruning(X_tr, y_tr, max_alphas=10)
    assert table.n_hojas.is_monotonic_decreasing
    assert alpha_1se >= search.best_params_["model__ccp_alpha"]


def test_evaluate_on_test_reports_required_metrics(data, rf):
    _, X_te, _, y_te = data
    table = mdl.evaluate_on_test({"rf": rf.best_estimator_}, X_te, y_te)
    required = {"precision", "recall", "f1_macro", "f1_ponderado", "roc_auc", "pr_auc", "brier", "ece"}
    assert required <= set(table.columns)
    row = table.loc["rf"]
    assert row.roc_auc_ic95_inf <= row.roc_auc <= row.roc_auc_ic95_sup
    assert 0 <= row.brier <= 1


def test_expected_calibration_error():
    y = np.array([0, 0, 1, 1] * 50)
    assert mdl.expected_calibration_error(y, np.full(len(y), 0.5)) == pytest.approx(0.0)
    assert mdl.expected_calibration_error(y, np.full(len(y), 0.9)) == pytest.approx(0.4)


def test_rf_oob_and_convergence(data, rf):
    X_tr, _, y_tr, _ = data
    oob = mdl.rf_oob(rf.best_estimator_, X_tr, y_tr)
    assert 0 < oob["oob_accuracy"] <= 1 and 0 < oob["oob_roc_auc"] <= 1
    curve = mdl.oob_convergence(rf.best_estimator_, X_tr, y_tr, n_grid=[10, 50])
    assert list(curve.n_estimators) == [10, 50]


def test_mdi_aggregates_to_original_features(rf):
    mdi = mdl.mdi_by_feature(rf.best_estimator_)
    assert set(mdi.index) == set(mdl.FEATURES)
    assert mdi.sum() == pytest.approx(1.0)


def test_permutation_vs_mdi_table(data, rf):
    _, X_te, _, y_te = data
    table = mdl.importance_mdi_vs_permutation(rf.best_estimator_, X_te, y_te, n_repeats=3)
    assert set(table.index) == set(mdl.FEATURES)
    assert {"perm_mean", "perm_std", "mdi", "rank_mdi", "rank_perm"} <= set(table.columns)


def test_naive_bayes_dependency_diagnostic(data):
    X_tr, _, y_tr, _ = data
    top = mdl.top_conditional_dependencies(pd.concat([X_tr, y_tr], axis=1), k=3)
    assert len(top) == 3 and set(top.clase) <= {0, 1}


@pytest.mark.skipif(not mdl.RESULTS_PATH.exists(), reason="artifacts/model_results.json aún no generado")
def test_saved_artifacts_are_consistent():
    import joblib

    results = json.loads(mdl.RESULTS_PATH.read_text(encoding="utf-8"))
    assert results["best_model"] == max(results["cv"], key=lambda k: results["cv"][k]["cv_roc_auc_mean"])
    model = joblib.load(mdl.BEST_MODEL_PATH)
    df = prep.make_clean_dataset() if prep.RAW_PATH.exists() else None
    if df is not None:
        _, X_te, _, _ = prep.load_split(df)
        proba = model.predict_proba(X_te[mdl.FEATURES])[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()

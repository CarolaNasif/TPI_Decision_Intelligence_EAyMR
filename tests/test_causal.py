"""Tests de src/causal.py (Fase 4): DAG, backdoor, estimadores con efecto conocido y artefactos."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import causal
from src import preprocessing as prep

TRUE_ATE = -0.10


@pytest.fixture(scope="module")
def dag():
    return causal.build_dag()


@pytest.fixture(scope="module")
def clean_df(tmp_path_factory) -> pd.DataFrame:
    """Dataset con el esquema real, confusión por Tenure y Complain, y ATE verdadero = -0,10.

    HourSpendOnApp = 4 implica T = 1 siempre (violación de positividad plantada, como en el dataset real).
    """
    rng = np.random.default_rng(prep.RANDOM_STATE)
    n = 3000
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + n),
            "PreferredLoginDevice": rng.choice(["Mobile Phone", "Computer"], n),
            "CityTier": rng.choice([1, 2, 3], n),
            "PreferredPaymentMode": rng.choice(["Debit Card", "Credit Card", "UPI"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "PreferedOrderCat": rng.choice(["Laptop & Accessory", "Mobile Phone", "Grocery"], n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        }
    )
    for col in prep.NUMERIC_COLS:
        df[col] = rng.lognormal(1.0, 0.5, n).round(2)
    df["Tenure"] = rng.integers(0, 30, n).astype(float)
    df["Complain"] = rng.integers(0, 2, n)
    df["SatisfactionScore"] = rng.integers(1, 6, n)
    df["HourSpendOnApp"] = rng.choice([1, 2, 3, 4], n, p=[0.2, 0.35, 0.35, 0.1]).astype(float)
    df["NumberOfDeviceRegistered"] = rng.integers(1, 6, n)
    p_t = 1 / (1 + np.exp(-(-0.5 + 0.08 * df["Tenure"] - 0.8 * df["Complain"])))
    t = (rng.random(n) < p_t).astype(int)
    t[df["HourSpendOnApp"].to_numpy() == 4] = 1
    df["CouponUsed"] = np.where(t == 1, rng.integers(1, 4, n), 0).astype(float)
    p_y = np.clip(0.35 - 0.008 * df["Tenure"] + 0.15 * df["Complain"] + TRUE_ATE * t, 0.01, 0.99)
    df["Churn"] = (rng.random(n) < p_y).astype(int)
    path = tmp_path_factory.mktemp("raw") / "fake.csv"
    df[prep.EXPECTED_COLUMNS].to_csv(path, index=False)
    return prep.make_clean_dataset(path)


@pytest.fixture(scope="module")
def frame(clean_df):
    return causal.prepare_causal_frame(clean_df)


# --------------------------------------------------------------------------- DAG y backdoor
def test_dag_roles_match_contract(dag):
    roles = dict(dag.nodes(data="role"))
    assert roles[causal.TREATMENT] == "T" and roles[causal.OUTCOME] == "Y"
    assert {n for n, r in roles.items() if r == "Z"} == set(prep.CONFOUNDERS)
    assert {n for n, r in roles.items() if r == "M"} == set(prep.MEDIATORS)
    assert {n for n, r in roles.items() if r == "C"} == set(prep.COLLIDERS)


def test_dag_is_acyclic_and_dot_marks_latent(dag):
    import networkx as nx

    assert nx.is_directed_acyclic_graph(dag)
    dot = causal.dag_to_dot(dag)
    assert f'"{causal.LATENT}" -> "{causal.TREATMENT}" [style=dashed]' in dot


def test_backdoor_satisfied_by_confounders_without_latent():
    g = causal.build_dag(latent=None)
    assert causal.check_backdoor(g, causal.TREATMENT, causal.OUTCOME, prep.CONFOUNDERS)["satisfied"]


def test_latent_confounder_breaks_identification(dag):
    res = causal.check_backdoor(dag, causal.TREATMENT, causal.OUTCOME, prep.CONFOUNDERS)
    assert not res["blocks_all_backdoor_paths"] and causal.LATENT in res["latent_nodes"]


@pytest.mark.parametrize("extra", [prep.MEDIATORS, prep.COLLIDERS])
def test_mediators_and_colliders_violate_backdoor(extra):
    g = causal.build_dag(latent=None)
    res = causal.check_backdoor(g, causal.TREATMENT, causal.OUTCOME, [*prep.CONFOUNDERS, *extra])
    assert not res["no_descendants_of_T"] and set(res["descendants_in_set"]) == set(extra)


def test_backdoor_paths_enter_treatment():
    g = causal.build_dag(latent=None)
    paths = causal.backdoor_paths(g)
    assert [causal.TREATMENT, "Tenure", causal.OUTCOME] in paths
    assert all(g.has_edge(p[1], causal.TREATMENT) for p in paths)


def test_collider_simulation_shows_berkson_bias():
    sim = causal.collider_simulation()
    assert abs(sim["estimate_without_C"] - sim["true_effect"]) < 0.05
    assert abs(sim["estimate_with_C"] - sim["true_effect"]) > 0.2


# --------------------------------------------------------------------------- estimadores
def test_prepare_frame_drops_missing_treatment_and_imputes(frame, clean_df):
    f, info = frame
    assert len(f) == clean_df[causal.TREATMENT].notna().sum()
    assert not f[info["z_cols"]].isna().any().any()
    assert not set(info["z_cols"]) & {*prep.MEDIATORS, *prep.COLLIDERS, causal.TREATMENT, causal.OUTCOME}


def test_adjusted_regression_recovers_true_ate_and_naive_is_biased(frame):
    f, info = frame
    adjusted = causal._ols_effect(f, info["z_cols"])
    naive = causal.naive_effect(f)
    assert abs(adjusted["ate"] - TRUE_ATE) < 0.04
    assert abs(naive["ate"] - TRUE_ATE) > abs(adjusted["ate"] - TRUE_ATE)
    assert naive["ate"] == pytest.approx(naive["churn_rate_T1"] - naive["churn_rate_T0"])


def test_positivity_violation_detected_and_removed(frame, clean_df):
    violations = causal.positivity_violations(clean_df)
    assert {"variable": "HourSpendOnApp", "nivel": "4.0"}.items() <= violations.iloc[0].to_dict().items()
    f, info = frame
    support, ps, summary = causal.common_support(f, clean_df, info)
    assert summary["n_excluidos_estratos"] == int((clean_df["HourSpendOnApp"] == 4).sum())
    assert ((ps >= causal.PS_BOUNDS[0]) & (ps <= causal.PS_BOUNDS[1])).all()


def test_ipw_recovers_effect_and_improves_balance(frame, clean_df):
    f, info = frame
    support, ps, _ = causal.common_support(f, clean_df, info)
    assert abs(causal.ipw_ate(support, ps) - TRUE_ATE) < 0.05
    bal = causal.balance_table(support, ps, info["z_cols"])
    assert bal.loc["Tenure", "smd_despues"] ** 2 < bal.loc["Tenure", "smd_antes"] ** 2


def test_dowhy_pipeline_and_refutations(frame, clean_df):
    f, info = frame
    support, _, _ = causal.common_support(f, clean_df, info)
    z = [c for c in info["z_cols"] if support[c].nunique() > 1]
    model, estimand, est = causal.dowhy_pipeline(support, causal.encoded_graph(z))
    assert set(estimand.get_backdoor_variables()) == set(z)
    assert abs(est["linear_regression"].value - TRUE_ATE) < 0.04
    refs = causal.run_refutations(model, estimand, est["linear_regression"], num_simulations=10)
    assert [r["refutador"] for r in refs] == ["Random Common Cause", "Placebo Treatment", "Data Subset (80%)"]
    assert {r["semaforo"] for r in refs} <= {"verde", "amarillo", "rojo"}
    placebo = next(r for r in refs if r["refutador"] == "Placebo Treatment")
    assert abs(placebo["ate_nuevo"]) < abs(TRUE_ATE) / 2


# --------------------------------------------------------------------------- artefactos reales
@pytest.mark.skipif(not causal.RESULTS_PATH.exists(), reason="artifacts/causal_results.json aún no generado")
def test_saved_causal_artifacts_follow_contract():
    p = json.loads(causal.RESULTS_PATH.read_text(encoding="utf-8"))
    for key in ("estimates", "refutations", "overlap", "balance", "cate_subgroups", "policy", "backdoor"):
        assert key in p
    assert p["primary_estimate"] in p["estimates"]
    assert set(p["dowhy_backdoor_set"]).isdisjoint({*prep.MEDIATORS, *prep.COLLIDERS})
    assert len(p["refutations"]) == 3
    grid = pd.read_parquet(causal.CATE_GRID_PATH)
    assert len(grid) == 32 * 2 * 5
    assert {"cate_linear_dml", "ci_inf_causal_forest", *causal.EFFECT_MODIFIERS} <= set(grid.columns)
    assert causal.TREATMENT in causal.DAG_PATH.read_text(encoding="utf-8")

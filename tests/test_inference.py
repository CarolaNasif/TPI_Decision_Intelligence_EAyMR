"""Tests unitarios y de integración de Fase 1 (src/inference.py y missingness_report)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import RANDOM_STATE
from src import inference as inf
from src import preprocessing as prep


# --------------------------------------------------------------------------- Cohen's d y 2 muestras
def test_cohens_d_known_effect_and_antisymmetry():
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.normal(loc=10.0, scale=2.0, size=1500)
    y = rng.normal(loc=12.0, scale=2.0, size=1500)
    # Inyectar NA para comprobar limpieza automática
    x_na = np.append(x, [np.nan, np.nan])
    d_xy = inf.cohens_d(x_na, y)
    d_yx = inf.cohens_d(y, x_na)
    assert d_xy == pytest.approx(-1.0, abs=0.1)
    assert d_xy == pytest.approx(-d_yx, rel=1e-12)


def test_two_sample_test_welch_mannwhitney_ic_and_power():
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.normal(loc=5.5, scale=1.5, size=300)
    y = rng.normal(loc=3.0, scale=2.5, size=300)  # diferencia real = +2.5, heterocedástico
    res = inf.two_sample_test(x, y, alpha=0.05, n_boot=1000)

    assert res["ci_low"] < 2.5 < res["ci_high"]
    assert res["boot_ci_low"] < 2.5 < res["boot_ci_high"]
    assert res["p_value"] < 1e-10
    assert res["p_mannwhitney"] < 1e-10
    assert res["cohens_d"] > 0.8
    assert res["effect_magnitude"] == "grande"
    assert res["power_observed"] > 0.99
    assert 0.0 < res["mde_d_80"] < 0.3
    assert res["reject_h0"] is True
    assert res["decision"] == "Se rechaza H0"
    assert "acepta" not in res["decision"].lower()


def test_two_sample_test_null_never_says_accepts_h0():
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.normal(loc=0.0, scale=1.0, size=120)
    y = rng.normal(loc=0.0, scale=1.0, size=120)
    res = inf.two_sample_test(x, y, alpha=0.05, n_boot=500)
    assert res["reject_h0"] is False
    assert res["decision"] == "No se rechaza H0"
    assert "acepta" not in res["decision"].lower()


# --------------------------------------------------------------------------- Chi-cuadrado
def test_chi2_independence_null_and_associated():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1000
    a = rng.integers(0, 2, size=n)
    b_indep = rng.integers(0, 2, size=n)
    b_dep = np.where(rng.random(n) < np.where(a == 1, 0.7, 0.2), 1, 0)
    df = pd.DataFrame({"a": a, "b_indep": b_indep, "b_dep": b_dep})

    res_null = inf.chi2_independence(df, "a", "b_indep")
    assert res_null["cramers_v"] < 0.1
    assert res_null["reject_h0"] is False
    assert res_null["decision"] == "No se rechaza H0"

    res_dep = inf.chi2_independence(df, "a", "b_dep")
    assert res_dep["p_value"] < 1e-10
    assert res_dep["cramers_v"] > 0.3
    assert res_dep["prop_ci_low"] < res_dep["diff_proportions"] < res_dep["prop_ci_high"]
    assert res_dep["odds_ratio"] > 3.0
    assert res_dep["reject_h0"] is True
    assert res_dep["decision"] == "Se rechaza H0"


# --------------------------------------------------------------------------- Normalidad
def test_normality_diagnostics_shape_and_shapiro_hypersensitivity():
    rng = np.random.default_rng(RANDOM_STATE)
    normal = rng.normal(loc=0.0, scale=1.0, size=2000)
    nd_norm = inf.normality_diagnostics(normal)
    assert abs(nd_norm["skewness"]) < 0.2
    assert abs(nd_norm["kurtosis_excess"]) < 0.3
    assert nd_norm["is_acceptable_shape"] is True
    assert len(nd_norm["qq_theoretical"]) == len(nd_norm["qq_sample"]) == 2000

    # Muestra grande con desvío mínimo (t con 20 gl): Shapiro rechaza pero forma sigue aceptable
    near_normal_large = rng.standard_t(df=20, size=4500)
    nd_large = inf.normality_diagnostics(near_normal_large)
    assert nd_large["shapiro_p"] < 0.05
    assert nd_large["is_acceptable_shape"] is True

    # Lognormal asimétrica: forma rechazada
    skewed = rng.lognormal(mean=0.0, sigma=1.1, size=1500)
    nd_skew = inf.normality_diagnostics(skewed)
    assert nd_skew["skewness"] > 1.0
    assert nd_skew["is_acceptable_shape"] is False


# --------------------------------------------------------------------------- ANOVA y Post-hoc
def test_one_way_anova_suite_decision_tree():
    rng = np.random.default_rng(RANDOM_STATE)

    # Caso 1: simétrico pero heterocedástico con diferencias -> Welch ANOVA + Games-Howell
    df_het = pd.DataFrame(
        {
            "g": np.repeat(["G1", "G2", "G3"], 200),
            "v": np.concatenate(
                [
                    rng.normal(10.0, 1.0, 200),
                    rng.normal(12.0, 4.0, 200),
                    rng.normal(15.0, 8.0, 200),
                ]
            ),
        }
    )
    res_het = inf.one_way_anova_suite(df_het, "v", "g")
    assert res_het["homoscedastic"] is False
    assert res_het["all_acceptable_shape"] is True
    assert res_het["recommended_test"] == "Welch ANOVA"
    assert res_het["recommended_posthoc"] == "games-howell"
    assert res_het["reject_h0"] is True
    assert res_het["eta_sq"] > 0.05
    assert res_het["omega_sq"] > 0.05
    assert len(res_het["summary_table"]) == 3

    # Caso 2: asimétrico -> Kruskal-Wallis + Bonferroni
    df_skew = pd.DataFrame(
        {
            "g": np.repeat(["G1", "G2", "G3"], 200),
            "v": np.concatenate(
                [
                    rng.lognormal(0.0, 1.2, 200),
                    rng.lognormal(0.5, 1.2, 200),
                    rng.lognormal(1.2, 1.2, 200),
                ]
            ),
        }
    )
    res_skew = inf.one_way_anova_suite(df_skew, "v", "g")
    assert res_skew["all_acceptable_shape"] is False
    assert res_skew["recommended_test"] == "Kruskal-Wallis"
    assert res_skew["recommended_posthoc"] == "bonferroni"

    # Caso 3: homocedástico y simétrico sin diferencias -> ANOVA clásico + No se rechaza H0
    df_null = pd.DataFrame(
        {"g": np.repeat(["G1", "G2", "G3"], 150), "v": rng.normal(0.0, 1.0, 450)}
    )
    res_null = inf.one_way_anova_suite(df_null, "v", "g")
    assert res_null["homoscedastic"] is True
    assert res_null["recommended_test"] == "ANOVA clásico"
    assert res_null["recommended_posthoc"] == "tukey"
    assert res_null["decision"] == "No se rechaza H0"
    assert not res_null["summary_table"]["decision"].str.contains("acepta", case=False).any()


@pytest.mark.parametrize("method", ["games-howell", "tukey", "bonferroni"])
def test_posthoc_methods_and_fwer(method):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame(
        {
            "g": np.repeat(["A", "B", "C"], 120),
            "v": np.concatenate(
                [rng.normal(0.0, 1.0, 120), rng.normal(2.0, 1.2, 120), rng.normal(4.0, 1.5, 120)]
            ),
        }
    )
    res = inf.posthoc(df, "v", "g", method=method)
    assert len(res) == 3
    assert set(["A", "B", "diff", "p_adj", "ci_low", "ci_high", "reject_h0", "decision"]).issubset(
        res.columns
    )
    assert (res["ci_low"] < res["diff"]).all() and (res["diff"] < res["ci_high"]).all()
    assert res["reject_h0"].all()
    assert res.attrs["fwer_info"]["n_comparisons"] == 3
    assert res.attrs["fwer_info"]["fwer_uncorrected"] == pytest.approx(1.0 - 0.95**3)


def test_posthoc_invalid_method_raises():
    df = pd.DataFrame({"g": ["A", "A", "B", "B"], "v": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="Método post-hoc desconocido"):
        inf.posthoc(df, "v", "g", method="desconocido")


# --------------------------------------------------------------------------- Missingness
def test_missingness_report_detects_mar_and_little_mcar():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1200
    x_obs = rng.normal(size=n)
    x_obs2 = rng.normal(size=n)
    churn = (rng.random(n) < 0.2).astype(int)

    # x_mar falta sistemáticamente cuando x_obs > 0.7 (MAR fuerte)
    x_mar = rng.normal(loc=5.0, scale=1.0, size=n)
    p_miss_mar = np.where(x_obs > 0.7, 0.45, 0.02)
    x_mar[rng.random(n) < p_miss_mar] = np.nan

    # x_mcar falta aleatoriamente 5%
    x_mcar = rng.normal(size=n)
    x_mcar[rng.random(n) < 0.05] = np.nan

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Churn": churn,
            "x_obs": x_obs,
            "x_obs2": x_obs2,
            "x_mar": x_mar,
            "x_mcar": x_mcar,
        }
    )
    rep = prep.missingness_report(df)
    row_mar = rep.loc[rep["column"] == "x_mar"].iloc[0]
    row_obs = rep.loc[rep["column"] == "x_obs"].iloc[0]

    assert row_mar["n_missing"] > 50
    assert row_mar["mar_logit_auc"] > 0.70
    assert row_mar["mar_logit_p_lrt"] < 0.001
    assert "MAR" in row_mar["mechanism"]
    assert row_obs["n_missing"] == 0
    assert row_obs["mechanism"] == "Completo (0% NA)"
    assert rep.attrs["little_mcar"]["df"] > 0
    assert rep.attrs["little_mcar"]["p_value"] < 0.05


# --------------------------------------------------------------------------- Fronteras de rol (Fase 2)
def test_fase2_functions_remain_not_implemented_for_qa():
    with pytest.raises(NotImplementedError):
        prep.fit_pca_train_only(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(NotImplementedError):
        inf.fit_ols(pd.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]}), "y ~ x")
    with pytest.raises(NotImplementedError):
        inf.gauss_markov_diagnostics(None)
    with pytest.raises(NotImplementedError):
        inf.influence_diagnostics(None)
    with pytest.raises(NotImplementedError):
        inf.vif_table(pd.DataFrame({"a": [1.0, 2.0]}))


# --------------------------------------------------------------------------- Integración dataset real
@pytest.mark.skipif(not prep.RAW_PATH.exists(), reason="data/raw/E Commerce Dataset.xlsx no descargado")
def test_real_dataset_fase1_integration():
    df = prep.make_clean_dataset()
    miss = prep.missingness_report(df)
    assert miss.attrs["little_mcar"]["p_value"] < 0.05
    assert (miss["n_missing"] > 0).sum() == 8  # 7 originales + coupon_any derivada

    c1 = inf.two_sample_test(
        df.loc[df["Churn"] == 1, "CashbackAmount"],
        df.loc[df["Churn"] == 0, "CashbackAmount"],
        n_boot=500,
    )
    assert c1["p_value"] < 1e-10
    assert c1["power_observed"] > 0.99

    c2 = inf.chi2_independence(df, "Complain", "Churn")
    assert c2["p_value"] < 1e-10
    assert c2["cramers_v"] > 0.2

    aov = inf.one_way_anova_suite(df, "CashbackAmount", "PreferedOrderCat")
    assert aov["welch_p"] < 1e-10
    assert aov["kruskal_p"] < 1e-10
    assert aov["omega_sq"] > 0.8

    ph = inf.posthoc(df, "CashbackAmount", "PreferedOrderCat", method="games-howell")
    assert len(ph) == 10
    assert ph["reject_h0"].all()

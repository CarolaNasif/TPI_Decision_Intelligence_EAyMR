"""Quality Gate determinístico del TPI (dueño: QA & Validation Engineer).

Veredicto: el PR recibe [APROBADO] solo si `ruff check .` y `pytest -q` pasan en verde.

Bloques:
    A. Integridad del split (proporciones, disjunción, estratificación, reproducibilidad).
    B. No-leakage en preprocesamiento/PCA (fit solo en Train).
    C. Integridad del dataset real (N >= 1500, >= 10 explicativas) -> se activa al existir data/processed.
    D. Contrato causal (M y C fuera del set de ajuste) -> se activa al completar src/preprocessing.py.
    E. Higiene del paquete (los módulos de src importan sin efectos secundarios).

Los bloques A y B usan datos sintéticos y validan el PATRÓN que el equipo debe reutilizar.
Cuando src/preprocessing.py esté implementado, se agregan tests espejo sobre esas funciones.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed" / "dataset_clean.parquet"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Dataset sintético con N=2000, 12 explicativas, target desbalanceado (~17%)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    X = rng.lognormal(mean=0.0, sigma=1.0, size=(n, 12))  # colas pesadas a propósito
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    df["y"] = (rng.random(n) < 0.17).astype(int)
    return df


@pytest.fixture(scope="module")
def split(synthetic_df):
    X = synthetic_df.drop(columns="y")
    y = synthetic_df["y"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- A. split
def test_split_proportions_80_20(split, synthetic_df):
    X_tr, X_te, _, _ = split
    assert len(X_tr) + len(X_te) == len(synthetic_df)
    assert abs(len(X_te) / len(synthetic_df) - 0.20) < 0.01


def test_split_indices_disjoint(split):
    X_tr, X_te, _, _ = split
    assert set(X_tr.index).isdisjoint(set(X_te.index)), "Fila presente en Train y Test"


def test_split_stratification_preserved(split):
    _, _, y_tr, y_te = split
    assert abs(y_tr.mean() - y_te.mean()) < 0.02, "La prevalencia del target difiere entre splits"


def test_split_reproducible(synthetic_df):
    X, y = synthetic_df.drop(columns="y"), synthetic_df["y"]
    a = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)[1].index
    b = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)[1].index
    assert a.equals(b), "El split no es reproducible con la semilla fija"


# --------------------------------------------------------------------------- B. no-leakage
def test_scaler_fitted_only_on_train(split):
    X_tr, X_te, _, _ = split
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=RANDOM_STATE))])
    pipe.fit(X_tr)
    mu = pipe.named_steps["scaler"].mean_
    np.testing.assert_allclose(mu, X_tr.mean().to_numpy(), rtol=1e-10)
    full_mean = pd.concat([X_tr, X_te]).mean().to_numpy()
    assert not np.allclose(mu, full_mean), "El escalador vio datos de Test (leakage)"


def test_pca_fitted_only_on_train(split):
    X_tr, X_te, _, _ = split
    pipe_train = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=3))]).fit(X_tr)
    pipe_full = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=3))]).fit(
        pd.concat([X_tr, X_te])
    )
    assert pipe_train.named_steps["p"].n_samples_ == len(X_tr)
    # Los autovectores ajustados con Train y con Train+Test NO deben coincidir.
    assert not np.allclose(
        np.abs(pipe_train.named_steps["p"].components_),
        np.abs(pipe_full.named_steps["p"].components_),
    )


def test_pca_transform_test_shape_and_train_centering(split):
    X_tr, X_te, _, _ = split
    pipe = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=5))]).fit(X_tr)
    Z_tr, Z_te = pipe.transform(X_tr), pipe.transform(X_te)
    assert Z_te.shape == (len(X_te), 5)
    # En Train los scores están centrados exactamente; en Test (proyección ciega) no tiene por qué.
    np.testing.assert_allclose(Z_tr.mean(axis=0), 0.0, atol=1e-10)


def test_kaiser_criterion_is_computable(split):
    X_tr, _, _, _ = split
    pca = Pipeline([("s", StandardScaler()), ("p", PCA())]).fit(X_tr).named_steps["p"]
    eig = pca.explained_variance_
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)
    assert (eig > 1).sum() >= 1


# --------------------------------------------------------------------------- C. dataset real
@pytest.mark.skipif(not PROCESSED.exists(), reason="data/processed/dataset_clean.parquet aún no existe")
def test_real_dataset_dimensions():
    df = pd.read_parquet(PROCESSED)
    prep = importlib.import_module("src.preprocessing")
    excluded = {prep.TARGET_COL, *prep.ID_COLS}
    n_features = len([c for c in df.columns if c not in excluded])
    assert len(df) >= 1500, f"N={len(df)} < 1500 (requisito de la cátedra)"
    assert n_features >= 10, f"Solo {n_features} explicativas (< 10)"


@pytest.mark.skipif(not PROCESSED.exists(), reason="data/processed/dataset_clean.parquet aún no existe")
def test_real_dataset_has_mixed_types():
    df = pd.read_parquet(PROCESSED)
    n_cat = df.select_dtypes(include=["object", "category", "bool"]).shape[1]
    n_num = df.select_dtypes(include="number").shape[1]
    assert n_cat >= 1 and n_num >= 1, "Se exige combinación de categóricas y numéricas"


# --------------------------------------------------------------------------- D. contrato causal
def _causal_contract():
    prep = importlib.import_module("src.preprocessing")
    if not prep.TREATMENT_COL:
        pytest.skip("Contrato causal todavía no definido en src/preprocessing.py")
    return prep


def test_adjustment_set_excludes_mediators_and_colliders():
    prep = _causal_contract()
    bad = set(prep.CONFOUNDERS) & (set(prep.MEDIATORS) | set(prep.COLLIDERS))
    assert not bad, f"Variables post-tratamiento en el set de ajuste: {bad}"


def test_treatment_and_outcome_not_in_confounders():
    prep = _causal_contract()
    assert prep.TREATMENT_COL not in prep.CONFOUNDERS
    assert prep.TARGET_COL not in prep.CONFOUNDERS


# --------------------------------------------------------------------------- E. higiene
@pytest.mark.parametrize("module", ["src", "src.preprocessing", "src.inference", "src.models", "src.causal"])
def test_src_modules_import_cleanly(module):
    assert importlib.import_module(module) is not None

"""Tests espejo de src/preprocessing.py (contrato de datos, limpieza, split y preprocesador)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fake_raw() -> pd.DataFrame:
    """Dataset con el esquema del original: N=2000, ~17% churn, NA y categorías duplicadas."""
    rng = np.random.default_rng(prep.RANDOM_STATE)
    n = 2000
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + n),
            "Churn": (rng.random(n) < 0.17).astype(int),
            "PreferredLoginDevice": rng.choice(["Mobile Phone", "Phone", "Computer"], n),
            "CityTier": rng.choice([1, 2, 3], n),
            "PreferredPaymentMode": rng.choice(["Debit Card", "CC", "Credit Card", "COD", "UPI"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "PreferedOrderCat": rng.choice(["Laptop & Accessory", "Mobile", "Mobile Phone", "Grocery"], n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        }
    )
    for col in prep.NUMERIC_COLS:
        df[col] = rng.lognormal(1.0, 0.8, n).round(2)
    df["Complain"] = rng.integers(0, 2, n)
    df["CouponUsed"] = rng.integers(0, 4, n).astype(float)
    df.loc[rng.choice(n, 100, replace=False), "CouponUsed"] = np.nan
    df.loc[rng.choice(n, 90, replace=False), "Tenure"] = np.nan
    return df[prep.EXPECTED_COLUMNS]


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory, fake_raw) -> str:
    dup = fake_raw.iloc[:30].copy()
    dup["CustomerID"] += 10_000  # mismo perfil, otro ID (como en el archivo original)
    path = tmp_path_factory.mktemp("raw") / "fake.csv"
    pd.concat([fake_raw, dup]).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def clean(raw_csv) -> pd.DataFrame:
    return prep.make_clean_dataset(raw_csv)


# --------------------------------------------------------------------------- contrato
def test_contract_columns_exist_in_schema():
    roles = {prep.TARGET_COL, prep.TREATMENT_SOURCE_COL, *prep.CONFOUNDERS, *prep.MEDIATORS, *prep.COLLIDERS}
    assert roles <= set(prep.EXPECTED_COLUMNS)


def test_contract_roles_are_disjoint():
    groups = [set(prep.CONFOUNDERS), set(prep.MEDIATORS), set(prep.COLLIDERS)]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not a & b
    assert prep.TREATMENT_SOURCE_COL not in prep.CONFOUNDERS


def test_numeric_and_categorical_partition_the_features():
    num, cat = set(prep.NUMERIC_COLS), set(prep.CATEGORICAL_COLS)
    assert not num & cat
    assert num | cat == set(prep.EXPECTED_COLUMNS) - {prep.TARGET_COL, *prep.ID_COLS}


# --------------------------------------------------------------------------- limpieza
def test_categories_are_normalized(clean):
    for col, mapping in prep.CATEGORY_FIXES.items():
        assert not set(mapping) & set(clean[col].astype(str).unique()), f"Quedan etiquetas viejas en {col}"


def test_duplicate_profiles_removed(clean, fake_raw):
    assert len(clean) == len(fake_raw)
    assert not prep.duplicate_profiles(clean.drop(columns=prep.TREATMENT_COL)).any()


def test_duplicates_kept_when_requested(raw_csv, fake_raw):
    assert len(prep.load_raw_data(raw_csv, drop_duplicates=False)) == len(fake_raw) + 30


def test_treatment_definition_and_na_preserved(clean):
    src, t = clean[prep.TREATMENT_SOURCE_COL], clean[prep.TREATMENT_COL]
    assert t.isna().equals(src.isna()), "Los NA de CouponUsed deben seguir siendo NA en T"
    observed = src.notna()
    assert (t[observed] == (src[observed] >= 1).astype(int)).all()


def test_clean_dataset_is_not_imputed(clean):
    assert clean["Tenure"].isna().sum() == 90


def test_load_rejects_missing_columns(tmp_path, fake_raw):
    path = tmp_path / "bad.csv"
    fake_raw.drop(columns="Complain").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Columnas faltantes"):
        prep.load_raw_data(path)


# --------------------------------------------------------------------------- split
def test_split_is_stratified_disjoint_and_80_20(clean):
    X_tr, X_te, y_tr, y_te = prep.split_train_test(clean)
    assert abs(len(X_te) / len(clean) - 0.2) < 0.01
    assert set(X_tr["CustomerID"]).isdisjoint(X_te["CustomerID"])
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert prep.TARGET_COL not in X_tr.columns


def test_split_roundtrip_through_json(clean, tmp_path):
    X_tr, X_te, y_tr, y_te = prep.split_train_test(clean)
    path = prep.save_split(X_tr, X_te, tmp_path / "split.json", random_state=prep.RANDOM_STATE)
    X_tr2, X_te2, y_tr2, y_te2 = prep.load_split(clean, path)
    assert set(X_tr2["CustomerID"]) == set(X_tr["CustomerID"])
    assert set(X_te2["CustomerID"]) == set(X_te["CustomerID"])
    assert y_tr2.sum() == y_tr.sum()


# --------------------------------------------------------------------------- preprocesador
def test_preprocessor_fits_only_on_train(clean):
    X_tr, X_te, _, _ = prep.split_train_test(clean)
    pre = prep.build_preprocessor(prep.NUMERIC_COLS, prep.CATEGORICAL_COLS).fit(X_tr)
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_, X_tr[prep.NUMERIC_COLS].median().to_numpy())
    Z_te = pre.transform(X_te)
    assert not np.isnan(Z_te).any()
    assert Z_te.shape[1] == len(pre.get_feature_names_out())


def test_preprocessor_handles_unseen_category(clean):
    X_tr, X_te, _, _ = prep.split_train_test(clean)
    pre = prep.build_preprocessor(prep.NUMERIC_COLS, prep.CATEGORICAL_COLS).fit(X_tr)
    X_new = X_te.head(3).copy()
    X_new["Gender"] = X_new["Gender"].astype(str)
    X_new.loc[X_new.index[0], "Gender"] = "Other"
    assert pre.transform(X_new).shape[0] == 3


# --------------------------------------------------------------------------- dataset real
@pytest.mark.skipif(not prep.RAW_PATH.exists(), reason="data/raw/E Commerce Dataset.xlsx no descargado")
def test_real_raw_dataset_cleaning():
    df = prep.make_clean_dataset()
    assert len(df) == 5073
    assert not prep.duplicate_profiles(df.drop(columns=prep.TREATMENT_COL)).any()
    assert abs(df[prep.TARGET_COL].mean() - 0.1658) < 0.001

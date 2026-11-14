"""Preprocesamiento y contrato de datos (dataset A: E-Commerce Customer Churn, Kaggle).

Dueño: Backend & ML Engineer. Revisor obligatorio: QA & Validation Engineer.

Contrato de datos (ver docs/01_propuestas_datasets_y_dags.md, Opción A):
    - TARGET_COL      : Y = "Churn" (1 = el cliente abandonó)
    - TREATMENT_COL   : T = "coupon_any" = 1{CouponUsed >= 1} (NA si CouponUsed es NA)
    - CONFOUNDERS     : Z, solo variables PRE-tratamiento (set de ajuste backdoor)
    - MEDIATORS       : M, post-tratamiento; NUNCA entran al set de ajuste causal
    - COLLIDERS       : C, descendientes de T e Y; NUNCA se condiciona sobre ellas
    - ID_COLS         : identificadores que se descartan del modelado

Reglas anti-leakage (verificadas en tests/):
    1. El split train/test (80/20, estratificado, RANDOM_STATE) ocurre ANTES de
       cualquier imputación, escalado, encoding o PCA.
    2. Todo transformador se ajusta (fit) solo sobre Train y se aplica (transform) a Test.
    3. Se usa sklearn.pipeline.Pipeline / ColumnTransformer para garantizar (1) y (2) en CV.

Uso: `python -m src.preprocessing` genera data/processed/dataset_clean.parquet (SIN imputar)
y artifacts/split_idx.json (IDs de Train/Test).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
RAW_SHEET = "E Comm"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_clean.parquet"
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "split_idx.json"

# --------------------------------------------------------------------------- contrato de datos
ID_COLS: list[str] = ["CustomerID"]
TARGET_COL: str = "Churn"
TREATMENT_SOURCE_COL: str = "CouponUsed"
TREATMENT_COL: str = "coupon_any"
CONFOUNDERS: list[str] = [
    "Tenure",
    "Complain",
    "SatisfactionScore",
    "CityTier",
    "PreferedOrderCat",
    "HourSpendOnApp",
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "MaritalStatus",
    "WarehouseToHome",
    "NumberOfAddress",
    "NumberOfDeviceRegistered",
]
MEDIATORS: list[str] = ["OrderCount", "CashbackAmount"]
COLLIDERS: list[str] = ["DaySinceLastOrder"]

# --------------------------------------------------------------------------- tipado
# CityTier es ordinal (1-3) pero se trata como nominal para no imponer distancias iguales.
CATEGORICAL_COLS: list[str] = [
    "PreferredLoginDevice",
    "CityTier",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]
NUMERIC_COLS: list[str] = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]
EXPECTED_COLUMNS: list[str] = [*ID_COLS, TARGET_COL, *CATEGORICAL_COLS, *NUMERIC_COLS]

# Categorías duplicadas del archivo original -> nombre canónico.
CATEGORY_FIXES: dict[str, dict[str, str]] = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


# --------------------------------------------------------------------------- carga y limpieza
def file_sha256(path: str | Path) -> str:
    """Hash SHA-256 del archivo crudo (trazabilidad, Bitácora Entrada 1)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica etiquetas duplicadas según CATEGORY_FIXES (devuelve una copia)."""
    out = df.copy()
    for col, mapping in CATEGORY_FIXES.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)
    return out


def duplicate_profiles(df: pd.DataFrame) -> pd.Series:
    """Máscara de filas idénticas a otra en todas las columnas salvo los IDs (se conserva la 1ra)."""
    cols = [c for c in df.columns if c not in ID_COLS]
    return df.duplicated(subset=cols, keep="first")


def load_raw_data(
    path: str | Path = RAW_PATH, sheet: str = RAW_SHEET, drop_duplicates: bool = True
) -> pd.DataFrame:
    """Carga el dataset crudo, normaliza categorías y elimina perfiles duplicados.

    Duplicados: el archivo original trae 556 pares de filas idénticas salvo el CustomerID
    (IDs separados por 1470), y 557 tras unificar categorías. Son copias del proceso de
    generación del dataset, no clientes distintos; dejarlas permitiría que el mismo cliente
    caiga en Train y en Test (leakage).
    `drop_duplicates=False` conserva el dataset original para análisis de sensibilidad.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset crudo: {sorted(missing)}")
    if df[ID_COLS].duplicated().any():
        raise ValueError("Hay CustomerID repetidos en el dataset crudo")

    df = normalize_categories(df[EXPECTED_COLUMNS])
    if drop_duplicates:
        df = df.loc[~duplicate_profiles(df)]
    df = df.reset_index(drop=True)

    n_features = len(df.columns) - len(ID_COLS) - 1
    if len(df) < 1500 or n_features < 10:
        raise ValueError(f"No cumple requisitos de la cátedra: N={len(df)}, explicativas={n_features}")
    return df


def add_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega T = coupon_any = 1{CouponUsed >= 1}, como entero nullable (NA si CouponUsed es NA).

    Los NA de T NO se imputan aquí: el análisis causal decide cómo tratarlos (Fase 4).
    """
    out = df.copy()
    src = out[TREATMENT_SOURCE_COL]
    out[TREATMENT_COL] = (src >= 1).astype("Int64").mask(src.isna())
    return out


def make_clean_dataset(path: str | Path = RAW_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Carga + limpieza + tratamiento. Sin imputar: la imputación va dentro del Pipeline."""
    df = add_treatment(load_raw_data(path, drop_duplicates=drop_duplicates))
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def cleaning_summary(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Métricas antes/después de la limpieza (insumo para la auditoría de Fase 1)."""
    raw = pd.read_excel(path, sheet_name=RAW_SHEET) if Path(path).suffix == ".xlsx" else pd.read_csv(path)
    clean = load_raw_data(path)
    rows = [
        ("Filas", len(raw), len(clean)),
        ("Tasa de churn", round(raw[TARGET_COL].mean(), 4), round(clean[TARGET_COL].mean(), 4)),
        ("Perfiles duplicados", int(duplicate_profiles(raw).sum()), int(duplicate_profiles(clean).sum())),
    ]
    for col in CATEGORY_FIXES:
        rows.append((f"Categorías en {col}", raw[col].nunique(), clean[col].nunique()))
    return pd.DataFrame(rows, columns=["métrica", "antes", "después"]).set_index("métrica")


def _little_mcar_test(df_num: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Test global MCAR de Little (1988) bajo normalidad multivariada (EM)."""
    clean = df_num.dropna(axis=1, how="all").copy()
    valid_cols = [c for c in clean.columns if clean[c].std(ddof=1) > 0]
    clean = clean[valid_cols]
    n_rows, p = clean.shape
    if p < 2 or int(clean.isna().sum().sum()) == 0:
        return {
            "stat": 0.0,
            "df": 0,
            "p_value": 1.0,
            "n_patterns": 1,
            "decision": "Sin faltantes suficientes para test de Little",
        }

    arr = clean.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0)
    sigma = clean.cov().fillna(0.0).to_numpy(dtype=float)
    sigma = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(p)

    masks = np.isnan(arr)
    unique_patterns, inverse = np.unique(masks, axis=0, return_inverse=True)
    pattern_indices = [np.flatnonzero(inverse == g) for g in range(len(unique_patterns))]

    for _ in range(60):
        sum_x = np.zeros(p, dtype=float)
        sum_xx = np.zeros((p, p), dtype=float)
        for pat, idx in zip(unique_patterns, pattern_indices, strict=True):
            obs = np.flatnonzero(~pat)
            mis = np.flatnonzero(pat)
            rows = arr[idx]
            n_g = len(idx)
            if len(obs) == 0:
                sum_x += n_g * mu
                sum_xx += n_g * (sigma + np.outer(mu, mu))
                continue
            if len(mis) == 0:
                sum_x += rows.sum(axis=0)
                sum_xx += rows.T @ rows
                continue

            s_oo = sigma[np.ix_(obs, obs)] + 1e-8 * np.eye(len(obs))
            s_mo = sigma[np.ix_(mis, obs)]
            s_mm = sigma[np.ix_(mis, mis)]
            beta = s_mo @ np.linalg.inv(s_oo)

            filled = rows.copy()
            filled[:, mis] = mu[mis] + (rows[:, obs] - mu[obs]) @ beta.T
            sum_x += filled.sum(axis=0)
            sum_xx += filled.T @ filled
            sum_xx[np.ix_(mis, mis)] += n_g * (s_mm - beta @ s_mo.T)

        mu_new = sum_x / n_rows
        sigma_new = (sum_xx / n_rows) - np.outer(mu_new, mu_new)
        sigma_new = 0.5 * (sigma_new + sigma_new.T) + 1e-6 * np.eye(p)
        if np.max(np.abs(mu_new - mu)) < 1e-6:
            mu, sigma = mu_new, sigma_new
            break
        mu, sigma = mu_new, sigma_new

    d2 = 0.0
    total_obs_dim = 0
    for pat, idx in zip(unique_patterns, pattern_indices, strict=True):
        obs = np.flatnonzero(~pat)
        if len(obs) == 0:
            continue
        rows = arr[idx][:, obs]
        diff_g = rows.mean(axis=0) - mu[obs]
        s_oo = sigma[np.ix_(obs, obs)] + 1e-8 * np.eye(len(obs))
        d2 += float(len(idx) * (diff_g @ np.linalg.solve(s_oo, diff_g)))
        total_obs_dim += len(obs)

    df_little = total_obs_dim - p
    if df_little <= 0:
        return {
            "stat": float(d2),
            "df": 0,
            "p_value": float("nan"),
            "n_patterns": len(unique_patterns),
            "decision": "Grados de libertad insuficientes para test de Little",
        }
    p_val = float(stats.chi2.sf(d2, df_little))
    decision = (
        "Se rechaza H0 (no es MCAR)" if p_val < alpha else "No se rechaza H0 (compatible con MCAR)"
    )
    return {
        "stat": float(d2),
        "df": int(df_little),
        "p_value": p_val,
        "n_patterns": len(unique_patterns),
        "decision": decision,
    }


def _build_mar_design_matrix(df: pd.DataFrame, cov_cols: list[str]) -> pd.DataFrame:
    """Construye matriz de diseño numérica estandarizada + dummies para logit de NA."""
    parts: list[pd.DataFrame] = []
    for c in cov_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = s.astype(float)
            if vals.isna().any():
                vals = vals.fillna(vals.median())
            std = float(vals.std(ddof=0))
            if std > 1e-12:
                parts.append(pd.DataFrame({c: (vals - float(vals.mean())) / std}, index=df.index))
        else:
            vals = s.astype("object")
            if vals.isna().any():
                mode = vals.dropna().mode()
                vals = vals.fillna(mode.iloc[0] if len(mode) else "MISSING")
            dummies = pd.get_dummies(vals.astype(str), prefix=c, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                parts.append(dummies)
    if not parts:
        return pd.DataFrame(index=df.index)
    X_mat = pd.concat(parts, axis=1)
    keep = [col for col in X_mat.columns if float(X_mat[col].std(ddof=0)) > 1e-12]
    return X_mat[keep]


def missingness_report(
    df: pd.DataFrame, target: str = TARGET_COL, alpha: float = 0.05
) -> pd.DataFrame:
    """Tabla de % de faltantes por columna + insumos para clasificar MCAR/MAR/MNAR.

    Incluye:
        - Conteo y % de NA por columna (excluyendo IDs).
        - Asociación chi-cuadrado y V de Cramér del indicador de NA contra `target`.
        - Regresión logística del indicador de NA sobre covariables observadas
          (p-valor LRT en statsmodels + ROC-AUC en CV estratificada como evidencia MAR).
        - Test global MCAR de Little sobre variables numéricas (guardado en `attrs["little_mcar"]`).
    """
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    excluded_from_little = {*ID_COLS, target, TREATMENT_COL}
    num_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in excluded_from_little and not pd.api.types.is_bool_dtype(df[c])
    ]
    little = _little_mcar_test(df[num_cols], alpha=alpha) if num_cols else {
        "stat": float("nan"),
        "df": 0,
        "p_value": float("nan"),
        "n_patterns": 0,
        "decision": "Sin variables numéricas",
    }

    inspect_cols = [c for c in df.columns if c not in ID_COLS]
    rows: list[dict[str, Any]] = []

    for col in inspect_cols:
        is_na = df[col].isna().astype(int)
        n_miss = int(is_na.sum())
        pct_miss = float(is_na.mean() * 100.0)

        chi2_tgt = float("nan")
        p_chi2_tgt = float("nan")
        v_tgt = float("nan")
        mar_p_lrt = float("nan")
        mar_auc = float("nan")

        if 0 < n_miss < len(df):
            if target in df.columns and col != target:
                valid = df[[col, target]].dropna(subset=[target])
                ct = pd.crosstab(valid[col].isna().astype(int), valid[target])
                if ct.shape == (2, 2):
                    chi2_val, p_val, _, _ = stats.chi2_contingency(ct, correction=False)
                    chi2_tgt = float(chi2_val)
                    p_chi2_tgt = float(p_val)
                    v_tgt = float(np.sqrt(chi2_val / len(valid)))

            paired_treatment = {TREATMENT_SOURCE_COL, TREATMENT_COL}
            exclude_covs = {*ID_COLS, target, col}
            if col in paired_treatment:
                exclude_covs |= paired_treatment

            cand_covs = [c for c in df.columns if c not in exclude_covs]
            obs_covs = [c for c in cand_covs if int(df[c].isna().sum()) == 0]
            cov_cols = obs_covs if obs_covs else [c for c in cand_covs if df[c].notna().any()]

            X_mar = _build_mar_design_matrix(df, cov_cols)
            if X_mar.shape[1] > 0:
                X_sm = sm.add_constant(X_mar, has_constant="add")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for method in ("bfgs", "newton"):
                        try:
                            res = sm.Logit(is_na, X_sm).fit(disp=False, method=method, maxiter=500)
                            if np.isfinite(res.llr_pvalue):
                                mar_p_lrt = float(res.llr_pvalue)
                                break
                        except Exception:  # noqa: BLE001
                            continue

                clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
                min_count = int(min(n_miss, len(df) - n_miss))
                if min_count >= 5:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    probs = cross_val_predict(clf, X_mar, is_na, cv=cv, method="predict_proba")[:, 1]
                else:
                    clf.fit(X_mar, is_na)
                    probs = clf.predict_proba(X_mar)[:, 1]
                mar_auc = float(roc_auc_score(is_na, probs))

            is_mar = (
                (pd.notna(mar_p_lrt) and mar_p_lrt < alpha)
                or (pd.notna(mar_auc) and mar_auc >= 0.60)
                or (pd.notna(p_chi2_tgt) and p_chi2_tgt < alpha)
            )
            mechanism = "MAR (posible componente MNAR por dominio)" if is_mar else "MCAR compatible"
            remediation = (
                "Imputación por mediana/moda + indicador binario de NA "
                "dentro del Pipeline (fit solo en Train)"
            )
        elif n_miss == len(df):
            mechanism = "100% faltante"
            remediation = "Descartar columna"
        else:
            mechanism = "Completo (0% NA)"
            remediation = "No requiere imputación"

        rows.append(
            {
                "column": col,
                "n_missing": n_miss,
                "pct_missing": pct_miss,
                "chi2_target": chi2_tgt,
                "p_chi2_target": p_chi2_tgt,
                "cramers_v_target": v_tgt,
                "mar_logit_p_lrt": mar_p_lrt,
                "mar_logit_auc": mar_auc,
                "mechanism": mechanism,
                "remediation": remediation,
                "little_mcar_stat": little["stat"],
                "little_mcar_p": little["p_value"],
            }
        )

    report = pd.DataFrame(rows).sort_values(
        by=["pct_missing", "column"], ascending=[False, True], kind="mergesort"
    )
    report = report.reset_index(drop=True)
    report.attrs["little_mcar"] = little
    return report


# --------------------------------------------------------------------------- split
def split_train_test(
    df: pd.DataFrame, target: str = TARGET_COL, test_size: float = 0.2, random_state: int = RANDOM_STATE
):
    """Split estratificado 80/20 sobre el target. Se ejecuta ANTES de imputar/escalar/PCA.

    Devuelve (X_train, X_test, y_train, y_test); X conserva todas las columnas salvo el target.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame, path: str | Path = SPLIT_PATH, **metadata
) -> Path:
    """Persiste los CustomerID de Train y Test para que todas las fases usen el MISMO split."""
    id_col = ID_COLS[0]
    payload = {
        "id_col": id_col,
        "train": sorted(int(i) for i in X_train[id_col]),
        "test": sorted(int(i) for i in X_test[id_col]),
        **metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def load_split(df: pd.DataFrame, path: str | Path = SPLIT_PATH, target: str = TARGET_COL):
    """Reconstruye (X_train, X_test, y_train, y_test) a partir de artifacts/split_idx.json."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    id_col = payload["id_col"]
    train = df[df[id_col].isin(payload["train"])]
    test = df[df[id_col].isin(payload["test"])]
    if len(train) + len(test) != len(df):
        raise ValueError("El split persistido no cubre exactamente el dataset actual")
    return train.drop(columns=[target]), test.drop(columns=[target]), train[target], test[target]


# --------------------------------------------------------------------------- preprocesador
def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
    """ColumnTransformer NO ajustado: para usar dentro de Pipelines y validación cruzada.

    - Numéricas: imputación por mediana + indicador de NA (preserva la señal MAR/MNAR) + escalado.
    - Categóricas: imputación por moda + One-Hot (categorías nuevas en Test -> vector de ceros).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, list(numeric_cols)), ("cat", categorical, list(categorical_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_pca_train_only(X_train_num, n_components=None):
    """Pipeline(StandardScaler -> PCA) ajustado SOLO en Train.

    Responsable: QA & Validation Engineer (Fase 2).
    TODO: devolver el pipeline ajustado + varianza explicada + loadings;
    luego aplicar .transform() a Test "a ciegas".
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- CLI
def main() -> None:
    df = make_clean_dataset(RAW_PATH)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    X_train, X_test, y_train, y_test = split_train_test(df)
    save_split(
        X_train,
        X_test,
        random_state=RANDOM_STATE,
        test_size=0.2,
        stratify=TARGET_COL,
        raw_sha256=file_sha256(RAW_PATH),
        n_rows=len(df),
    )
    print(cleaning_summary(RAW_PATH).to_string())
    print(f"\nTrain: {len(X_train)} (churn {y_train.mean():.4f})")
    print(f"Test:  {len(X_test)} (churn {y_test.mean():.4f})")
    t = df[TREATMENT_COL]
    print(f"T = {TREATMENT_COL}: {t.mean():.4f} tratados, {t.isna().sum()} NA")
    print(f"-> {PROCESSED_PATH.relative_to(PROJECT_ROOT)} | {SPLIT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

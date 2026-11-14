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
from pathlib import Path

import pandas as pd

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


def missingness_report(df):
    """Tabla de % de faltantes por columna + insumos para clasificar MCAR/MAR/MNAR.

    Responsable: Project Lead (Fase 1).
    TODO: % NA, patrón conjunto (missingno), test de Little (MCAR) y regresiones
    logísticas del indicador de NA sobre las demás variables (evidencia MAR).
    """
    raise NotImplementedError


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

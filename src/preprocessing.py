"""Preprocesamiento y contrato de datos (Fases 1-2).

ESQUELETO - todavía sin implementación (sprint de diseño 12/11/2026).
Dueño: Backend & ML Engineer. Revisor obligatorio: QA & Validation Engineer.

Contrato de datos (a completar cuando se elija el dataset):
    - TARGET_COL      : outcome Y (p.ej. "Churn")
    - TREATMENT_COL   : tratamiento T (p.ej. "coupon_any" = CouponUsed >= 1)
    - CONFOUNDERS     : lista Z (solo variables PRE-tratamiento)
    - MEDIATORS       : lista M (post-tratamiento; NUNCA entran al set de ajuste causal)
    - COLLIDERS       : lista C (descendientes de T e Y; NUNCA se condiciona sobre ellas)
    - ID_COLS         : identificadores que se descartan del modelado

Reglas anti-leakage (verificadas en tests/test_quality_gate.py):
    1. El split train/test (80/20, estratificado, RANDOM_STATE) ocurre ANTES de
       cualquier imputación, escalado, encoding o PCA.
    2. Todo transformador se ajusta (fit) solo sobre Train y se aplica (transform) a Test.
    3. Se usa sklearn.pipeline.Pipeline / ColumnTransformer para garantizar (1) y (2) en CV.
"""

from __future__ import annotations

# TODO(Backend): definir constantes del contrato de datos una vez elegido el dataset.
TARGET_COL: str = ""
TREATMENT_COL: str = ""
CONFOUNDERS: list[str] = []
MEDIATORS: list[str] = []
COLLIDERS: list[str] = []
ID_COLS: list[str] = []


def load_raw_data(path: str):
    """Carga el dataset crudo desde data/raw/ y valida N >= 1500 y >= 10 explicativas.

    TODO: leer CSV/XLSX, tipar columnas (continuas/discretas/categóricas),
    normalizar categorías inconsistentes (p.ej. "Phone" vs "Mobile Phone").
    """
    raise NotImplementedError


def missingness_report(df):
    """Tabla de % de faltantes por columna + insumos para clasificar MCAR/MAR/MNAR.

    TODO: % NA, patrón conjunto (missingno), test de Little (MCAR) y regresiones
    logísticas del indicador de NA sobre las demás variables (evidencia MAR).
    """
    raise NotImplementedError


def split_train_test(df, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split estratificado 80/20. Debe ejecutarse ANTES de imputar/escalar/PCA.

    TODO: sklearn.model_selection.train_test_split(stratify=y).
    """
    raise NotImplementedError


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
    """ColumnTransformer: imputación (mediana / moda / indicadores de NA) + escalado + OneHot.

    TODO: devolver un transformador NO ajustado, para usar dentro de Pipelines y CV.
    """
    raise NotImplementedError


def fit_pca_train_only(X_train_num, n_components=None):
    """Pipeline(StandardScaler -> PCA) ajustado SOLO en Train.

    TODO: devolver el pipeline ajustado + varianza explicada + loadings;
    luego aplicar .transform() a Test "a ciegas".
    """
    raise NotImplementedError

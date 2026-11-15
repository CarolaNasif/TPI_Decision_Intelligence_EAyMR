"""Torneo predictivo y calibración (Fase 3).

Dueño: Backend & ML Engineer. Revisor: QA & Validation Engineer.

Protocolo inmutable:
    - Test (20%) se toca UNA sola vez, al final, para la matriz comparativa.
    - Toda búsqueda de hiperparámetros usa StratifiedKFold(n_splits=5, shuffle=True,
      random_state=RANDOM_STATE) SOLO dentro de Train.
    - Todo modelo es un Pipeline(preprocessor, estimator) para evitar leakage en CV.
    - El "mejor modelo" se elige por ROC-AUC de CV (nunca por Test).

Familias:
    1. Logística regularizada (ElasticNet: L2 / mixta / L1) + Logit statsmodels para Odds Ratios con IC 95%.
    2. Naive Bayes Gaussiano + diagnóstico de independencia condicional (correlación intra-clase).
    3. CART sin restricción (overfit) vs CART podado con ccp_alpha óptimo por CV.
    4. Random Forest + RandomizedSearchCV + OOB (OOB vs CV).

Uso: `python -m src.models` corre el torneo completo y guarda artifacts/best_model.joblib
y artifacts/model_results.json.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import RANDOM_STATE
from src import preprocessing as prep

ARTIFACTS = prep.PROJECT_ROOT / "artifacts"
BEST_MODEL_PATH = ARTIFACTS / "best_model.joblib"
RESULTS_PATH = ARTIFACTS / "model_results.json"

# Features del torneo predictivo: todas salvo ID, target y coupon_any (redundante con CouponUsed).
# DaySinceLastOrder entra aquí (es predictiva) pero NUNCA en el set de ajuste causal (Fase 4).
FEATURE_NUMERIC: list[str] = list(prep.NUMERIC_COLS)
FEATURE_CATEGORICAL: list[str] = list(prep.CATEGORICAL_COLS)
FEATURES: list[str] = FEATURE_NUMERIC + FEATURE_CATEGORICAL

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "neg_brier": "neg_brier_score"}


# --------------------------------------------------------------------------- utilidades
def make_cv(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """StratifiedKFold de 5 pliegues con shuffle y semilla fija."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def default_preprocessor():
    """Preprocesador NO ajustado sobre las features del torneo."""
    return prep.build_preprocessor(FEATURE_NUMERIC, FEATURE_CATEGORICAL)


def _pipe(preprocessor, model) -> Pipeline:
    return Pipeline([("pre", clone(preprocessor)), ("model", model)])


def _search(estimator, grid, X, y, *, randomized: bool = False, n_iter: int = 25):
    kwargs = dict(scoring=SCORING, refit="roc_auc", cv=make_cv(), n_jobs=-1, return_train_score=True)
    if randomized:
        search = RandomizedSearchCV(estimator, grid, n_iter=n_iter, random_state=RANDOM_STATE, **kwargs)
    else:
        search = GridSearchCV(estimator, grid, **kwargs)
    return search.fit(X[FEATURES], y)


def cv_summary(search) -> dict:
    """Métricas de CV (media ± desvío) del mejor candidato de una búsqueda."""
    res, i = search.cv_results_, search.best_index_
    out = {"best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()}}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    out["train_roc_auc_mean"] = float(res["mean_train_roc_auc"][i])
    return out


# --------------------------------------------------------------------------- familia 1
def fit_logistic_family(X_train, y_train, preprocessor=None):
    """Logística ElasticNet: grilla en C y l1_ratio (0 = Ridge/L2, 1 = Lasso/L1). Devuelve la búsqueda."""
    preprocessor = preprocessor or default_preprocessor()
    model = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    grid = {"model__C": np.logspace(-3, 2, 8), "model__l1_ratio": [0.0, 0.5, 1.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _search(_pipe(preprocessor, model), grid, X_train, y_train)


def logistic_coef_path(X_train, y_train, preprocessor=None, Cs=None, l1_ratio: float = 1.0) -> pd.DataFrame:
    """Senda de coeficientes vs C (L1 por defecto): muestra cómo la penalización apaga variables."""
    preprocessor = preprocessor or default_preprocessor()
    Cs = np.logspace(-3, 2, 15) if Cs is None else Cs
    pre = clone(preprocessor).fit(X_train[FEATURES])
    Xt = pre.transform(X_train[FEATURES])
    rows = {}
    for C in Cs:
        m = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows[C] = m.fit(Xt, y_train).coef_[0]
    return pd.DataFrame(rows, index=pre.get_feature_names_out()).T.rename_axis("C")


def logit_odds_ratios(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """statsmodels Logit sin penalizar -> OR = exp(beta) con IC 95% y p-valor (casos completos)."""
    import statsmodels.formula.api as smf

    res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    ci = res.conf_int()
    table = pd.DataFrame(
        {
            "beta": res.params,
            "OR": np.exp(res.params),
            "IC95_inf": np.exp(ci[0]),
            "IC95_sup": np.exp(ci[1]),
            "p_valor": res.pvalues,
        }
    )
    table.attrs["n_obs"] = int(res.nobs)
    table.attrs["pseudo_r2"] = float(res.prsquared)
    return table


def default_logit_formula() -> str:
    """Fórmula con todas las features del torneo (categóricas como C())."""
    terms = FEATURE_NUMERIC + [f"C({c})" for c in FEATURE_CATEGORICAL]
    return f"{prep.TARGET_COL} ~ " + " + ".join(terms)


# --------------------------------------------------------------------------- familia 2
def fit_naive_bayes(X_train, y_train, preprocessor=None):
    """GaussianNB (única perilla: var_smoothing). Devuelve la búsqueda."""
    preprocessor = preprocessor or default_preprocessor()
    grid = {"model__var_smoothing": np.logspace(-12, -3, 10)}
    return _search(_pipe(preprocessor, GaussianNB()), grid, X_train, y_train)


def within_class_correlation(df: pd.DataFrame, target: str = prep.TARGET_COL, cols=None) -> dict:
    """Correlación de Spearman entre features DENTRO de cada clase (NB asume que es 0)."""
    cols = cols or FEATURE_NUMERIC
    return {int(k): g[cols].corr(method="spearman") for k, g in df.groupby(target)}


def top_conditional_dependencies(df: pd.DataFrame, target: str = prep.TARGET_COL, k: int = 8) -> pd.DataFrame:
    """Pares de features con mayor |correlación| intra-clase (violaciones de independencia condicional)."""
    rows = []
    for cls, corr in within_class_correlation(df, target).items():
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        for (a, b), r in upper.items():
            rows.append({"clase": cls, "var_1": a, "var_2": b, "spearman": r})
    out = pd.DataFrame(rows)
    out["abs"] = out.spearman.abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").head(k).reset_index(drop=True)


# --------------------------------------------------------------------------- familia 3
def fit_cart_unpruned(X_train, y_train, preprocessor=None) -> Pipeline:
    """Árbol sin restricciones: ilustra el sobreajuste (AUC Train ≈ 1)."""
    preprocessor = preprocessor or default_preprocessor()
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return _pipe(preprocessor, model).fit(X_train[FEATURES], y_train)


def fit_cart_with_pruning(X_train, y_train, preprocessor=None, max_alphas: int = 40):
    """Cost-complexity pruning: camino de alphas en Train -> alpha óptimo por CV.

    Devuelve (búsqueda, tabla con AUC de CV, AUC de Train y nº de hojas por alpha, alpha con regla 1-SE).
    """
    preprocessor = preprocessor or default_preprocessor()
    pre = clone(preprocessor).fit(X_train[FEATURES])
    path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(
        pre.transform(X_train[FEATURES]), y_train
    )
    alphas = np.unique(path.ccp_alphas[:-1])  # el último alpha deja un árbol de un solo nodo
    if len(alphas) > max_alphas:
        alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, max_alphas)))
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    search = _search(_pipe(preprocessor, model), {"model__ccp_alpha": alphas}, X_train, y_train)

    res = search.cv_results_
    table = pd.DataFrame(
        {
            "ccp_alpha": alphas,
            "cv_roc_auc": res["mean_test_roc_auc"],
            "cv_roc_auc_std": res["std_test_roc_auc"],
            "train_roc_auc": res["mean_train_roc_auc"],
        }
    )
    pre_t = pre.transform(X_train[FEATURES])
    table["n_hojas"] = [
        DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=a).fit(pre_t, y_train).get_n_leaves()
        for a in alphas
    ]
    best = table.cv_roc_auc.idxmax()
    se = table.cv_roc_auc_std[best] / np.sqrt(make_cv().get_n_splits())
    alpha_1se = float(table.loc[table.cv_roc_auc >= table.cv_roc_auc[best] - se, "ccp_alpha"].max())
    return search, table, alpha_1se


# --------------------------------------------------------------------------- familia 4
RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 6, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5],
}


def fit_random_forest(X_train, y_train, preprocessor=None, n_iter: int = 25):
    """RandomizedSearchCV sobre los 5 hiperparámetros pedidos por la consigna."""
    preprocessor = preprocessor or default_preprocessor()
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    return _search(
        _pipe(preprocessor, model), RF_PARAM_DISTRIBUTIONS, X_train, y_train, randomized=True, n_iter=n_iter
    )


def rf_oob(rf_pipeline: Pipeline, X_train, y_train) -> dict:
    """Re-ajusta el RF elegido con oob_score=True y reporta OOB (accuracy y ROC-AUC)."""
    pipe = clone(rf_pipeline)
    pipe.set_params(model__oob_score=True, model__n_jobs=-1)
    pipe.fit(X_train[FEATURES], y_train)
    rf = pipe.named_steps["model"]
    return {
        "oob_accuracy": float(rf.oob_score_),
        "oob_roc_auc": float(roc_auc_score(y_train, rf.oob_decision_function_[:, 1])),
    }


def oob_convergence(rf_pipeline: Pipeline, X_train, y_train, n_grid=None) -> pd.DataFrame:
    """Error OOB vs n_estimators (warm_start), para comparar su convergencia con la CV."""
    n_grid = n_grid or [25, 50, 100, 150, 200, 300, 400, 500]
    pre = clone(rf_pipeline.named_steps["pre"]).fit(X_train[FEATURES])
    Xt = pre.transform(X_train[FEATURES])
    rf = clone(rf_pipeline.named_steps["model"]).set_params(
        warm_start=True, oob_score=True, n_jobs=-1, random_state=RANDOM_STATE
    )
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in n_grid:
            rf.set_params(n_estimators=n).fit(Xt, y_train)
            proba = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=float(np.mean(y_train)))
            rows.append(
                {
                    "n_estimators": n,
                    "oob_error": 1 - rf.oob_score_,
                    "oob_roc_auc": roc_auc_score(y_train, proba),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- calibración
def calibrate(pipeline: Pipeline, X_train, y_train, method: str = "isotonic") -> CalibratedClassifierCV:
    """Modelo de resguardo: recalibra las probabilidades con CV interna en Train (isotónica o sigmoid)."""
    cal = CalibratedClassifierCV(estimator=clone(pipeline), method=method, cv=make_cv())
    return cal.fit(X_train[FEATURES], y_train)


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """ECE con bins por cuantiles: promedio ponderado de |frecuencia observada - probabilidad media|."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in np.unique(idx):
        mask = idx == b
        ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def reliability_curves(models: dict, X_test, y_test, n_bins: int = 10) -> pd.DataFrame:
    """Curvas de calibración (bins por cuantiles) de cada modelo, en formato largo."""
    frames = []
    for name, model in models.items():
        proba = model.predict_proba(X_test[FEATURES])[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
        frames.append(
            pd.DataFrame({"modelo": name, "prob_media": mean_pred, "frecuencia_observada": frac_pos})
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- evaluación en Test
def bootstrap_auc_ci(y_true, proba, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """IC por bootstrap percentil del ROC-AUC."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() != y_true[i].max():
            aucs.append(roc_auc_score(y_true[i], proba[i]))
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def evaluate_on_test(models: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """Matriz comparativa en Test (se llama UNA vez): umbral, ranking y calibración."""
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test[FEATURES])[:, 1]
        pred = (proba >= threshold).astype(int)
        lo, hi = bootstrap_auc_ci(y_test, proba)
        rows.append(
            {
                "modelo": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "f1_macro": f1_score(y_test, pred, average="macro"),
                "f1_ponderado": f1_score(y_test, pred, average="weighted"),
                "roc_auc": roc_auc_score(y_test, proba),
                "roc_auc_ic95_inf": lo,
                "roc_auc_ic95_sup": hi,
                "pr_auc": average_precision_score(y_test, proba),
                "brier": brier_score_loss(y_test, proba),
                "log_loss": log_loss(y_test, proba),
                "ece": expected_calibration_error(y_test, proba),
            }
        )
    return pd.DataFrame(rows).set_index("modelo")


# --------------------------------------------------------------------------- importancias
def _original_feature(name: str, numeric: list[str], categorical: list[str]) -> str:
    if name.startswith("missingindicator_"):
        return name.removeprefix("missingindicator_")
    if name in numeric:
        return name
    matches = [c for c in categorical if name.startswith(f"{c}_")]
    return max(matches, key=len) if matches else name


def mdi_by_feature(pipeline: Pipeline, numeric=None, categorical=None) -> pd.Series:
    """MDI (Gini) del RF agregado a la variable original (suma de sus dummies e indicador de NA)."""
    numeric, categorical = numeric or FEATURE_NUMERIC, categorical or FEATURE_CATEGORICAL
    names = pipeline.named_steps["pre"].get_feature_names_out()
    mdi = pd.Series(pipeline.named_steps["model"].feature_importances_, index=names)
    return mdi.groupby([_original_feature(n, numeric, categorical) for n in names]).sum()


def importance_mdi_vs_permutation(
    rf_pipeline: Pipeline, X_test, y_test, n_repeats: int = 20, features=None
) -> pd.DataFrame:
    """MDI (calculada en Train) vs Permutation Importance en TEST (caída de ROC-AUC)."""
    features = features or FEATURES
    perm = permutation_importance(
        rf_pipeline,
        X_test[features],
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {"perm_mean": perm.importances_mean, "perm_std": perm.importances_std}, index=features
    ).join(mdi_by_feature(rf_pipeline, *_split_types(features)).rename("mdi"))
    table["rank_mdi"] = table.mdi.rank(ascending=False).astype(int)
    table["rank_perm"] = table.perm_mean.rank(ascending=False).astype(int)
    return table.sort_values("perm_mean", ascending=False)


def _split_types(features):
    return [f for f in features if f not in FEATURE_CATEGORICAL], [
        f for f in features if f in FEATURE_CATEGORICAL
    ]


def mdi_noise_experiment(rf_pipeline: Pipeline, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Control del sesgo de MDI: agrega ruido continuo y ruido binario sin relación con el target.

    Si MDI asigna importancia > 0 al ruido continuo (muchos puntos de corte) y la permutación
    en Test ≈ 0, el sesgo de cardinalidad queda demostrado empíricamente (Strobl et al., 2007).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    def add_noise(X):
        X = X.copy()
        X["ruido_continuo"] = rng.normal(size=len(X))
        X["ruido_binario"] = rng.integers(0, 2, len(X))
        return X

    Xtr, Xte = add_noise(X_train), add_noise(X_test)
    numeric = FEATURE_NUMERIC + ["ruido_continuo", "ruido_binario"]
    pipe = Pipeline(
        [
            ("pre", prep.build_preprocessor(numeric, FEATURE_CATEGORICAL)),
            ("model", clone(rf_pipeline.named_steps["model"])),
        ]
    )
    feats = numeric + FEATURE_CATEGORICAL
    pipe.fit(Xtr[feats], y_train)
    return importance_mdi_vs_permutation(pipe, Xte, y_test, features=feats)


# --------------------------------------------------------------------------- explicación local
def align_to_train(X_new: pd.DataFrame, X_train: pd.DataFrame, features=None) -> pd.DataFrame:
    """Castea cada columna de X_new al dtype que tenía en Train.

    El OneHotEncoder usa `handle_unknown="ignore"`: si una categoría llega con otro tipo
    (p. ej. CityTier "1" en vez de 1) no falla, la codifica como todo-ceros y la predicción
    cambia en silencio. Alinear tipos antes de predecir evita ese error.
    """
    features = FEATURES if features is None else features
    numeric, categorical = _split_types(features)
    out = X_new[features].copy()
    for col in numeric:
        out[col] = pd.to_numeric(out[col]).astype(float)  # sin truncar medianas x,5
    for col in categorical:
        dtype = X_train[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            cats = dtype.categories
            out[col] = pd.Categorical(out[col].astype(cats.dtype), categories=cats)
        else:
            out[col] = out[col].astype(dtype)
    return out


def train_reference_profile(X_train: pd.DataFrame, features=None) -> pd.Series:
    """Cliente típico de Train: mediana en numéricas y moda en categóricas (sin mirar Test)."""
    features = FEATURES if features is None else features
    numeric, categorical = _split_types(features)
    ref = {c: X_train[c].median() for c in numeric}
    ref.update({c: X_train[c].mode(dropna=True).iloc[0] for c in categorical})
    return pd.Series(ref)[features]


def local_explanation(model, x_row: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Explicación de UNA predicción por sustitución de a una variable.

    Para cada variable j se reemplaza su valor por el del cliente típico de Train y se
    recalcula la probabilidad: `aporte_j = p(x) - p(x con x_j típico)`. Un aporte positivo
    indica que el valor de ese cliente en j le SUBE el riesgo respecto de un valor típico.

    Es una explicación local, no SHAP: no reparte interacciones, así que los aportes no suman
    exactamente `p(x) - p(típico)`. Todas las variantes se predicen en una sola llamada.
    """
    features = list(reference.index)
    x_row = x_row[features].iloc[[0]]
    base = x_row.iloc[0].to_dict()
    rows = [base] + [{**base, col: reference[col]} for col in features]
    # x_row ya viene alineado a Train: se reutilizan sus dtypes para las variantes.
    variants = align_to_train(pd.DataFrame(rows), x_row, features)
    proba = model.predict_proba(variants)[:, 1]
    p = float(proba[0])
    table = pd.DataFrame(
        {
            "variable": features,
            "valor_cliente": [x_row[c].iloc[0] for c in features],
            "valor_tipico": [reference[c] for c in features],
            "proba_con_tipico": proba[1:],
            "aporte": p - proba[1:],
        }
    )
    table.attrs["proba"] = p
    order = table["aporte"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


# --------------------------------------------------------------------------- torneo completo
def run_tournament(df: pd.DataFrame | None = None, rf_iter: int = 25) -> dict:
    """Corre las 4 familias con CV en Train, elige por CV y evalúa en Test una sola vez."""
    df = prep.make_clean_dataset() if df is None else df
    X_train, X_test, y_train, y_test = prep.load_split(df)

    searches = {
        "Logística ElasticNet": fit_logistic_family(X_train, y_train),
        "Naive Bayes": fit_naive_bayes(X_train, y_train),
    }
    cart_search, ccp_table, alpha_1se = fit_cart_with_pruning(X_train, y_train)
    searches["CART podado"] = cart_search
    searches["Random Forest"] = fit_random_forest(X_train, y_train, n_iter=rf_iter)

    models = {name: s.best_estimator_ for name, s in searches.items()}
    cv = {name: cv_summary(s) for name, s in searches.items()}
    unpruned = fit_cart_unpruned(X_train, y_train)

    best_name = max(cv, key=lambda k: cv[k]["cv_roc_auc_mean"])
    best_calibrated = calibrate(models[best_name], X_train, y_train)

    test_models = {"CART sin podar": unpruned, **models, f"{best_name} + isotónica": best_calibrated}
    test = evaluate_on_test(test_models, X_test, y_test)
    unpruned_train_auc = roc_auc_score(y_train, unpruned.predict_proba(X_train[FEATURES])[:, 1])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "searches": searches,
        "models": models,
        "unpruned": unpruned,
        "best_name": best_name,
        "best_calibrated": best_calibrated,
        "cv": cv,
        "test": test,
        "ccp_table": ccp_table,
        "ccp_alpha_1se": alpha_1se,
        "unpruned_train_auc": float(unpruned_train_auc),
        "rf_oob": rf_oob(models["Random Forest"], X_train, y_train),
    }


def save_artifacts(results: dict) -> None:
    """best_model.joblib (pipeline calibrado que recibe las columnas crudas) + model_results.json."""
    import joblib

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(results["best_calibrated"], BEST_MODEL_PATH, compress=3)
    payload = {
        "best_model": results["best_name"],
        "best_model_file": BEST_MODEL_PATH.name,
        "calibration": "isotonic (CalibratedClassifierCV, 5-fold en Train)",
        "selection_metric": "ROC-AUC media en StratifiedKFold(5) sobre Train",
        "threshold": 0.5,
        "features": {"numeric": FEATURE_NUMERIC, "categorical": FEATURE_CATEGORICAL},
        "categories": {
            c: sorted(str(v) for v in results["X_train"][c].dropna().unique()) for c in FEATURE_CATEGORICAL
        },
        "cv": results["cv"],
        "test": results["test"].round(4).reset_index().to_dict(orient="records"),
        "cart": {
            "unpruned_train_roc_auc": results["unpruned_train_auc"],
            "ccp_alpha_best": results["cv"]["CART podado"]["best_params"]["ccp_alpha"],
            "ccp_alpha_1se": results["ccp_alpha_1se"],
        },
        "rf_oob": results["rf_oob"],
        "global_importance": (
            importance_mdi_vs_permutation(
                results["models"]["Random Forest"], results["X_test"], results["y_test"]
            )
            .round(5)
            .reset_index(names="feature")
            .to_dict(orient="records")
        ),
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, default=float, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    results = run_tournament()
    save_artifacts(results)
    pd.set_option("display.width", 200)
    print("CV (Train, 5 folds):")
    print(pd.DataFrame(results["cv"]).T.drop(columns="best_params").round(4).to_string())
    print("\nTest:")
    print(results["test"][["recall", "f1", "roc_auc", "pr_auc", "brier", "ece"]].round(4).to_string())
    print(f"\nCART sin podar: AUC Train {results['unpruned_train_auc']:.4f}")
    print(f"RF OOB: {results['rf_oob']}")
    print(f"Mejor por CV: {results['best_name']} -> {BEST_MODEL_PATH.relative_to(prep.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

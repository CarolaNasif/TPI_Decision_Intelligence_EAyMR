"""Inferencia causal y Causal ML (Fase 4).

Dueño: Backend & ML / Causal Engineer. Revisor: QA & Validation Engineer.

Reglas de identificación (Pearl):
    - El set de ajuste = CONFOUNDERS (Z) pre-tratamiento.
    - MEDIATORS (M) y COLLIDERS (C) se declaran en el DAG pero NUNCA entran al ajuste.
    - Se reporta siempre el coeficiente ingenuo Y ~ T para cuantificar el sesgo de confusión.

Flujo DoWhy: Model -> Identify -> Estimate (linear_regression + propensity_score_weighting)
             -> Refute (random_common_cause, placebo_treatment_refuter, data_subset_refuter).
Causal ML : EconML LinearDML / CausalForestDML con RandomForest como nuisance (model_y, model_t).

Unidad de los efectos: Y = Churn (0/1) y T = coupon_any (0/1), así que el ATE es una diferencia de
probabilidades de churn (puntos porcentuales / 100). Un ATE NEGATIVO significa que el cupón RETIENE.

Uso: `python -m src.causal` corre la Fase 4 completa y guarda artifacts/dag.dot,
artifacts/causal_results.json y artifacts/cate_grid.parquet (contrato del módulo 3 de la app).
"""

from __future__ import annotations

import json
import warnings
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from src import RANDOM_STATE
from src import preprocessing as prep

TREATMENT = prep.TREATMENT_COL
OUTCOME = prep.TARGET_COL
LATENT = "U_engagement"
LATENT_CHILDREN = ["HourSpendOnApp", TREATMENT, OUTCOME]

# Discretas de pocos niveles con relación NO monótona con T: se codifican como categóricas en Z.
Z_DISCRETE_AS_CATEGORICAL = ["HourSpendOnApp", "NumberOfDeviceRegistered"]
Z_NUMERIC = [z for z in prep.CONFOUNDERS if z in prep.NUMERIC_COLS and z not in Z_DISCRETE_AS_CATEGORICAL]
Z_CATEGORICAL = [z for z in prep.CONFOUNDERS if z in prep.CATEGORICAL_COLS] + Z_DISCRETE_AS_CATEGORICAL
EFFECT_MODIFIERS = ["Tenure", "Complain", "SatisfactionScore"]

# Positividad: un nivel de Z con >= MIN_CELL clientes y TODOS tratados (o ninguno) no tiene contrafáctico.
MIN_CELL = 20
PS_BOUNDS = (0.05, 0.95)

ARTIFACTS = prep.PROJECT_ROOT / "artifacts"
DAG_PATH = ARTIFACTS / "dag.dot"
RESULTS_PATH = ARTIFACTS / "causal_results.json"
CATE_GRID_PATH = ARTIFACTS / "cate_grid.parquet"


# --------------------------------------------------------------------------- DAG
def build_dag(
    confounders=None,
    treatment: str = TREATMENT,
    outcome: str = OUTCOME,
    mediators=None,
    colliders=None,
    latent: str | None = LATENT,
) -> nx.DiGraph:
    """DAG a nivel de variable (contrato de docs/01). Cada nodo lleva su rol en el atributo `role`."""
    confounders = list(prep.CONFOUNDERS if confounders is None else confounders)
    mediators = list(prep.MEDIATORS if mediators is None else mediators)
    colliders = list(prep.COLLIDERS if colliders is None else colliders)
    g = nx.DiGraph()
    g.add_node(treatment, role="T")
    g.add_node(outcome, role="Y")
    g.add_edge(treatment, outcome)
    for z in confounders:
        g.add_node(z, role="Z")
        g.add_edges_from([(z, treatment), (z, outcome)])
    for m in mediators:
        g.add_node(m, role="M")
        g.add_edges_from([(treatment, m), (m, outcome)])
    for c in colliders:
        g.add_node(c, role="C")
        g.add_edges_from([(treatment, c), (outcome, c)])
    if latent:
        g.add_node(latent, role="U")
        g.add_edges_from((latent, child) for child in LATENT_CHILDREN if child in g)
    return g


ROLE_STYLE = {
    "T": 'shape=box, style="filled,bold", fillcolor="#f4a259"',
    "Y": 'shape=box, style="filled,bold", fillcolor="#5b8e7d"',
    "Z": 'style=filled, fillcolor="#dfe7fd"',
    "M": 'style=filled, fillcolor="#fde2e4"',
    "C": 'style=filled, fillcolor="#fff1c1"',
    "U": 'style=dashed, fontcolor="#666666"',
}


def dag_to_dot(g: nx.DiGraph) -> str:
    """DOT con estilo por rol (para la app: st.graphviz_chart). Las aristas del latente van punteadas."""
    lines = ["digraph DAG {", "  rankdir=LR;", '  node [fontname="Helvetica", fontsize=11];']
    for node, data in g.nodes(data=True):
        lines.append(f'  "{node}" [{ROLE_STYLE[data["role"]]}];')
    for a, b in g.edges():
        style = " [style=dashed]" if g.nodes[a]["role"] == "U" else ""
        lines.append(f'  "{a}" -> "{b}"{style};')
    lines.append("}")
    return "\n".join(lines)


def _d_separated(g: nx.DiGraph, x: set, y: set, z: set) -> bool:
    fn = getattr(nx, "is_d_separator", None) or nx.d_separated
    return fn(g, x, y, z)


def backdoor_paths(g: nx.DiGraph, treatment: str = TREATMENT, outcome: str = OUTCOME) -> list[list[str]]:
    """Caminos entre T e Y (en el esqueleto no dirigido) que entran a T por una flecha (T <- ...)."""
    skeleton = g.to_undirected()
    paths = []
    for path in nx.all_simple_paths(skeleton, treatment, outcome, cutoff=4):
        if g.has_edge(path[1], treatment):
            paths.append(path)
    return paths


def check_backdoor(g: nx.DiGraph, treatment: str, outcome: str, adjustment_set) -> dict:
    """Criterio backdoor (Pearl): (i) ningún nodo de Z desciende de T; (ii) Z d-separa T e Y en el grafo
    sin las aristas salientes de T. Los nodos latentes no pueden condicionarse."""
    adjustment_set = set(adjustment_set)
    descendants = nx.descendants(g, treatment)
    latent = {n for n, r in g.nodes(data="role") if r == "U"}
    g_back = g.copy()
    g_back.remove_edges_from(list(g.out_edges(treatment)))
    bad_descendants = sorted(adjustment_set & descendants)
    blocks = _d_separated(g_back, {treatment}, {outcome}, adjustment_set - latent)
    return {
        "no_descendants_of_T": not bad_descendants,
        "descendants_in_set": bad_descendants,
        "blocks_all_backdoor_paths": bool(blocks),
        "satisfied": not bad_descendants and bool(blocks),
        "latent_nodes": sorted(latent),
    }


# --------------------------------------------------------------------------- datos
def prepare_causal_frame(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Matriz numérica para DoWhy/EconML.

    - Se excluyen las filas con T faltante (CouponUsed NA): no se imputa el tratamiento.
    - Z numéricas: mediana + indicador de NA (los NA son MAR según Fase 1). Z categóricas: one-hot
      con categoría base. M y C se imputan por mediana solo para las estimaciones "trampa".
    """
    df = prep.make_clean_dataset() if df is None else df
    d = df[df[TREATMENT].notna()].copy()
    d[TREATMENT] = d[TREATMENT].astype(int)

    z_num = d[Z_NUMERIC].astype(float)
    na_cols = [c for c in [*Z_NUMERIC, *Z_DISCRETE_AS_CATEGORICAL] if d[c].isna().any()]
    for c in na_cols:
        z_num[f"na_{c}"] = d[c].isna().astype(int)
    z_num = z_num.fillna(z_num.median())
    z_cat_raw = d[Z_CATEGORICAL].copy()
    for c in Z_DISCRETE_AS_CATEGORICAL:
        z_cat_raw[c] = d[c].fillna(d[c].median()).astype(int)
    z_cat = pd.get_dummies(z_cat_raw.astype(str), drop_first=True, dtype=int)
    z_cat.columns = [c.replace(" ", "_").replace("&", "and") for c in z_cat.columns]

    mc_cols = prep.MEDIATORS + prep.COLLIDERS
    mc = d[mc_cols].astype(float)
    mc = mc.fillna(mc.median())

    frame = pd.concat(
        [d[[prep.ID_COLS[0], TREATMENT, OUTCOME, prep.TREATMENT_SOURCE_COL]], z_num, z_cat, mc], axis=1
    )
    frame = frame.reset_index(drop=True)
    info = {
        "z_cols": list(z_num.columns) + list(z_cat.columns),
        "m_cols": list(prep.MEDIATORS),
        "c_cols": list(prep.COLLIDERS),
        "n_total": int(len(df)),
        "n_used": int(len(frame)),
        "n_dropped_treatment_na": int(df[TREATMENT].isna().sum()),
    }
    return frame, info


def positivity_violations(df: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Niveles de Z (valor crudo) con >= min_cell clientes y sin variación en T: no hay contrafáctico."""
    d = df[df[TREATMENT].notna()]
    rows = []
    for z in prep.CONFOUNDERS:
        g = d.groupby(d[z].astype(str), observed=True)[TREATMENT].agg(["mean", "size"])
        for level, row in g[g["mean"].isin([0, 1]) & (g["size"] >= min_cell)].iterrows():
            rows.append(
                {"variable": z, "nivel": level, "n": int(row["size"]), "share_tratados": float(row["mean"])}
            )
    return pd.DataFrame(rows, columns=["variable", "nivel", "n", "share_tratados"])


def common_support(
    frame: pd.DataFrame, df: pd.DataFrame, info: dict
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Restringe a la población con soporte común: (1) quita los estratos con violación estructural de
    positividad y (2) recorta propensity fuera de PS_BOUNDS. Devuelve (frame_soporte, ps, resumen)."""
    d = df[df[TREATMENT].notna()].reset_index(drop=True)
    violations = positivity_violations(df)
    excluded = np.zeros(len(d), bool)
    for _, v in violations.iterrows():
        excluded |= (d[v["variable"]].astype(str) == v["nivel"]).to_numpy()
    f = frame.loc[~excluded].reset_index(drop=True)
    ps = propensity_scores(f, info["z_cols"])
    keep = (ps >= PS_BOUNDS[0]) & (ps <= PS_BOUNDS[1])
    summary = {
        "n_antes": int(len(frame)),
        "n_excluidos_estratos": int(excluded.sum()),
        "n_excluidos_trimming": int((~keep).sum()),
        "n_soporte": int(keep.sum()),
        "estratos_excluidos": violations.to_dict(orient="records"),
        "ps_bounds": list(PS_BOUNDS),
    }
    return f.loc[keep].reset_index(drop=True), ps[keep], summary


def encoded_graph(z_cols, mediators=None, colliders=None) -> nx.DiGraph:
    """DAG equivalente al de variables pero con las columnas codificadas (nombres = columnas de datos)."""
    mediators = prep.MEDIATORS if mediators is None else mediators
    colliders = prep.COLLIDERS if colliders is None else colliders
    g = nx.DiGraph([(TREATMENT, OUTCOME)])
    g.add_edges_from((z, n) for z in z_cols for n in (TREATMENT, OUTCOME))
    g.add_edges_from((a, b) for m in mediators for a, b in ((TREATMENT, m), (m, OUTCOME)))
    g.add_edges_from((a, c) for c in colliders for a in (TREATMENT, OUTCOME))
    return g


# --------------------------------------------------------------------------- estimadores simples
def _ols_effect(frame: pd.DataFrame, controls: list[str], treatment: str = TREATMENT) -> dict:
    import statsmodels.api as sm

    X = sm.add_constant(frame[[treatment, *controls]].astype(float))
    res = sm.OLS(frame[OUTCOME].astype(float), X).fit(cov_type="HC3")
    lo, hi = res.conf_int().loc[treatment]
    return {
        "ate": float(res.params[treatment]),
        "ci": [float(lo), float(hi)],
        "p_value": float(res.pvalues[treatment]),
    }


def naive_effect(frame: pd.DataFrame, treatment: str = TREATMENT, outcome: str = OUTCOME) -> dict:
    """Regresión ingenua Y ~ T (modelo de probabilidad lineal, errores HC3) = diferencia de tasas de churn."""
    out = _ols_effect(frame, [], treatment)
    out["churn_rate_T1"] = float(frame.loc[frame[treatment] == 1, outcome].mean())
    out["churn_rate_T0"] = float(frame.loc[frame[treatment] == 0, outcome].mean())
    return out


def trap_estimates(frame: pd.DataFrame, info: dict) -> dict:
    """Ajuste correcto (Z) vs las trampas: agregar mediadores (efecto directo) o el colisionador (Berkson)."""
    z = info["z_cols"]
    return {
        "Z (correcto)": _ols_effect(frame, z),
        "Z + M (trampa del mediador)": _ols_effect(frame, z + info["m_cols"]),
        "Z + C (trampa del colisionador)": _ols_effect(frame, z + info["c_cols"]),
        "Z + M + C": _ols_effect(frame, z + info["m_cols"] + info["c_cols"]),
    }


def dose_response(frame: pd.DataFrame, info: dict) -> dict:
    """Sensibilidad a la definición de T: CouponUsed continua (efecto por cupón extra), ajustada por Z."""
    return _ols_effect(frame, info["z_cols"], treatment=prep.TREATMENT_SOURCE_COL)


def collider_simulation(n: int = 20_000, true_effect: float = 0.10, seed: int = RANDOM_STATE) -> dict:
    """Mini-simulación con efecto conocido: condicionar en un colisionador sesga la estimación."""
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    t = rng.binomial(1, 0.5, n)
    y = true_effect * t + rng.normal(size=n)
    c = t + y + rng.normal(size=n)  # T -> C <- Y
    without_c = sm.OLS(y, sm.add_constant(t)).fit().params[1]
    with_c = sm.OLS(y, sm.add_constant(np.column_stack([t, c]))).fit().params[1]
    return {
        "true_effect": true_effect,
        "estimate_without_C": float(without_c),
        "estimate_with_C": float(with_c),
    }


# --------------------------------------------------------------------------- DoWhy
def dowhy_pipeline(
    frame: pd.DataFrame, graph: nx.DiGraph, treatment: str = TREATMENT, outcome: str = OUTCOME
):
    """CausalModel -> identify_effect -> estimate_effect (regresión lineal e IPW), con IC 95%."""
    from dowhy import CausalModel

    gml = "".join(nx.generate_gml(graph))
    np.random.seed(RANDOM_STATE)  # el bootstrap del IC de IPW en DoWhy usa el generador global de numpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(
            data=frame.drop(columns=prep.ID_COLS), treatment=treatment, outcome=outcome, graph=gml
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        lr = model.estimate_effect(
            estimand, method_name="backdoor.linear_regression", confidence_intervals=True
        )
        ipw = model.estimate_effect(
            estimand,
            method_name="backdoor.propensity_score_weighting",
            # Logística escalada y con iteraciones suficientes para converger: con el modelo por defecto
            # el resultado dependía del orden de las columnas (DoWhy las toma de un set).
            method_params={"weighting_scheme": "ips_weight", "propensity_score_model": _propensity_model()},
            confidence_intervals=True,
        )
    return model, estimand, {"linear_regression": lr, "ipw": ipw}


def _estimate_dict(estimate) -> dict:
    ci = np.asarray(estimate.get_confidence_intervals()).ravel()
    return {"ate": float(estimate.value), "ci": [float(ci[0]), float(ci[1])]}


# --------------------------------------------------------------------------- overlap y balance
def _propensity_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, tol=1e-8))


def propensity_scores(frame: pd.DataFrame, z_cols) -> np.ndarray:
    """P(T=1 | Z) con logística sobre Z estandarizada (misma especificación que IPW de DoWhy)."""
    model = _propensity_model()
    return model.fit(frame[z_cols], frame[TREATMENT]).predict_proba(frame[z_cols])[:, 1]


def ipw_ate(frame: pd.DataFrame, ps: np.ndarray, bounds: tuple[float, float] | None = None) -> float:
    """ATE por IPW (Horvitz-Thompson normalizado). Con `bounds` se recortan las unidades sin overlap."""
    t, y = frame[TREATMENT].to_numpy(), frame[OUTCOME].to_numpy(dtype=float)
    keep = np.ones(len(t), bool) if bounds is None else (ps >= bounds[0]) & (ps <= bounds[1])
    t, y, p = t[keep], y[keep], ps[keep]
    w1, w0 = t / p, (1 - t) / (1 - p)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def overlap_diagnostics(frame: pd.DataFrame, ps: np.ndarray) -> dict:
    """Positividad: distribución del propensity por grupo y proporción fuera de [0,05; 0,95]."""
    t = frame[TREATMENT].to_numpy()
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    return {
        "quantiles": q,
        "ps_treated": np.quantile(ps[t == 1], q).round(4).tolist(),
        "ps_control": np.quantile(ps[t == 0], q).round(4).tolist(),
        "share_outside_05_95": float(((ps < 0.05) | (ps > 0.95)).mean()),
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
        "ipw_ate_trimmed_05_95": ipw_ate(frame, ps, (0.05, 0.95)),
        "ipw_ate_trimmed_10_90": ipw_ate(frame, ps, (0.10, 0.90)),
    }


def balance_table(frame: pd.DataFrame, ps: np.ndarray, z_cols) -> pd.DataFrame:
    """Diferencia de medias estandarizada (SMD) de cada Z antes y después de ponderar por IPW."""
    t = frame[TREATMENT].to_numpy()
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    rows = []
    for z in z_cols:
        x = frame[z].to_numpy(dtype=float)
        pooled_sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2) or 1.0
        raw = (x[t == 1].mean() - x[t == 0].mean()) / pooled_sd
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        rows.append({"variable": z, "smd_antes": raw, "smd_despues": (m1 - m0) / pooled_sd})
    out = pd.DataFrame(rows).set_index("variable")
    return out.reindex(out.smd_antes.abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- refutaciones
def _semaforo(ok: bool, warn: bool) -> str:
    return "verde" if ok else ("amarillo" if warn else "rojo")


def run_refutations(model, identified_estimand, estimate, num_simulations: int = 100) -> list[dict]:
    """3 refutadores de DoWhy con semáforo (verde/amarillo/rojo), serializable a JSON para la app.

    Criterios: Random Common Cause -> |Δ| < 10% del ATE o < 0,5 pp (verde), < 25% o < 1 pp (amarillo).
    Placebo -> el efecto placebo debe ser ≈ 0: |placebo| < 25% del |ATE| o < 0,5 pp.
    Data Subset (80%) -> el ATE re-estimado queda dentro del IC 95% original.
    """
    ate = float(estimate.value)
    lo, hi = np.asarray(estimate.get_confidence_intervals()).ravel()
    specs = [
        ("Random Common Cause", "random_common_cause", {}),
        ("Placebo Treatment", "placebo_treatment_refuter", {"placebo_type": "permute"}),
        ("Data Subset (80%)", "data_subset_refuter", {"subset_fraction": 0.8}),
    ]
    out = []
    for label, method, kwargs in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name=method,
                num_simulations=num_simulations,
                random_seed=RANDOM_STATE,
                **kwargs,
            )
        new = float(res.new_effect)
        delta = abs(new - ate)
        if method == "random_common_cause":
            color = _semaforo(delta < max(0.1 * abs(ate), 0.005), delta < max(0.25 * abs(ate), 0.01))
            meaning = "Agregar un confusor aleatorio no debería mover el ATE."
        elif method == "placebo_treatment_refuter":
            color = _semaforo(abs(new) < max(0.25 * abs(ate), 0.005), abs(new) < max(0.5 * abs(ate), 0.01))
            meaning = "Con T permutado al azar el efecto debe ser ≈ 0."
        else:
            color = _semaforo(lo <= new <= hi, delta < 0.5 * (hi - lo))
            meaning = "Re-estimar en submuestras del 80% debe dejar el ATE dentro del IC original."
        out.append(
            {
                "refutador": label,
                "ate_original": ate,
                "ate_nuevo": new,
                "p_value_dowhy": float(res.refutation_result["p_value"]) if res.refutation_result else None,
                "semaforo": color,
                "interpretacion": meaning,
            }
        )
    return out


def unobserved_confounder_sensitivity(
    model, identified_estimand, estimate, strengths=(0.01, 0.05, 0.1)
) -> dict:
    """Extra informativo: rango del ATE si existiera un confusor no observado (p. ej. U_engagement)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method="direct-simulation",
            confounders_effect_on_treatment="binary_flip",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=list(strengths),
            effect_strength_on_outcome=list(strengths),
            plotmethod=None,
        )
    lo, hi = np.asarray(res.new_effect, dtype=float).ravel()[:2]
    return {"strengths": list(strengths), "ate_min": float(lo), "ate_max": float(hi)}


# --------------------------------------------------------------------------- EconML
def _nuisance():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    t = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    return y, t


def fit_dml_cate(frame: pd.DataFrame, info: dict, effect_modifiers=None) -> dict:
    """LinearDML (CATE lineal en X) y CausalForestDML (CATE no lineal), con RF como nuisance y cross-fitting.

    X = modificadores de efecto interpretables; W = resto de Z (controles). Devuelve modelos, ATE con IC
    y el CATE individual del bosque causal.
    """
    from econml.dml import CausalForestDML, LinearDML

    effect_modifiers = list(effect_modifiers or EFFECT_MODIFIERS)
    X = frame[effect_modifiers].astype(float)
    W = frame[[c for c in info["z_cols"] if c not in effect_modifiers]].astype(float)
    y, t = frame[OUTCOME].astype(float), frame[TREATMENT]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, mt = _nuisance()
        ldml = LinearDML(model_y=my, model_t=mt, discrete_treatment=True, cv=5, random_state=RANDOM_STATE)
        ldml.fit(y, t, X=X, W=W)
        my, mt = _nuisance()
        cf = CausalForestDML(
            model_y=my,
            model_t=mt,
            discrete_treatment=True,
            cv=5,
            n_estimators=500,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        )
        cf.fit(y, t, X=X, W=W)

    def ate(est):
        inf = est.ate_inference(X)
        lo, hi = inf.conf_int_mean()
        return {"ate": float(inf.mean_point), "ci": [float(lo), float(hi)]}

    def forest_ate(est):
        # ATE doblemente robusto del bosque (ate__inference): su IC refleja la incertidumbre de la media,
        # no la dispersión de los CATE individuales.
        inf = est.ate__inference()
        lo, hi = (np.ravel(v)[0] for v in inf.conf_int())
        return {"ate": float(np.ravel(inf.point_estimate)[0]), "ci": [float(lo), float(hi)]}

    coefs = ldml.summary().tables[0].data
    coef_table = pd.DataFrame(coefs[1:], columns=coefs[0]).set_index(coefs[0][0])
    return {
        "linear_dml": ldml,
        "causal_forest": cf,
        "X": X,
        "ate_linear_dml": ate(ldml),
        "ate_causal_forest": forest_ate(cf),
        "linear_dml_coefs": coef_table,
        "cate_individual": cf.effect(X),
    }


def cate_by_subgroup(frame: pd.DataFrame, dml: dict) -> pd.DataFrame:
    """CATE promedio del bosque causal por segmento, con IC 95% (ate_inference sobre el subgrupo)."""
    cf, X = dml["causal_forest"], dml["X"]
    segments = {
        "Complain": frame["Complain"].map({0: "sin reclamo", 1: "con reclamo"}),
        "Tenure": pd.cut(
            frame["Tenure"], [-1, 2, 5, 10, 20, 100], labels=["0-2", "3-5", "6-10", "11-20", "21+"]
        ),
        "SatisfactionScore": frame["SatisfactionScore"].astype(int).astype(str),
    }
    rows = []
    for var, labels in segments.items():
        for level in labels.dropna().unique():
            mask = (labels == level).to_numpy()
            inf = cf.ate_inference(X[mask])
            lo, hi = inf.conf_int_mean()
            rows.append(
                {
                    "variable": var,
                    "segmento": str(level),
                    "n": int(mask.sum()),
                    "cate": float(inf.mean_point),
                    "ci_inf": float(lo),
                    "ci_sup": float(hi),
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "segmento"]).reset_index(drop=True)


def cate_grid(dml: dict) -> pd.DataFrame:
    """Grilla de perfiles (Tenure x Complain x SatisfactionScore) con CATE e IC de ambos estimadores."""
    grid = pd.DataFrame(
        list(product(range(0, 32), [0, 1], [1, 2, 3, 4, 5])), columns=EFFECT_MODIFIERS, dtype=float
    )
    out = grid.astype(int).copy()
    for name, est in (("linear_dml", dml["linear_dml"]), ("causal_forest", dml["causal_forest"])):
        lo, hi = est.effect_interval(grid)
        out[f"cate_{name}"] = est.effect(grid)
        out[f"ci_inf_{name}"] = lo
        out[f"ci_sup_{name}"] = hi
    return out


def policy_summary(dml: dict) -> dict:
    """Política: dar cupón solo a quien tiene CATE < 0 (el cupón le baja la probabilidad de churn)."""
    cate = np.asarray(dml["cate_individual"])
    targeted = cate < 0
    return {
        "share_cate_negativo": float(targeted.mean()),
        "efecto_medio_dar_a_todos_pp": float(cate.mean() * 100),
        "efecto_medio_solo_cate_negativo_pp": float(cate[targeted].mean() * 100) if targeted.any() else 0.0,
        "churn_evitado_por_100_clientes_targeting": float(-cate[targeted].sum() / len(cate) * 100),
    }


# --------------------------------------------------------------------------- orquestación
def run_causal_analysis(df: pd.DataFrame | None = None, num_simulations: int = 100) -> dict:
    """Fase 4 completa sobre el dataset limpio (efecto poblacional: no se usa el split de Fase 3).

    El ATE principal se estima en la población con SOPORTE COMÚN; la regresión sobre la muestra completa
    se reporta como sensibilidad (extrapola a estratos donde todos usaron cupón).
    """
    df = prep.make_clean_dataset() if df is None else df
    frame_full, info = prepare_causal_frame(df)
    full_sample_lr = _ols_effect(frame_full, info["z_cols"])
    ps_full = propensity_scores(frame_full, info["z_cols"])
    overlap_full = overlap_diagnostics(frame_full, ps_full)
    frame, ps, support = common_support(frame_full, df, info)
    # Dummies de los estratos excluidos quedan constantes en la submuestra: se quitan del set de ajuste.
    z_cols = [c for c in info["z_cols"] if frame[c].nunique() > 1]
    frame = frame.drop(columns=[c for c in info["z_cols"] if c not in z_cols])
    info = {**info, "z_cols": z_cols, "n_used": int(len(frame)), "support": support}
    dag = build_dag()
    bd = check_backdoor(dag, TREATMENT, OUTCOME, prep.CONFOUNDERS)
    bd_with_m = check_backdoor(dag, TREATMENT, OUTCOME, [*prep.CONFOUNDERS, *prep.MEDIATORS])
    bd_with_c = check_backdoor(dag, TREATMENT, OUTCOME, [*prep.CONFOUNDERS, *prep.COLLIDERS])
    bd_no_latent = check_backdoor(build_dag(latent=None), TREATMENT, OUTCOME, prep.CONFOUNDERS)

    model, estimand, est = dowhy_pipeline(frame, encoded_graph(info["z_cols"]))
    dml = fit_dml_cate(frame, info)

    return {
        "frame": frame,
        "info": info,
        "dag": dag,
        "backdoor": {
            "con_latente": bd,
            "sin_latente": bd_no_latent,
            "Z+M": bd_with_m,
            "Z+C": bd_with_c,
            "caminos": [" - ".join(p) for p in backdoor_paths(build_dag(latent=None))],
        },
        "identified_backdoor_set": sorted(estimand.get_backdoor_variables()),
        "model": model,
        "estimand": estimand,
        "estimates_dowhy": est,
        "naive": naive_effect(frame),
        "traps": trap_estimates(frame, info),
        "dose_response": dose_response(frame, info),
        "collider_simulation": collider_simulation(),
        "propensity": ps,
        "overlap": overlap_diagnostics(frame, ps),
        "overlap_full_sample": overlap_full,
        "full_sample_lr": full_sample_lr,
        "balance": balance_table(frame, ps, info["z_cols"]),
        "refutations": run_refutations(model, estimand, est["linear_regression"], num_simulations),
        "sensitivity": unobserved_confounder_sensitivity(model, estimand, est["linear_regression"]),
        "dml": dml,
        "cate_subgroups": cate_by_subgroup(frame, dml),
        "policy": policy_summary(dml),
    }


def results_payload(r: dict) -> dict:
    """Resumen serializable (contrato del módulo 3 de la app)."""
    lr = _estimate_dict(r["estimates_dowhy"]["linear_regression"])
    estimates = {
        "Ingenuo (Y ~ T)": {k: r["naive"][k] for k in ("ate", "ci")},
        "Backdoor: regresión lineal": lr,
        "Backdoor: IPW": _estimate_dict(r["estimates_dowhy"]["ipw"]),
        "LinearDML (EconML)": r["dml"]["ate_linear_dml"],
        "CausalForestDML (EconML)": r["dml"]["ate_causal_forest"],
        "Con M (trampa del mediador)": {
            k: r["traps"]["Z + M (trampa del mediador)"][k] for k in ("ate", "ci")
        },
        "Con C (trampa del colisionador)": {
            k: r["traps"]["Z + C (trampa del colisionador)"][k] for k in ("ate", "ci")
        },
        "Regresión en muestra completa (extrapola)": {k: r["full_sample_lr"][k] for k in ("ate", "ci")},
    }
    balance = r["balance"]
    return {
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "unidad": "diferencia de probabilidad de churn (x100 = pp); negativo = el cupón retiene",
        "n": r["info"],
        "dag_file": DAG_PATH.name,
        "cate_grid_file": CATE_GRID_PATH.name,
        "adjustment_set": prep.CONFOUNDERS,
        "mediators": prep.MEDIATORS,
        "colliders": prep.COLLIDERS,
        "latent": LATENT,
        "backdoor": r["backdoor"],
        "dowhy_backdoor_set": r["identified_backdoor_set"],
        "primary_estimate": "Backdoor: regresión lineal",
        "estimates": estimates,
        "naive_churn_rates": {"T=1": r["naive"]["churn_rate_T1"], "T=0": r["naive"]["churn_rate_T0"]},
        "confounding_bias": r["naive"]["ate"] - lr["ate"],
        "dose_response_per_coupon": r["dose_response"],
        "collider_simulation": r["collider_simulation"],
        "overlap": r["overlap"],
        "overlap_full_sample": r["overlap_full_sample"],
        "balance": {
            "max_abs_smd_antes": float(balance.smd_antes.abs().max()),
            "max_abs_smd_despues": float(balance.smd_despues.abs().max()),
            "n_smd_mayor_0_1_antes": int((balance.smd_antes.abs() > 0.1).sum()),
            "n_smd_mayor_0_1_despues": int((balance.smd_despues.abs() > 0.1).sum()),
        },
        "refutations": r["refutations"],
        "sensitivity_unobserved": r["sensitivity"],
        "linear_dml_coefs": r["dml"]["linear_dml_coefs"]
        .astype(float)
        .reset_index()
        .to_dict(orient="records"),
        "cate_subgroups": r["cate_subgroups"].to_dict(orient="records"),
        "cate_individual_quantiles": dict(
            zip(
                ["p05", "p25", "p50", "p75", "p95"],
                np.quantile(r["dml"]["cate_individual"], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                strict=True,
            )
        ),
        "policy": r["policy"],
        "effect_modifiers": EFFECT_MODIFIERS,
    }


def save_artifacts(r: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    DAG_PATH.write_text(dag_to_dot(r["dag"]) + "\n", encoding="utf-8")
    cate_grid(r["dml"]).to_parquet(CATE_GRID_PATH, index=False)
    RESULTS_PATH.write_text(
        json.dumps(results_payload(r), indent=2, default=float, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    r = run_causal_analysis()
    save_artifacts(r)
    p = results_payload(r)
    s = p["n"]["support"]
    print(
        f"N: {p['n']['n_total']} -> {s['n_antes']} (sin T faltante) -> {s['n_soporte']} con soporte común "
        f"(-{s['n_excluidos_estratos']} en estratos sin controles, -{s['n_excluidos_trimming']} por trimming)"
    )
    print(
        f"Overlap muestra completa: {p['overlap_full_sample']['share_outside_05_95']:.1%} fuera de [.05; .95]"
    )
    print(
        f"Backdoor con Z (sin latente): {r['backdoor']['sin_latente']['satisfied']} | con U: "
        f"{r['backdoor']['con_latente']['satisfied']}"
    )
    print("\nATE (pp de churn):")
    for name, e in p["estimates"].items():
        print(f"  {name:<34} {e['ate'] * 100:+6.2f}  IC95 [{e['ci'][0] * 100:+.2f}; {e['ci'][1] * 100:+.2f}]")
    print(f"\nOverlap: {p['overlap']['share_outside_05_95']:.1%} fuera de [0,05; 0,95]")
    print(
        f"Balance max|SMD|: {p['balance']['max_abs_smd_antes']:.3f} -> "
        f"{p['balance']['max_abs_smd_despues']:.3f}"
    )
    print("\nRefutaciones:")
    for ref in p["refutations"]:
        print(f"  {ref['refutador']:<20} {ref['ate_nuevo'] * 100:+.2f} pp  [{ref['semaforo']}]")
    print(
        f"\nSensibilidad a confusor no observado: [{p['sensitivity_unobserved']['ate_min'] * 100:+.2f}; "
        f"{p['sensitivity_unobserved']['ate_max'] * 100:+.2f}] pp"
    )
    print(f"Por cupón adicional: {p['dose_response_per_coupon']['ate'] * 100:+.2f} pp")
    print(f"Política: {p['policy']}")


if __name__ == "__main__":
    main()

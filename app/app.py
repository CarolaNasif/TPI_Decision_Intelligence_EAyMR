"""Decision Intelligence Capstone - Web App (Streamlit).

Dueño: Frontend & UI/UX Engineer. La app SOLO consume artefactos serializados en
/artifacts (joblib/json/parquet) generados por `src/` vía el notebook maestro.
Nunca re-entrena ni importa econml/dowhy.

Ejecutar localmente:  streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Rangos indicativos del formulario (observados en el dataset limpio, N=2.982).
NUMERIC_RANGES: dict[str, tuple[float, float, float]] = {
    "Tenure": (0.0, 60.0, 9.0),
    "WarehouseToHome": (5.0, 130.0, 14.0),
    "HourSpendOnApp": (0.0, 5.0, 3.0),
    "NumberOfDeviceRegistered": (1.0, 6.0, 4.0),
    "SatisfactionScore": (1.0, 5.0, 3.0),
    "NumberOfAddress": (1.0, 22.0, 3.0),
    "OrderAmountHikeFromlastYear": (11.0, 26.0, 15.0),
    "CouponUsed": (0.0, 16.0, 1.0),
    "OrderCount": (1.0, 16.0, 2.0),
    "DaySinceLastOrder": (0.0, 31.0, 3.0),
    "CashbackAmount": (0.0, 330.0, 163.0),
}

SEMAFORO_COLOR = {"verde": "#2e9e5b", "amarillo": "#d9a300", "rojo": "#c0392b"}


@st.cache_resource
def load_best_model():
    return joblib.load(ARTIFACTS / "best_model.joblib")


@st.cache_data
def load_model_results() -> dict:
    return json.loads((ARTIFACTS / "model_results.json").read_text(encoding="utf-8"))


@st.cache_data
def load_causal_results() -> dict:
    return json.loads((ARTIFACTS / "causal_results.json").read_text(encoding="utf-8"))


@st.cache_data
def load_cate_grid() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "cate_grid.parquet")


@st.cache_data
def load_dag_dot() -> str:
    return (ARTIFACTS / "dag.dot").read_text(encoding="utf-8")


@st.cache_resource
def load_pca_pipeline():
    path = ARTIFACTS / "pca.joblib"
    return joblib.load(path) if path.exists() else None


st.set_page_config(page_title="Decision Intelligence Capstone", page_icon="🧭", layout="wide")
st.title("🧭 Decision Intelligence Capstone — Retorno causal del motor de recomendaciones")
st.caption("EAyMR 2026 · UCA · P(Y | X) vs P(Y | do(T = t))")

with st.sidebar:
    st.header("Proyecto")
    st.markdown(
        "**Dataset:** E-Commerce Customer Churn (Kaggle), N=2.982  \n"
        "**T:** `coupon_any` = usó cupón  \n"
        "**Y:** `Churn`  \n"
        "**Z:** 13 confusores pre-tratamiento"
    )
    st.info(
        "La app solo lee de `artifacts/`: nunca reentrena modelos ni corre `dowhy`/`econml` en vivo."
    )

tab_diag, tab_pred, tab_causal = st.tabs(
    ["1 · Diagnóstico & PCA", "2 · Simulador Predictivo (What-Is)", "3 · Simulador Causal (What-If)"]
)

# --------------------------------------------------------------------------- Módulo 1
with tab_diag:
    st.subheader("Diagnóstico y exploración")
    pca_pipeline = load_pca_pipeline()
    if pca_pipeline is None:
        st.warning(
            "Pendiente: requiere `artifacts/pca.joblib` (PR #17, Fase 2) y la decisión del equipo "
            "sobre si se puede subir una muestra del dataset al repo para este módulo (licencia Kaggle)."
        )
    else:
        st.success("`artifacts/pca.joblib` disponible. Módulo de biplot pendiente de datos reales.")

# --------------------------------------------------------------------------- Módulo 2 — What-Is
with tab_pred:
    st.subheader("¿Qué es probable que pase? — P(Y | X)")
    model_results = load_model_results()
    model = load_best_model()
    numeric_feats = model_results["features"]["numeric"]
    categorical_feats = model_results["features"]["categorical"]
    categories = model_results["categories"]

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown("**Cliente nuevo**")
        inputs: dict[str, float | str] = {}
        for feat in numeric_feats:
            if feat == "Complain":
                continue  # se pide abajo como checkbox, más legible que un slider 0/1
            lo, hi, default = NUMERIC_RANGES[feat]
            inputs[feat] = st.slider(feat, min_value=lo, max_value=hi, value=default, step=1.0)
        for feat in categorical_feats:
            inputs[feat] = st.selectbox(feat, options=categories[feat])
        inputs["Complain"] = 1.0 if st.checkbox("¿Tuvo un reclamo reciente?") else 0.0
        predict_clicked = st.button("Calcular probabilidad de churn", type="primary")

    with col_result:
        if predict_clicked:
            X_new = pd.DataFrame([inputs])[numeric_feats + categorical_feats]
            proba = float(model.predict_proba(X_new)[:, 1][0])
            st.metric("Probabilidad calibrada de churn", f"{proba:.1%}")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#c0392b" if proba >= 0.5 else "#2e9e5b"},
                        "steps": [
                            {"range": [0, 30], "color": "#e9f7ef"},
                            {"range": [30, 60], "color": "#fef9e7"},
                            {"range": [60, 100], "color": "#fdedec"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin={"t": 30, "b": 0})
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.caption("Completá el formulario y presioná el botón para ver la probabilidad.")

        st.markdown(
            f"**Modelo de producción:** {model_results['best_model']} ({model_results['calibration']})"
        )
        test_df = pd.DataFrame(model_results["test"])
        best_row = test_df[test_df["modelo"].str.contains("isotónica", na=False)]
        if best_row.empty:
            best_row = test_df[test_df["modelo"] == model_results["best_model"]]
        st.dataframe(
            best_row[["modelo", "roc_auc", "pr_auc", "brier", "ece", "f1"]].round(4),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**MDI vs. Permutación — el sesgo de cardinalidad, mostrado con datos reales**")
    imp_df = pd.DataFrame(model_results["global_importance"]).sort_values("rank_perm")
    fig_imp = go.Figure()
    fig_imp.add_bar(name="MDI", x=imp_df["feature"], y=imp_df["mdi"], marker_color="#6a8caf")
    fig_imp.add_bar(
        name="Permutación (Test)", x=imp_df["feature"], y=imp_df["perm_mean"], marker_color="#b23a48"
    )
    fig_imp.update_layout(
        barmode="group",
        height=400,
        xaxis_title="Variable",
        yaxis_title="Importancia",
        legend={"orientation": "h", "y": 1.1},
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "Lectura: MDI sobrevalora variables de alta cardinalidad (medido en Train); la permutación en Test "
        "es insesgada respecto de la cardinalidad. Ver BITACORA.md, Entrada 3, Experimento #2/#3."
    )

# --------------------------------------------------------------------------- Módulo 3 — What-If
with tab_causal:
    st.subheader("¿Qué pasa si intervenimos? — P(Y | do(T = t))")
    causal_results = load_causal_results()

    st.markdown("**Grafo causal (DAG)**")
    st.graphviz_chart(load_dag_dot())
    st.caption(
        "Naranja/verde = T/Y. Celeste = confusores Z. Rosa = mediadores M (excluidos del ajuste). "
        "Amarillo = colisionador C (nunca condicionar). Punteado = latente no observada."
    )

    st.markdown("---")
    st.markdown(f"**Estimaciones del ATE** (unidad: {causal_results['unidad']})")
    primary = causal_results["primary_estimate"]
    rows = []
    for name, est in causal_results["estimates"].items():
        rows.append(
            {
                "estimador": name,
                "ATE (pp)": est["ate"] * 100,
                "IC 95% inf (pp)": est["ci"][0] * 100,
                "IC 95% sup (pp)": est["ci"][1] * 100,
                "principal": "⭐" if name == primary else "",
            }
        )
    est_df = pd.DataFrame(rows)
    st.dataframe(est_df.round(2), hide_index=True, use_container_width=True)

    fig_est = go.Figure()
    fig_est.add_trace(
        go.Scatter(
            x=est_df["ATE (pp)"],
            y=est_df["estimador"],
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": est_df["IC 95% sup (pp)"] - est_df["ATE (pp)"],
                "arrayminus": est_df["ATE (pp)"] - est_df["IC 95% inf (pp)"],
            },
            marker={"size": 10, "color": "#2e6f95"},
        )
    )
    fig_est.add_vline(x=0, line_dash="dash", line_color="grey")
    fig_est.update_layout(
        height=380, xaxis_title="ATE sobre churn (pp, negativo = el cupón retiene)", yaxis_title=""
    )
    st.plotly_chart(fig_est, use_container_width=True)

    st.markdown("---")
    col_overlap, col_refut = st.columns(2)

    with col_overlap:
        st.markdown("**Positividad / soporte común**")
        support = causal_results["n"]["support"]
        st.metric("Población con soporte común", f"{support['n_soporte']:,}".replace(",", "."))
        st.caption(
            f"De {causal_results['n']['n_total']:,} clientes: "
            f"{causal_results['n']['n_dropped_treatment_na']} sin T observado, "
            f"{support['n_excluidos_estratos']} en estratos sin controles y "
            f"{support['n_excluidos_trimming']} excluidos por trimming del propensity "
            f"{support['ps_bounds']}.".replace(",", ".")
        )
        share_fuera = causal_results["overlap_full_sample"]["share_outside_05_95"]
        st.metric("% fuera de [0,05; 0,95] (muestra completa)", f"{share_fuera:.1%}")

        overlap = causal_results["overlap"]
        quant_df = pd.DataFrame(
            {
                "cuantil": overlap["quantiles"],
                "propensity | T=1": overlap["ps_treated"],
                "propensity | T=0": overlap["ps_control"],
            }
        )
        st.caption("Cuantiles del propensity score por grupo, en soporte común:")
        st.dataframe(quant_df.round(3), hide_index=True, use_container_width=True)

    with col_refut:
        st.markdown("**Semáforo de refutaciones**")
        for r in causal_results["refutations"]:
            color = SEMAFORO_COLOR.get(r["semaforo"], "#999999")
            st.markdown(
                f"<div style='padding:8px;border-left:6px solid {color};margin-bottom:6px;'>"
                f"<b>{r['refutador']}</b>: {r['ate_nuevo'] * 100:.2f} pp"
                f"<br><span style='font-size:0.85em;color:#666;'>{r['interpretacion']}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("**Curva de CATE por perfil**")
    cate_grid = load_cate_grid()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        complain_sel = st.selectbox("Complain", sorted(cate_grid["Complain"].unique()))
    with col_b:
        satisfaction_sel = st.selectbox("SatisfactionScore", sorted(cate_grid["SatisfactionScore"].unique()))
    with col_c:
        estimator_sel = st.radio("Estimador", ["linear_dml", "causal_forest"], horizontal=True)

    subset = cate_grid[
        (cate_grid["Complain"] == complain_sel) & (cate_grid["SatisfactionScore"] == satisfaction_sel)
    ].sort_values("Tenure")
    cate_col, lo_col, hi_col = f"cate_{estimator_sel}", f"ci_inf_{estimator_sel}", f"ci_sup_{estimator_sel}"

    fig_cate = go.Figure()
    fig_cate.add_trace(
        go.Scatter(
            x=pd.concat([subset["Tenure"], subset["Tenure"][::-1]]),
            y=pd.concat([subset[hi_col], subset[lo_col][::-1]]),
            fill="toself",
            fillcolor="rgba(46,111,149,0.2)",
            line={"color": "rgba(255,255,255,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_cate.add_trace(
        go.Scatter(
            x=subset["Tenure"], y=subset[cate_col], mode="lines", line={"color": "#2e6f95"}, name="CATE"
        )
    )
    fig_cate.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_cate.update_layout(
        height=380,
        xaxis_title="Tenure (meses)",
        yaxis_title="CATE sobre churn (proporción, negativo = el cupón retiene)",
    )
    st.plotly_chart(fig_cate, use_container_width=True)

    policy = causal_results["policy"]
    churn_evitado = policy["churn_evitado_por_100_clientes_targeting"]
    efecto_a_todos = policy["efecto_medio_dar_a_todos_pp"]
    st.caption(
        f"Política simulada: targetear solo donde CATE<0 cubre {policy['share_cate_negativo']:.1%} de los "
        f"clientes y evitaría ~{churn_evitado:.2f} churns por cada 100 clientes targeteados, frente a dar "
        f"el incentivo a todos ({efecto_a_todos:.2f} pp en promedio). Ningún CATE por segmento es "
        "significativo (ver BITACORA.md, Entrada 4 §4.4): no se recomienda esta política sin "
        "validación experimental (A/B test)."
    )

# TODO(Bonus): 4ª pestaña "Copiloto" con tool use sobre el motor causal y guardrails anti-invención.

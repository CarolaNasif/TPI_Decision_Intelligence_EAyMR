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

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "data" / "processed" / "dataset_clean.parquet"

# Nombres de negocio de las componentes, interpretados en Fase 2 (BITACORA.md, Entrada 2 §2.2).
PC_LABELS = {
    "PC1": "intensidad transaccional",
    "PC2": "antigüedad vs. uso multi-dispositivo",
}

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


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


@st.cache_data
def load_split_idx() -> dict:
    return json.loads((ARTIFACTS / "split_idx.json").read_text(encoding="utf-8"))


@st.cache_data
def pca_scores() -> pd.DataFrame:
    """Proyecta TODO el dataset sobre la PCA ajustada SOLO en Train (Test va 'a ciegas')."""
    pipeline = load_pca_pipeline()
    df = load_dataset()
    split = load_split_idx()
    numeric_cols = list(pipeline.feature_names_in_)
    scores = pipeline.transform(df[numeric_cols])
    out = df[[split["id_col"], "Churn"]].copy()
    out["PC1"] = scores[:, 0]
    out["PC2"] = scores[:, 1]
    train_ids = set(split["train"])
    out["particion"] = [
        "Train" if cid in train_ids else "Test" for cid in out[split["id_col"]]
    ]
    return out


@st.cache_data
def pca_loadings() -> pd.DataFrame:
    pipeline = load_pca_pipeline()
    pca = pipeline.named_steps["pca"]
    return pd.DataFrame(
        pca.components_[:2].T,
        index=list(pipeline.feature_names_in_),
        columns=["PC1", "PC2"],
    )


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
    KAGGLE_URL = (
        "https://www.kaggle.com/datasets/"
        "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
    )
    LICENCIA_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    st.caption(
        f"Fuente: **E-Commerce Customer Churn** de Ankit Verma ([Kaggle]({KAGGLE_URL})) · "
        f"Licencia [CC BY-NC-SA 4.0]({LICENCIA_URL}) (uso académico, no comercial) · "
        "Se muestra la **versión limpia** (N = 2.982): se eliminaron 2.638 copias casi duplicadas "
        "(`CustomerID` + 1470) y 10 duplicados exactos, ver BITACORA.md Entrada 3."
    )

    df_full = load_dataset()
    split_idx = load_split_idx()
    id_col = split_idx["id_col"]

    st.markdown("#### Filtros de la muestra")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        churn_sel = st.radio("Churn", ["Todos", "Solo churn", "Solo retenidos"], index=0)
    with f2:
        city_sel = st.multiselect(
            "CityTier", sorted(df_full["CityTier"].astype(str).unique()), default=[]
        )
    with f3:
        cat_sel = st.multiselect(
            "PreferedOrderCat", sorted(df_full["PreferedOrderCat"].astype(str).unique()), default=[]
        )
    with f4:
        t_min, t_max = int(df_full["Tenure"].min()), int(df_full["Tenure"].max())
        tenure_sel = st.slider("Tenure (meses)", t_min, t_max, (t_min, t_max))

    mask = pd.Series(True, index=df_full.index)
    if churn_sel == "Solo churn":
        mask &= df_full["Churn"] == 1
    elif churn_sel == "Solo retenidos":
        mask &= df_full["Churn"] == 0
    if city_sel:
        mask &= df_full["CityTier"].astype(str).isin(city_sel)
    if cat_sel:
        mask &= df_full["PreferedOrderCat"].astype(str).isin(cat_sel)
    # Tenure tiene NA: se conservan salvo que el usuario recorte el rango.
    in_range = df_full["Tenure"].between(tenure_sel[0], tenure_sel[1])
    if (tenure_sel[0], tenure_sel[1]) != (t_min, t_max):
        mask &= in_range.fillna(False)
    df_sel = df_full[mask]

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Clientes seleccionados", f"{len(df_sel):,}".replace(",", "."))
    c_b.metric("Tasa de churn", f"{df_sel['Churn'].mean():.1%}" if len(df_sel) else "—")
    c_c.metric("Sobre el total", f"{len(df_sel) / len(df_full):.0%}")

    if df_sel.empty:
        st.warning("Ningún cliente cumple los filtros. Ampliá la selección.")
    else:
        st.markdown("---")
        st.markdown("#### Inspección de variables")
        col_var, col_na = st.columns([1.3, 1])

        with col_var:
            numeric_cols = list(load_pca_pipeline().feature_names_in_)
            categorical_cols = [
                c for c in df_full.columns
                if c not in numeric_cols and c not in {id_col, "Churn", "coupon_any"}
            ]
            var = st.selectbox("Variable", numeric_cols + categorical_cols)
            fig_var = go.Figure()
            if var in numeric_cols:
                for churn_val, color, name in [(0, "#6a8caf", "Retenido"), (1, "#b23a48", "Churn")]:
                    serie = df_sel.loc[df_sel["Churn"] == churn_val, var].dropna()
                    fig_var.add_trace(
                        go.Histogram(x=serie, name=name, marker_color=color, opacity=0.65, nbinsx=30)
                    )
                fig_var.update_layout(barmode="overlay", yaxis_title="Clientes", xaxis_title=var)
            else:
                tab = (
                    df_sel.groupby([df_sel[var].astype(str), "Churn"]).size().unstack(fill_value=0)
                )
                for churn_val, color, name in [(0, "#6a8caf", "Retenido"), (1, "#b23a48", "Churn")]:
                    if churn_val in tab.columns:
                        fig_var.add_trace(
                            go.Bar(x=tab.index, y=tab[churn_val], name=name, marker_color=color)
                        )
                fig_var.update_layout(barmode="group", yaxis_title="Clientes", xaxis_title=var)
            fig_var.update_layout(
                height=330, title=f"{var} según Churn", legend={"orientation": "h", "y": 1.15},
                margin={"t": 60},
            )
            st.plotly_chart(fig_var, use_container_width=True)

        with col_na:
            na = (
                df_sel.drop(columns=[id_col])
                .isna()
                .sum()
                .rename("n_faltantes")
                .to_frame()
                .assign(pct=lambda d: (d["n_faltantes"] / len(df_sel) * 100).round(2))
                .query("n_faltantes > 0")
                .sort_values("n_faltantes", ascending=False)
            )
            st.markdown("**Faltantes en la selección**")
            if na.empty:
                st.info("Sin faltantes en la selección actual.")
            else:
                st.dataframe(na, use_container_width=True)
                st.caption(
                    "Mecanismo MAR (no MCAR): el test de Little rechaza y el indicador de NA es "
                    "predecible desde el resto (ver BITACORA.md, Entrada 2, fila 2). Por eso la "
                    "imputación va con indicador binario y **dentro** del Pipeline, ajustada solo en Train."
                )

        st.markdown("---")
        st.markdown("#### Proyección PCA (PC1-PC2)")
        pipeline = load_pca_pipeline()
        evr = pipeline.named_steps["pca"].explained_variance_ratio_
        st.caption(
            f"`artifacts/pca.joblib` = `SimpleImputer(mediana) → StandardScaler → PCA`, **ajustado solo "
            f"con Train** (2.385 clientes); a Test se le aplica `.transform()` a ciegas. "
            f"PC1 explica {evr[0]:.1%} de la varianza y PC2 {evr[1]:.1%} "
            f"({evr[0] + evr[1]:.1%} acumulado)."
        )

        scores = pca_scores()
        scores_sel = scores[scores[id_col].isin(df_sel[id_col])]
        mostrar_particion = st.checkbox("Distinguir Train / Test", value=True)

        fig_pca = go.Figure()
        for churn_val, color, name in [(0, "#6a8caf", "Retenido"), (1, "#b23a48", "Churn")]:
            for particion, symbol in [("Train", "circle"), ("Test", "x")]:
                sub = scores_sel[(scores_sel["Churn"] == churn_val)]
                if mostrar_particion:
                    sub = sub[sub["particion"] == particion]
                elif particion == "Test":
                    continue
                if sub.empty:
                    continue
                etiqueta = f"{name} · {particion}" if mostrar_particion else name
                fig_pca.add_trace(
                    go.Scatter(
                        x=sub["PC1"], y=sub["PC2"], mode="markers", name=etiqueta,
                        marker={"color": color, "size": 6, "opacity": 0.5, "symbol": symbol},
                    )
                )
        fig_pca.add_hline(y=0, line_color="lightgrey", line_width=1)
        fig_pca.add_vline(x=0, line_color="lightgrey", line_width=1)
        fig_pca.update_layout(
            height=480,
            xaxis_title=f"PC1 — {PC_LABELS['PC1']} ({evr[0]:.1%})",
            yaxis_title=f"PC2 — {PC_LABELS['PC2']} ({evr[1]:.1%})",
            legend={"orientation": "h", "y": -0.18},
            margin={"t": 30},
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        load_df = pca_loadings()
        col_l1, col_l2 = st.columns(2)
        for col, pc in [(col_l1, "PC1"), (col_l2, "PC2")]:
            with col:
                top = load_df[pc].reindex(load_df[pc].abs().sort_values(ascending=False).index).head(5)
                fig_l = go.Figure(
                    go.Bar(
                        x=top.to_numpy(), y=top.index, orientation="h",
                        marker_color=["#b23a48" if v < 0 else "#2e6f95" for v in top],
                    )
                )
                fig_l.update_layout(
                    height=260, title=f"{pc} — {PC_LABELS[pc]}", xaxis_title="carga",
                    yaxis={"autorange": "reversed"}, margin={"t": 50},
                )
                st.plotly_chart(fig_l, use_container_width=True)

        st.caption(
            "**Lectura:** PC1 junta con el mismo signo a `OrderCount`, `CouponUsed`, `CashbackAmount` y "
            "`DaySinceLastOrder`: es un eje de *cuánto transacciona* el cliente. PC2 opone `Tenure` (+) a "
            "`HourSpendOnApp` y `NumberOfDeviceRegistered` (−): separa a los clientes antiguos de los de "
            "uso intensivo multi-dispositivo. No se observa una separación limpia de Churn en este plano "
            "(los dos colores se superponen): la PCA acá es descriptiva, no un clasificador — para predecir "
            "se usan las 12 variables originales en el Módulo 2."
        )

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

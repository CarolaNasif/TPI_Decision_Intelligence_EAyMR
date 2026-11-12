"""Decision Intelligence Capstone - Web App (Streamlit).

ESQUELETO BASE - sin lógica de modelado (sprint de diseño 12/11/2026).
Dueño: Frontend & UI/UX Engineer. La app SOLO consume artefactos serializados en
/artifacts (joblib/json) generados por el notebook maestro vía `src/`. Nunca re-entrena.

Ejecutar localmente:  streamlit run app/app.py
"""

from pathlib import Path

import streamlit as st

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

st.set_page_config(page_title="Decision Intelligence Capstone", page_icon="🧭", layout="wide")
st.title("🧭 Decision Intelligence Capstone — Retorno causal del motor de recomendaciones")
st.caption("EAyMR 2026 · UCA · P(Y | X) vs P(Y | do(T = t))")

with st.sidebar:
    st.header("Proyecto")
    st.markdown("**Dataset:** _a definir_  \n**T:** _a definir_  \n**Y:** _a definir_")
    st.info("Artefactos esperados en /artifacts: preprocessor.joblib, pca.joblib, "
            "best_model.joblib, dag.dot, causal_results.json, cate_grid.parquet")

tab_diag, tab_pred, tab_causal = st.tabs(
    ["1 · Diagnóstico & PCA", "2 · Simulador Predictivo (What-Is)", "3 · Simulador Causal (What-If)"]
)

with tab_diag:
    st.subheader("Diagnóstico y exploración")
    # TODO(Frontend): filtros de muestra, histogramas/boxplots, mapa de faltantes,
    # proyección PC1-PC2 (pca.joblib ajustado SOLO en Train) coloreada por Y.
    st.warning("Pendiente: requiere artifacts/pca.joblib")

with tab_pred:
    st.subheader("¿Qué es probable que pase? — P(Y | X)")
    # TODO(Frontend): formulario de un nuevo cliente -> probabilidad calibrada del mejor modelo,
    # intervalo (bootstrap/conformal) y explicación local de importancia.
    st.warning("Pendiente: requiere artifacts/best_model.joblib")

with tab_causal:
    st.subheader("¿Qué pasa si intervenimos? — P(Y | do(T = t))")
    # TODO(Frontend): DAG interactivo (graphviz), ATE naive vs backdoor vs IPW con IC,
    # curva CATE según perfil seleccionado, semáforo de 3 refutaciones (verde/amarillo/rojo).
    st.warning("Pendiente: requiere artifacts/causal_results.json")

# TODO(Bonus): 4ª pestaña "Copiloto" con tool use sobre el motor causal y guardrails anti-invención.

"""Paquete `src` del TPI Decision Intelligence Capstone (EAyMR 2026 - UCA).

Módulos:
    preprocessing : carga, contrato de datos, split estratificado, imputación, PCA sin leakage.
    inference     : Fase 1-2 (contrastes, ANOVA/Welch/Kruskal, Gauss-Markov, outliers, VIF).
    models        : Fase 3 (torneo predictivo: Logit/Ridge, Naive Bayes, CART+ccp_alpha, RF+OOB).
    causal        : Fase 4 (DAG, DoWhy 4 pasos, EconML DML/CATE, refutaciones).

Regla de arquitectura: el notebook maestro y la app IMPORTAN desde `src`;
nunca se duplica lógica de modelado en celdas sueltas ni en `app/app.py`.
"""

__version__ = "0.1.0"

# Semilla global única del proyecto (reproducibilidad - rúbrica 4.4).
RANDOM_STATE: int = 42

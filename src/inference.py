"""Inferencia clásica y diagnóstico econométrico (Fases 1 y 2).

ESQUELETO - todavía sin implementación (sprint de diseño 12/11/2026).
Dueño: Backend & ML Engineer. Revisor: QA & Validation Engineer.

Convenciones:
    - alpha se fija A PRIORI (0.05) y se declara en la Bitácora (Entrada 2).
    - Todo contraste devuelve: estadístico, p-valor exacto, IC 95% y tamaño de efecto.
    - Con N >= 1500 NO se decide normalidad por Shapiro-Wilk: se usa asimetría,
      curtosis y Q-Q plot (el test formal se reporta solo como evidencia de hipersensibilidad).
"""

from __future__ import annotations


def cohens_d(x, y):
    """d de Cohen con desvío combinado (pooled). TODO."""
    raise NotImplementedError


def two_sample_test(x, y, alpha: float = 0.05):
    """Welch t-test + Mann-Whitney U (robustez) + IC 95% de la diferencia + Cohen's d. TODO."""
    raise NotImplementedError


def chi2_independence(df, col_a: str, col_b: str):
    """Chi-cuadrado de independencia + V de Cramér. TODO."""
    raise NotImplementedError


def normality_diagnostics(x):
    """Asimetría, curtosis (exceso), Shapiro/D'Agostino solo informativos, datos para Q-Q. TODO."""
    raise NotImplementedError


def one_way_anova_suite(df, value_col: str, group_col: str):
    """ANOVA clásico + Levene (center='median') + Welch ANOVA + Kruskal-Wallis + eta²/omega².

    TODO: pingouin.welch_anova / scipy.stats.kruskal; devolver tabla comparativa.
    """
    raise NotImplementedError


def posthoc(df, value_col: str, group_col: str, method: str = "games-howell"):
    """Post-hoc: Tukey HSD (homocedástico), Games-Howell (heterocedástico) o Bonferroni. TODO."""
    raise NotImplementedError


def fit_ols(df, formula: str):
    """MCO con statsmodels (R², R² aj., F, t-tests) + versión con errores HC3. TODO."""
    raise NotImplementedError


def gauss_markov_diagnostics(ols_result):
    """Breusch-Pagan, White, Jarque-Bera, Durbin-Watson. TODO."""
    raise NotImplementedError


def influence_diagnostics(ols_result):
    """Residuos estudentizados externos (|r|>3), leverage (h>2(p+1)/n), Cook (D>4/n, D>0.5). TODO."""
    raise NotImplementedError


def vif_table(X):
    """VIF_j = 1/(1-R_j²) con constante; marca >5 y >10. TODO."""
    raise NotImplementedError

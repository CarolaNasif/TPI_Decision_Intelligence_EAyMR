"""Inferencia clásica y diagnóstico econométrico (Fases 1 y 2).

Fase 1 implementada por: Project Lead (Carola Nasif).
Fase 2 (`fit_ols`, `gauss_markov_diagnostics`, `influence_diagnostics`, `vif_table`):
    pendiente de implementación por QA & Validation Engineer (Emilia Mortarini).

Convenciones:
    - alpha se fija A PRIORI (0.05) y se declara en la Bitácora (Entrada 2).
    - Todo contraste devuelve: estadístico, p-valor exacto, IC 95% y tamaño de efecto.
    - Con N >= 1500 NO se decide normalidad por Shapiro-Wilk: se usa asimetría,
      curtosis y Q-Q plot (el test formal se reporta solo como evidencia de hipersensibilidad).
    - Nunca se afirma que se "acepta" H0: solo "Se rechaza H0" o "No se rechaza H0".
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep

from src import RANDOM_STATE


def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    """Devuelve el primer nombre de columna presente (compatibilidad pingouin 0.5/0.6)."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Ninguna de las columnas {candidates} está en {list(df.columns)}")


def _decision_text(p_value: float, alpha: float) -> str:
    """Etiqueta inferencial estándar (nunca afirma 'aceptar H0')."""
    if not np.isfinite(p_value):
        return "No evaluable"
    return "Se rechaza H0" if p_value < alpha else "No se rechaza H0"


def _classify_cohens_d(d: float) -> str:
    """Clasificación convencional de magnitud para |d| de Cohen."""
    if not np.isfinite(d):
        return "no evaluable"
    ad = abs(d)
    if ad < 0.2:
        return "despreciable"
    if ad < 0.5:
        return "pequeño"
    if ad < 0.8:
        return "mediano"
    return "grande"


def cohens_d(x, y) -> float:
    """d de Cohen con desvío combinado (pooled) entre dos muestras independientes."""
    x_arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    y_arr = np.asarray(pd.Series(y).dropna(), dtype=float)
    nx, ny = len(x_arr), len(y_arr)
    if nx < 2 or ny < 2:
        raise ValueError("Se requieren al menos 2 observaciones por grupo para calcular Cohen's d.")
    vx = float(np.var(x_arr, ddof=1))
    vy = float(np.var(y_arr, ddof=1))
    pooled_var = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled_var <= 0.0:
        return 0.0 if float(np.mean(x_arr)) == float(np.mean(y_arr)) else float("nan")
    return float((np.mean(x_arr) - np.mean(y_arr)) / np.sqrt(pooled_var))


def two_sample_test(x, y, alpha: float = 0.05, n_boot: int = 5000) -> dict[str, Any]:
    """Welch t-test + Mann-Whitney U + IC 95% (analítico y bootstrap) + Cohen's d + potencia."""
    x_arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    y_arr = np.asarray(pd.Series(y).dropna(), dtype=float)
    nx, ny = len(x_arr), len(y_arr)
    if nx < 2 or ny < 2:
        raise ValueError("Se requieren al menos 2 observaciones por grupo.")

    mean_x, mean_y = float(np.mean(x_arr)), float(np.mean(y_arr))
    std_x, std_y = float(np.std(x_arr, ddof=1)), float(np.std(y_arr, ddof=1))
    median_x, median_y = float(np.median(x_arr)), float(np.median(y_arr))
    iqr_x = float(np.percentile(x_arr, 75) - np.percentile(x_arr, 25))
    iqr_y = float(np.percentile(y_arr, 75) - np.percentile(y_arr, 25))
    diff_means = mean_x - mean_y
    se_diff = float(np.sqrt((std_x**2) / nx + (std_y**2) / ny))

    t_res = stats.ttest_ind(x_arr, y_arr, equal_var=False)
    t_stat = float(t_res.statistic)
    p_value = float(t_res.pvalue)
    df_welch = float(t_res.df)
    ci = t_res.confidence_interval(confidence_level=1.0 - alpha)
    ci_low, ci_high = float(ci.low), float(ci.high)

    rng = np.random.default_rng(RANDOM_STATE)
    boot_diffs = np.empty(n_boot, dtype=float)
    chunk = 500
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        m = end - start
        bx = x_arr[rng.integers(0, nx, size=(m, nx))].mean(axis=1)
        by = y_arr[rng.integers(0, ny, size=(m, ny))].mean(axis=1)
        boot_diffs[start:end] = bx - by
    boot_ci_low, boot_ci_high = (
        float(v) for v in np.quantile(boot_diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    )

    mw_res = stats.mannwhitneyu(x_arr, y_arr, alternative="two-sided")
    u_stat, p_mannwhitney = float(mw_res.statistic), float(mw_res.pvalue)

    d = cohens_d(x_arr, y_arr)
    power_calc = TTestIndPower()
    ratio = ny / nx
    if np.isfinite(d) and abs(d) > 0:
        power_observed = float(
            power_calc.power(
                effect_size=abs(d), nobs1=nx, ratio=ratio, alpha=alpha, alternative="two-sided"
            )
        )
    else:
        power_observed = float(alpha)
    mde_d_80 = float(
        power_calc.solve_power(
            effect_size=None, nobs1=nx, ratio=ratio, power=0.80, alpha=alpha, alternative="two-sided"
        )
    )

    return {
        "n_x": nx,
        "n_y": ny,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "median_x": median_x,
        "median_y": median_y,
        "iqr_x": iqr_x,
        "iqr_y": iqr_y,
        "diff_means": diff_means,
        "se_diff": se_diff,
        "t_stat": t_stat,
        "df_welch": df_welch,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_ci_low": boot_ci_low,
        "boot_ci_high": boot_ci_high,
        "u_stat": u_stat,
        "p_mannwhitney": p_mannwhitney,
        "cohens_d": d,
        "effect_magnitude": _classify_cohens_d(d),
        "power_observed": power_observed,
        "mde_d_80": mde_d_80,
        "alpha": alpha,
        "reject_h0": bool(p_value < alpha),
        "decision": _decision_text(p_value, alpha),
    }


def chi2_independence(
    df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05
) -> dict[str, Any]:
    """Chi-cuadrado de independencia + V de Cramér + diferencia de proporciones (IC 95%)."""
    sub = df[[col_a, col_b]].dropna()
    ct = pd.crosstab(sub[col_a], sub[col_b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Se requieren al menos 2 categorías por variable para chi-cuadrado.")

    chi2_val, p_value, dof, expected = stats.chi2_contingency(ct, correction=False)
    n = int(ct.to_numpy().sum())
    k_min = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2_val / (n * k_min))) if n > 0 and k_min > 0 else 0.0
    min_expected = float(np.min(expected))

    prop_1 = float("nan")
    prop_0 = float("nan")
    diff_prop = float("nan")
    prop_ci_low = float("nan")
    prop_ci_high = float("nan")
    odds_ratio = float("nan")

    if ct.shape == (2, 2):
        c1, n1 = int(ct.iloc[-1, -1]), int(ct.iloc[-1].sum())
        c0, n0 = int(ct.iloc[0, -1]), int(ct.iloc[0].sum())
        prop_1 = float(c1 / n1)
        prop_0 = float(c0 / n0)
        diff_prop = float(prop_1 - prop_0)
        lo, hi = confint_proportions_2indep(c1, n1, c0, n0, method="wald", alpha=alpha)
        prop_ci_low, prop_ci_high = float(lo), float(hi)
        if c0 > 0 and (n1 - c1) > 0:
            odds_ratio = float((c1 * (n0 - c0)) / (c0 * (n1 - c1)))

    return {
        "n": n,
        "chi2": float(chi2_val),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "min_expected": min_expected,
        "prop_1": prop_1,
        "prop_0": prop_0,
        "diff_proportions": diff_prop,
        "prop_ci_low": prop_ci_low,
        "prop_ci_high": prop_ci_high,
        "odds_ratio": odds_ratio,
        "contingency_table": ct,
        "alpha": alpha,
        "reject_h0": bool(p_value < alpha),
        "decision": _decision_text(float(p_value), alpha),
    }


def normality_diagnostics(x, alpha: float = 0.05) -> dict[str, Any]:
    """Asimetría, curtosis (exceso), Shapiro/D'Agostino solo informativos y datos para Q-Q."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = int(len(arr))
    if n < 3:
        raise ValueError("Se requieren al menos 3 observaciones para diagnóstico de normalidad.")

    skewness = float(stats.skew(arr, bias=False))
    kurtosis_excess = float(stats.kurtosis(arr, fisher=True, bias=False)) if n >= 4 else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sh_res = stats.shapiro(arr)
    shapiro_stat, shapiro_p = float(sh_res.statistic), float(sh_res.pvalue)

    if n >= 20:
        dag_res = stats.normaltest(arr)
        dagostino_stat, dagostino_p = float(dag_res.statistic), float(dag_res.pvalue)
    else:
        dagostino_stat, dagostino_p = float("nan"), float("nan")

    qq_theor, qq_samp = stats.probplot(arr, dist="norm", fit=False)
    is_acceptable_shape = bool(
        abs(skewness) <= 1.0 and np.isfinite(kurtosis_excess) and abs(kurtosis_excess) <= 2.0
    )

    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "shapiro_decision": _decision_text(shapiro_p, alpha),
        "dagostino_stat": dagostino_stat,
        "dagostino_p": dagostino_p,
        "is_acceptable_shape": is_acceptable_shape,
        "qq_theoretical": np.asarray(qq_theor, dtype=float),
        "qq_sample": np.asarray(qq_samp, dtype=float),
        "shapiro_note": (
            "Con N grande la potencia de Shapiro-Wilk tiende a 1 y rechaza H0 ante desvíos "
            "microscópicos; la decisión práctica se toma con asimetría in [-1, 1], curtosis "
            "en exceso in [-2, 2] y Q-Q plot (TCL cubre la media muestral)."
        ),
    }


def one_way_anova_suite(
    df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05
) -> dict[str, Any]:
    """ANOVA clásico + Levene (center='median') + Welch ANOVA + Kruskal-Wallis + eta²/omega²."""
    sub = df[[value_col, group_col]].dropna().copy()
    sub[group_col] = sub[group_col].astype(str)
    group_names = sorted(sub[group_col].unique())
    k = len(group_names)
    n = len(sub)
    if k < 2:
        raise ValueError("Se requieren al menos 2 grupos para ejecutar ANOVA.")

    diag_rows: list[dict[str, Any]] = []
    group_arrays: list[np.ndarray] = []
    for g in group_names:
        vals = sub.loc[sub[group_col] == g, value_col].to_numpy(dtype=float)
        group_arrays.append(vals)
        nd = normality_diagnostics(vals, alpha=alpha)
        diag_rows.append(
            {
                "group": g,
                "n": nd["n"],
                "mean": nd["mean"],
                "std": nd["std"],
                "median": nd["median"],
                "iqr": nd["iqr"],
                "skewness": nd["skewness"],
                "kurtosis_excess": nd["kurtosis_excess"],
                "shapiro_stat": nd["shapiro_stat"],
                "shapiro_p": nd["shapiro_p"],
                "is_acceptable_shape": nd["is_acceptable_shape"],
            }
        )
    group_diag = pd.DataFrame(diag_rows)

    vars_ = np.array([np.var(arr, ddof=1) for arr in group_arrays], dtype=float)
    var_ratio = float(vars_.max() / vars_.min()) if vars_.min() > 0 else float("inf")
    lev_stat, lev_p = stats.levene(*group_arrays, center="median")
    homoscedastic = bool(lev_p >= alpha)

    grand_mean = float(sub[value_col].mean())
    ss_between = float(sum(len(arr) * (float(arr.mean()) - grand_mean) ** 2 for arr in group_arrays))
    ss_within = float(sum(((arr - float(arr.mean())) ** 2).sum() for arr in group_arrays))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within if df_within > 0 else float("nan")

    anova_f, anova_p = stats.f_oneway(*group_arrays)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    omega_sq = (
        float(max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within)))
        if ss_total + ms_within > 0
        else 0.0
    )

    welch_df = pg.welch_anova(data=sub, dv=value_col, between=group_col)
    welch_p_col = _pick_col(welch_df, "p_unc", "p-unc")
    welch_f = float(welch_df.loc[0, "F"])
    welch_p = float(welch_df.loc[0, welch_p_col])
    welch_ddof1 = float(welch_df.loc[0, "ddof1"])
    welch_ddof2 = float(welch_df.loc[0, "ddof2"])

    kruskal_h, kruskal_p = stats.kruskal(*group_arrays)
    kruskal_df = k - 1
    epsilon_sq = float(max(0.0, (kruskal_h - k + 1) / (n - k))) if n > k else 0.0

    all_acceptable_shape = bool(group_diag["is_acceptable_shape"].all())
    if not all_acceptable_shape:
        recommended_test = "Kruskal-Wallis"
        recommended_posthoc = "bonferroni"
        primary_p = float(kruskal_p)
        rationale = (
            "Al menos un grupo excede |asimetría|<=1 o |curtosis_exceso|<=2; se prioriza "
            "Kruskal-Wallis (y se reporta Welch ANOVA como contraste de medias robusto)."
        )
    elif not homoscedastic:
        recommended_test = "Welch ANOVA"
        recommended_posthoc = "games-howell"
        primary_p = welch_p
        rationale = (
            "Asimetría y curtosis aceptables por grupo pero Levene (mediana) rechaza "
            "homocedasticidad; aplica Welch ANOVA + Games-Howell."
        )
    else:
        recommended_test = "ANOVA clásico"
        recommended_posthoc = "tukey"
        primary_p = float(anova_p)
        rationale = (
            "Normalidad práctica aceptable y Levene (mediana) no rechaza homocedasticidad; "
            "aplica ANOVA clásico + Tukey HSD."
        )

    summary_table = pd.DataFrame(
        [
            {
                "test": "ANOVA clásico (Fisher)",
                "statistic_name": "F",
                "statistic": float(anova_f),
                "df1": float(df_between),
                "df2": float(df_within),
                "p_value": float(anova_p),
                "effect_size_name": "omega_sq",
                "effect_size": omega_sq,
                "decision": _decision_text(float(anova_p), alpha),
            },
            {
                "test": "Welch ANOVA",
                "statistic_name": "F_welch",
                "statistic": welch_f,
                "df1": welch_ddof1,
                "df2": welch_ddof2,
                "p_value": welch_p,
                "effect_size_name": "eta_sq",
                "effect_size": eta_sq,
                "decision": _decision_text(welch_p, alpha),
            },
            {
                "test": "Kruskal-Wallis",
                "statistic_name": "H",
                "statistic": float(kruskal_h),
                "df1": float(kruskal_df),
                "df2": float("nan"),
                "p_value": float(kruskal_p),
                "effect_size_name": "epsilon_sq",
                "effect_size": epsilon_sq,
                "decision": _decision_text(float(kruskal_p), alpha),
            },
        ]
    )

    return {
        "n": n,
        "k": k,
        "alpha": alpha,
        "group_diagnostics": group_diag,
        "all_acceptable_shape": all_acceptable_shape,
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "levene_decision": _decision_text(float(lev_p), alpha),
        "homoscedastic": homoscedastic,
        "var_ratio": var_ratio,
        "anova_f": float(anova_f),
        "anova_p": float(anova_p),
        "df_between": df_between,
        "df_within": df_within,
        "welch_f": welch_f,
        "welch_p": welch_p,
        "welch_ddof1": welch_ddof1,
        "welch_ddof2": welch_ddof2,
        "kruskal_h": float(kruskal_h),
        "kruskal_p": float(kruskal_p),
        "kruskal_df": kruskal_df,
        "eta_sq": eta_sq,
        "omega_sq": omega_sq,
        "epsilon_sq_kruskal": epsilon_sq,
        "recommended_test": recommended_test,
        "recommended_posthoc": recommended_posthoc,
        "rationale": rationale,
        "reject_h0": bool(primary_p < alpha),
        "decision": _decision_text(primary_p, alpha),
        "summary_table": summary_table,
    }


def posthoc(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    method: str = "games-howell",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Post-hoc: Tukey HSD (homocedástico), Games-Howell (heterocedástico) o Bonferroni."""
    sub = df[[value_col, group_col]].dropna().copy()
    sub[group_col] = sub[group_col].astype(str)
    groups = sorted(sub[group_col].unique())
    k = len(groups)
    if k < 2:
        raise ValueError("Se requieren al menos 2 grupos para comparaciones post-hoc.")

    m = k * (k - 1) // 2
    fwer_uncorrected = float(1.0 - (1.0 - alpha) ** m)
    method_norm = method.lower().strip()

    if method_norm == "games-howell":
        gh = pg.pairwise_gameshowell(data=sub, dv=value_col, between=group_col)
        mean_a_col = _pick_col(gh, "mean_A", "mean(A)")
        mean_b_col = _pick_col(gh, "mean_B", "mean(B)")
        q_crit = stats.studentized_range.ppf(1.0 - alpha, k, gh["df"].to_numpy(dtype=float))
        margin = (q_crit / np.sqrt(2.0)) * gh["se"].to_numpy(dtype=float)
        diff = gh["diff"].to_numpy(dtype=float)
        p_adj = gh["pval"].to_numpy(dtype=float)
        out = pd.DataFrame(
            {
                "A": gh["A"],
                "B": gh["B"],
                "mean_A": gh[mean_a_col],
                "mean_B": gh[mean_b_col],
                "diff": diff,
                "se": gh["se"],
                "stat": gh["T"],
                "p_adj": p_adj,
                "ci_low": diff - margin,
                "ci_high": diff + margin,
                "hedges": gh["hedges"],
                "reject_h0": p_adj < alpha,
                "decision": [_decision_text(float(p), alpha) for p in p_adj],
            }
        )
    elif method_norm == "tukey":
        tk_sm = pairwise_tukeyhsd(
            endog=sub[value_col].to_numpy(dtype=float), groups=sub[group_col].to_numpy(), alpha=alpha
        )
        tk_pg = pg.pairwise_tukey(data=sub, dv=value_col, between=group_col)
        mean_a_col = _pick_col(tk_pg, "mean_A", "mean(A)")
        mean_b_col = _pick_col(tk_pg, "mean_B", "mean(B)")
        # statsmodels devuelve diff = B - A; alineamos el signo con pingouin (A - B).
        sm_rows = {}
        for idx, (g1, g2) in enumerate(
            itertools.combinations(list(tk_sm.groupsunique), 2)
        ):
            sm_rows[(str(g1), str(g2))] = (
                float(tk_sm.confint[idx, 0]),
                float(tk_sm.confint[idx, 1]),
            )
        ci_lows, ci_highs = [], []
        for a, b, diff_val, se_val in zip(
            tk_pg["A"], tk_pg["B"], tk_pg["diff"], tk_pg["se"], strict=True
        ):
            if (str(a), str(b)) in sm_rows:
                lo_ba, hi_ba = sm_rows[(str(a), str(b))]
                # sm_rows tiene (mean_B - mean_A); invertimos para (mean_A - mean_B)
                ci_lows.append(-hi_ba)
                ci_highs.append(-lo_ba)
            elif (str(b), str(a)) in sm_rows:
                lo_ab, hi_ab = sm_rows[(str(b), str(a))]
                ci_lows.append(lo_ab)
                ci_highs.append(hi_ab)
            else:
                q_c = stats.studentized_range.ppf(1.0 - alpha, k, len(sub) - k)
                marg = (q_c / np.sqrt(2.0)) * float(se_val)
                ci_lows.append(float(diff_val) - marg)
                ci_highs.append(float(diff_val) + marg)
        p_tukey_col = _pick_col(tk_pg, "p_tukey", "p-tukey")
        p_adj = tk_pg[p_tukey_col].to_numpy(dtype=float)
        out = pd.DataFrame(
            {
                "A": tk_pg["A"],
                "B": tk_pg["B"],
                "mean_A": tk_pg[mean_a_col],
                "mean_B": tk_pg[mean_b_col],
                "diff": tk_pg["diff"],
                "se": tk_pg["se"],
                "stat": tk_pg["T"],
                "p_adj": p_adj,
                "ci_low": ci_lows,
                "ci_high": ci_highs,
                "hedges": tk_pg["hedges"],
                "reject_h0": p_adj < alpha,
                "decision": [_decision_text(float(p), alpha) for p in p_adj],
            }
        )
    elif method_norm == "bonferroni":
        pairs = list(itertools.combinations(groups, 2))
        raw_p_mw: list[float] = []
        raw_rows: list[dict[str, Any]] = []
        for g_a, g_b in pairs:
            xa = sub.loc[sub[group_col] == g_a, value_col].to_numpy(dtype=float)
            xb = sub.loc[sub[group_col] == g_b, value_col].to_numpy(dtype=float)
            t_res = stats.ttest_ind(xa, xb, equal_var=False)
            mw_res = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            diff_val = float(xa.mean() - xb.mean())
            se_val = float(np.sqrt(np.var(xa, ddof=1) / len(xa) + np.var(xb, ddof=1) / len(xb)))
            t_crit = float(stats.t.ppf(1.0 - alpha / (2.0 * m), df=float(t_res.df)))
            raw_p_mw.append(float(mw_res.pvalue))
            raw_rows.append(
                {
                    "A": g_a,
                    "B": g_b,
                    "mean_A": float(xa.mean()),
                    "mean_B": float(xb.mean()),
                    "diff": diff_val,
                    "se": se_val,
                    "stat": float(mw_res.statistic),
                    "ci_low": diff_val - t_crit * se_val,
                    "ci_high": diff_val + t_crit * se_val,
                }
            )
        reject_arr, p_adj_arr, _, _ = multipletests(raw_p_mw, alpha=alpha, method="bonferroni")
        out = pd.DataFrame(raw_rows)
        out["p_adj"] = p_adj_arr
        out["reject_h0"] = reject_arr
        out["decision"] = [_decision_text(float(p), alpha) for p in p_adj_arr]
    else:
        raise ValueError(
            f"Método post-hoc desconocido: {method!r}. "
            "Opciones válidas: 'games-howell', 'tukey', 'bonferroni'."
        )

    out.attrs["fwer_info"] = {
        "method": method_norm,
        "n_groups": k,
        "n_comparisons": m,
        "alpha": alpha,
        "fwer_uncorrected": fwer_uncorrected,
    }
    return out


def fit_ols(df: pd.DataFrame, formula: str, robust: str | None = None):
    """MCO de referencia con statsmodels (fórmula de Wilkinson/patsy).

    Se ajusta sobre `df` tal cual se le pase (en el notebook, siempre sobre Train).
    `robust`: None para errores clásicos, o "HC0"/"HC1"/"HC2"/"HC3" para errores
    robustos a heterocedasticidad (cuando Breusch-Pagan/White rechazan homocedasticidad).
    Expone R², R² ajustado, F global y p vía los atributos estándar del resultado
    (`.rsquared`, `.rsquared_adj`, `.fvalue`, `.f_pvalue`); ver `ols_coef_table` para
    la tabla de beta/t/p/IC por coeficiente.
    """
    import statsmodels.formula.api as smf

    model = smf.ols(formula, data=df)
    return model.fit(cov_type=robust) if robust else model.fit()


def ols_coef_table(ols_result, alpha: float = 0.05) -> pd.DataFrame:
    """Tabla de coeficientes del MCO: beta, error estándar, t, p exacto e IC (1-alpha)."""
    ci = ols_result.conf_int(alpha=alpha)
    ci.columns = ["ci_low", "ci_high"]
    pvalues = ols_result.pvalues.to_numpy()
    return pd.DataFrame(
        {
            "term": ols_result.params.index,
            "coef": ols_result.params.to_numpy(),
            "std_err": ols_result.bse.to_numpy(),
            "t": ols_result.tvalues.to_numpy(),
            "p_value": pvalues,
            "ci_low": ci["ci_low"].to_numpy(),
            "ci_high": ci["ci_high"].to_numpy(),
            "decision": [_decision_text(float(p), alpha) for p in pvalues],
        }
    )


def gauss_markov_diagnostics(ols_result, alpha: float = 0.05) -> dict[str, Any]:
    """Auditoría de supuestos de Gauss-Markov sobre un resultado de `fit_ols`.

    - Homocedasticidad: Breusch-Pagan y White.
    - Normalidad de residuos: Jarque-Bera (mismo caveat de N grande que Shapiro) + datos de Q-Q.
    - Independencia: Durbin-Watson, informativo (sin orden temporal/espacial en datos transversales).
    """
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    resid = np.asarray(ols_result.resid, dtype=float)
    exog = np.asarray(ols_result.model.exog, dtype=float)

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    white_stat, white_p, _, _ = het_white(resid, exog)
    jb_stat, jb_p, resid_skew, resid_kurtosis = jarque_bera(resid)
    dw_stat = float(durbin_watson(resid))
    qq_theor, qq_samp = stats.probplot(resid, dist="norm", fit=False)

    homoscedastic = bool(bp_p >= alpha and white_p >= alpha)

    return {
        "n": int(len(resid)),
        "alpha": alpha,
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "breusch_pagan_decision": _decision_text(float(bp_p), alpha),
        "white_stat": float(white_stat),
        "white_p": float(white_p),
        "white_decision": _decision_text(float(white_p), alpha),
        "homoscedastic": homoscedastic,
        "recommend_hc3": not homoscedastic,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "jarque_bera_decision": _decision_text(float(jb_p), alpha),
        "resid_skew": float(resid_skew),
        "resid_kurtosis_excess": float(resid_kurtosis) - 3.0,
        "durbin_watson": dw_stat,
        "durbin_watson_note": (
            "Informativo: con datos transversales el orden de las filas es arbitrario, "
            "por lo que Durbin-Watson no tiene la interpretación de autocorrelación para la "
            "que fue diseñado; se reporta solo porque lo pide la cátedra."
        ),
        "qq_theoretical": np.asarray(qq_theor, dtype=float),
        "qq_sample": np.asarray(qq_samp, dtype=float),
    }


def influence_diagnostics(ols_result, cook_alarm: float = 0.5) -> pd.DataFrame:
    """Descomposición de observaciones extremas de un resultado de `fit_ols`.

    Una fila por observación, distinguiendo los tres conceptos:
        - Outlier en Y: residuo estudentizado externo |r| > 3.
        - Leverage (outlier en X): h_ii > 2(p+1)/n.
        - Influencia: Cook D_i > 4/n (cribado) y D_i > `cook_alarm` (alarma, default 0.5).
    """
    from statsmodels.stats.outliers_influence import OLSInfluence

    infl = OLSInfluence(ols_result)
    n = int(ols_result.nobs)
    p = int(ols_result.df_model)  # cantidad de predictores, sin contar la constante
    leverage_threshold = 2.0 * (p + 1) / n
    cook_screen_threshold = 4.0 / n

    student_resid = np.asarray(infl.resid_studentized_external, dtype=float)
    leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
    cooks_d = np.asarray(infl.cooks_distance[0], dtype=float)

    is_outlier = np.abs(student_resid) > 3.0
    is_high_leverage = leverage > leverage_threshold
    is_influential = cooks_d > cook_screen_threshold
    is_cook_alarm = cooks_d > cook_alarm

    def _classify(outlier: bool, lev: bool, infl_flag: bool) -> str:
        tags = []
        if outlier:
            tags.append("outlier(Y)")
        if lev:
            tags.append("leverage(X)")
        if infl_flag:
            tags.append("influyente(Cook)")
        return "+".join(tags) if tags else "normal"

    classification = [
        _classify(o, lv, i)
        for o, lv, i in zip(is_outlier, is_high_leverage, is_influential, strict=True)
    ]

    row_labels = ols_result.model.data.row_labels
    obs_index = np.asarray(row_labels) if row_labels is not None else np.arange(n)

    out = pd.DataFrame(
        {
            "obs_index": obs_index,
            "studentized_resid": student_resid,
            "leverage": leverage,
            "cooks_d": cooks_d,
            "is_outlier": is_outlier,
            "is_high_leverage": is_high_leverage,
            "is_influential": is_influential,
            "is_cook_alarm": is_cook_alarm,
            "classification": classification,
        }
    )
    out.attrs["leverage_threshold"] = leverage_threshold
    out.attrs["cook_screen_threshold"] = cook_screen_threshold
    out.attrs["cook_alarm_threshold"] = cook_alarm
    return out


def vif_table(X: pd.DataFrame, add_constant: bool = True) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R_j²) de cada regresora contra el resto (incluye constante en el
    ajuste auxiliar por default, sin reportarla); marca > 5 (atención) y > 10 (alto)."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    design = sm.add_constant(X.astype(float), has_constant="add") if add_constant else X.astype(float)
    design_arr = design.to_numpy()

    rows = [
        {"variable": col, "VIF": float(variance_inflation_factor(design_arr, i))}
        for i, col in enumerate(design.columns)
        if not (add_constant and col == "const")
    ]
    out = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["flag"] = np.select([out["VIF"] > 10, out["VIF"] > 5], [">10", ">5"], default="OK")
    return out

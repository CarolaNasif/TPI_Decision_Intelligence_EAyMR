"""Gráficos de las slides, dibujados al tamaño real de inserción (PDF vectorial, Calibri 8 pt).

Uso: python figs.py <raíz del repo> <curves.json> <carpeta de salida>
Todos los números salen de los artefactos commiteados y del dataset limpio; las curvas ROC y de
calibración salen de reentrenar el torneo en memoria (mismas semillas: reproduce model_results.json).
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT, CURVES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from src import inference as inf  # noqa: E402
from src import preprocessing as prep  # noqa: E402

for f in ("calibri.ttf", "calibrib.ttf", "calibrii.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")

INK, MUTED, CORAL, TEAL, AMBER, SLATE = "#1B2A2F", "#6B7B7C", "#D9482B", "#22706B", "#C9821B", "#4F5F86"
GRID = "#DDE4E3"
mpl.rcParams.update(
    {
        "font.family": "Calibri",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "savefig.transparent": True,
    }
)
MM = 1 / 25.4


def es(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def comma(nd):
    return FuncFormatter(lambda v, _: es(v, nd))


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("ok", name)


df = pd.read_parquet(ROOT / "data" / "processed" / "dataset_clean.parquet")
model_results = json.loads((ROOT / "artifacts" / "model_results.json").read_text(encoding="utf-8"))
causal_results = json.loads((ROOT / "artifacts" / "causal_results.json").read_text(encoding="utf-8"))
curves = json.loads(CURVES.read_text(encoding="utf-8"))

# ------------------------------------------------------------------ Fase 1: churn según reclamo
g = df.groupby("Complain")["Churn"].agg(["mean", "size"])
p, n = g["mean"].to_numpy(), g["size"].to_numpy()
ci = 1.96 * np.sqrt(p * (1 - p) / n)
fig, ax = plt.subplots(figsize=(54 * MM, 42 * MM))
bars = ax.bar([0, 1], p * 100, yerr=ci * 100, color=[SLATE, CORAL], width=0.58,
              error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": INK})
for x, v, e in zip([0, 1], p * 100, ci * 100, strict=True):
    ax.text(x, v + e + 1.2, es(v, 1) + " %", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=[SLATE, CORAL][x])
names = ("Sin reclamo", "Con reclamo")
xlabels = [f"{name}\n(n = {k:,})".replace(",", ".") for name, k in zip(names, n, strict=True)]
ax.set_xticks([0, 1], xlabels)
ax.set_ylabel("Tasa de churn (%)")
ax.set_ylim(0, 44)
ax.set_yticks([0, 10, 20, 30, 40])
ax.tick_params(axis="x", length=0)
fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.2)
save(fig, "f1_reclamo.pdf")
print("   churn sin/con reclamo:", p.round(4), n)

# ------------------------------------------------------------------ Fase 2: residuos del MCO
X_train, _, _, _ = prep.load_split(prep.make_clean_dataset())
train = X_train.copy()
num_conf = [c for c in prep.CONFOUNDERS if c in prep.NUMERIC_COLS]
cat_conf = [c for c in prep.CONFOUNDERS if c in prep.CATEGORICAL_COLS]
for c in num_conf:
    if train[c].isna().any():
        train[c] = train[c].fillna(train[c].median())
formula = "CashbackAmount ~ " + " + ".join(num_conf + [f"C({c})" for c in cat_conf])
ols = inf.fit_ols(train, formula)
print("   MCO: N =", int(ols.nobs), "R2 =", round(ols.rsquared, 4))
(osm, osr), (slope, intercept, _) = stats.probplot(ols.resid, dist="norm")
fig, axes = plt.subplots(1, 2, figsize=(84 * MM, 40 * MM))
axes[0].scatter(osm, osr, s=3, color=SLATE, alpha=0.6, linewidths=0, rasterized=True)
xx = np.array([osm.min(), osm.max()])
axes[0].plot(xx, intercept + slope * xx, color=CORAL, lw=1.1)
axes[0].set_title("Q-Q de los residuos")
axes[0].set_xlabel("Cuantiles teóricos (normal)")
axes[0].set_ylabel("Residuos ($)")
axes[1].scatter(ols.fittedvalues, ols.resid, s=3, color=SLATE, alpha=0.35, linewidths=0, rasterized=True)
axes[1].axhline(0, color=CORAL, lw=1.0)
axes[1].set_title("Residuos vs. ajustados")
axes[1].set_xlabel("Cashback predicho ($)")
for a in axes:
    a.yaxis.set_major_formatter(comma(0))
fig.subplots_adjust(left=0.15, right=0.985, top=0.87, bottom=0.23, wspace=0.36)
fig.savefig(OUT / "f2_residuos.pdf", dpi=450)
plt.close(fig)
print("ok f2_residuos.pdf")

# ------------------------------------------------------------------ Fase 3: curvas ROC en Test
order = [
    ("Random Forest + isotónica", "RF + isotónica", TEAL, 1.9),
    ("Logística ElasticNet", "Logística", SLATE, 1.1),
    ("CART podado", "CART podado", AMBER, 1.1),
    ("Naive Bayes", "Naive Bayes", CORAL, 1.1),
]
fig, ax = plt.subplots(figsize=(62 * MM, 41 * MM))
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
for key, label, color, lw in order:
    r = curves["roc"][key]
    ax.plot(r["fpr"], r["tpr"], color=color, lw=lw, label=f"{label} ({es(curves['auc'][key], 3)})")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(comma(1))
ax.yaxis.set_major_formatter(comma(1))
ax.legend(loc="lower right", frameon=False, title="AUC en Test", title_fontsize=7.5, handlelength=1.4,
          borderaxespad=0.2, labelspacing=0.3)
fig.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.22)
save(fig, "f3_roc.pdf")

# ------------------------------------------------------------------ Fase 3: calibración
rel = pd.DataFrame(curves["reliability"])
lines = [
    ("Random Forest", "RF sin calibrar", TEAL, "--", 1.0, 0.65),
    ("Random Forest + isotónica", "RF + isotónica", TEAL, "-", 1.8, 1.0),
    ("Naive Bayes", "Naive Bayes", CORAL, "-", 1.2, 1.0),
]
fig, ax = plt.subplots(figsize=(52 * MM, 40 * MM))
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
for key, label, color, ls, lw, alpha in lines:
    d = rel[rel["modelo"] == key].sort_values("prob_media")
    ax.plot(d["prob_media"], d["frecuencia_observada"], color=color, ls=ls, lw=lw, alpha=alpha, marker="o",
            ms=2.4, label=label)
ax.text(0.57, 0.83, "RF sin calibrar", color=TEAL, fontsize=7.5, ha="right", va="center", alpha=0.8)
ax.text(0.68, 0.60, "RF + isotónica", color=TEAL, fontsize=7.5, fontweight="bold", ha="left", va="center")
ax.text(0.62, 0.21, "Naive Bayes", color=CORAL, fontsize=7.5, ha="center", va="center")
ax.set_xlabel("Probabilidad predicha")
ax.set_ylabel("Frecuencia observada")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(comma(1))
ax.yaxis.set_major_formatter(comma(1))
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.subplots_adjust(left=0.19, right=0.95, top=0.97, bottom=0.23)
save(fig, "f3_calibracion.pdf")

# ------------------------------------------------------------------ Fase 3: MDI vs permutación
imp = pd.DataFrame(model_results["global_importance"])
top = set(imp.nsmallest(4, "rank_perm")["feature"]) | set(imp.nsmallest(4, "rank_mdi")["feature"])
top |= {"CouponUsed"}
sel = imp[imp["feature"].isin(top)].sort_values("rank_perm", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(84 * MM, 40 * MM), sharey=True)
y = np.arange(len(sel))
hot = (sel["feature"] == "CashbackAmount").to_numpy()
for ax, col, rank, base, title in (
    (axes[0], "mdi", "rank_mdi", MUTED, "MDI (calculada en Train)"),
    (axes[1], "perm_mean", "rank_perm", TEAL, "Permutación (en Test)"),
):
    vals = sel[col].to_numpy()
    ax.barh(y, vals, color=[CORAL if h else base for h in hot], height=0.62)
    span = vals.max()
    for yi, v, r, h in zip(y, vals, sel[rank], hot, strict=True):
        ax.text(max(v, 0) + span * 0.03, yi, f"#{int(r)}", va="center", ha="left", fontsize=7.5,
                fontweight="bold" if h else "normal", color=CORAL if h else INK)
    ax.set_title(title)
    ax.set_xlim(min(0, vals.min()) - span * 0.02, span * 1.22)
    ax.xaxis.set_major_formatter(comma(2))
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y, sel["feature"])
axes[0].set_xlabel("Importancia")
axes[1].set_xlabel("Caída del AUC")
for lab, h in zip(axes[0].get_yticklabels(), hot, strict=True):
    if h:
        lab.set_color(CORAL)
        lab.set_fontweight("bold")
fig.subplots_adjust(left=0.27, right=0.99, top=0.89, bottom=0.23, wspace=0.12)
save(fig, "f3_importancia.pdf")
print("   variables:", list(sel["feature"][::-1]))

# ------------------------------------------------------------------ Fase 4: forest plot
est = causal_results["estimates"]
rows = [
    ("Ingenuo (Y ~ T)", "Ingenuo, sin ajustar", MUTED),
    ("Backdoor: regresión lineal", "Regresión (backdoor)", CORAL),
    ("Backdoor: IPW", "IPW (propensity)", CORAL),
    ("LinearDML (EconML)", "LinearDML", SLATE),
    ("CausalForestDML (EconML)", "CausalForestDML", SLATE),
]
fig, ax = plt.subplots(figsize=(76 * MM, 44 * MM))
ax.axvline(0, color=INK, lw=0.8)
for i, (key, _label, color) in enumerate(rows):
    yv = len(rows) - 1 - i
    ate, (lo, hi) = est[key]["ate"] * 100, [v * 100 for v in est[key]["ci"]]
    ax.plot([lo, hi], [yv, yv], color=color, lw=1.4, solid_capstyle="butt")
    ax.plot([lo, lo], [yv - 0.11, yv + 0.11], color=color, lw=1.4)
    ax.plot([hi, hi], [yv - 0.11, yv + 0.11], color=color, lw=1.4)
    ax.plot(ate, yv, "o", color=color, ms=5)
    near0 = abs(ate) < 0.8
    ax.text(ate - (0.25 if near0 else 0), yv + 0.2, es(ate, 2, sign=True), ha="right" if near0 else "center",
            va="bottom", fontsize=8, fontweight="bold", color=color)
ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
ax.set_ylim(-1.05, len(rows) - 0.3)
ax.set_xlim(-6.5, 5.2)
ax.set_xlabel("Efecto sobre el churn, en pp (IC 95 %)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v, 0)))
ax.tick_params(axis="y", length=0)
ax.text(-6.3, -1.0, "← el cupón retiene", fontsize=7, color=MUTED, ha="left", va="bottom")
ax.text(5.1, -1.0, "aumenta el churn →", fontsize=7, color=MUTED, ha="right", va="bottom")
fig.subplots_adjust(left=0.33, right=0.98, top=0.99, bottom=0.2)
save(fig, "f4_forest.pdf")

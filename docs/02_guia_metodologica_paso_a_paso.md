# 02 · Guía Metodológica Paso a Paso (objetivo: Sobresaliente 90-100)

> Ejemplos instanciados sobre la **Opción A** (E-Commerce Churn). Si se elige otro dataset, se reemplazan nombres de variables; la lógica no cambia.

> [!NOTE]
> **Ponderación oficial (Rúbrica Detallada):** D1 Inferencia/ANOVA 15 · D2 Gauss-Markov/PCA 15 · D3 Torneo ML 20 · D4 Causal 25 · D5 Scrum/Bitácora 10 · D6 Deploy/Código/Defensa 15. Además rige la **Rúbrica complementaria** (Analizar · Modelar · Explorar · Informar), mapeada en §6.

---

## Principios transversales (aplican a todas las fases)

1. **Teoría antes que código:** cada bloque del notebook abre con una celda Markdown con la ecuación / hipótesis / supuesto que se va a testear (rúbrica compl. 2.1 "Abstracción conceptual vs código").
2. **α = 0,05 fijado a priori** y declarado en Bitácora Entrada 2. Reportar **p-valor exacto**, nunca "p < 0,05" a secas.
3. **Toda significancia va acompañada de magnitud** (Cohen's d, η², V de Cramér, OR con IC). Con N ≥ 1.500 todo es "significativo": lo que importa es el tamaño del efecto.
4. **Una única semilla** (`RANDOM_STATE = 42`) y **un único split** persistido (índices guardados en `artifacts/split_idx.json`).
5. **Cada gráfico:** título, ejes con unidades, leyenda, y una frase de interpretación debajo.

---

## FASE 1 — Inferencia clásica y ANOVA (15 pts)

### Paso 1.1 · Auditoría forense del DGP (rúbrica compl. 1.3)
- Tipado: continuas / discretas / categóricas / ordinales. Normalizar categorías duplicadas (`Phone`→`Mobile Phone`, `CC`→`Credit Card`, `COD`→`Cash on Delivery`).
- Rangos imposibles, outliers univariados (IQR y z robusto con MAD), duplicados de `CustomerID`.
- Univariado: histograma + KDE + boxplot por variable; asimetría y curtosis en tabla.
- Bivariado: matrices **Pearson y Spearman** lado a lado (la diferencia entre ambas delata no linealidad/outliers); variables vs `Churn` (boxplots, tasas por categoría).

### Paso 1.2 · Mecanismo de faltantes (MCAR / MAR / MNAR)
| Mecanismo | Definición | Evidencia empírica | Remediación |
|---|---|---|---|
| MCAR | $P(R \mid Y_{obs}, Y_{mis}) = P(R)$ | Test de Little no rechaza; indicador de NA no asociado a nada | Eliminación por lista aceptable (sin sesgo, solo pérdida de potencia) |
| MAR | $P(R \mid Y_{obs}, Y_{mis}) = P(R \mid Y_{obs})$ | Logit de `is_na_X ~ resto` con predictores significativos / AUC > 0,6 | Imputación condicional (Iterative/KNN) **dentro del Pipeline** + indicador de NA |
| MNAR | depende del propio valor faltante | No testeable directamente: argumento de dominio (p.ej. `Tenure` NA en clientes recién llegados) + análisis de sensibilidad | Indicador de NA como feature + escenarios (imputar extremo bajo/alto) |

- Herramientas: `missingno.matrix/heatmap`, χ² de NA vs `Churn`, logit por columna con NA.
- ⚠️ **Trampa:** imputar antes del split = leakage. La imputación se *ajusta* en Train.

### Paso 1.3 · Dos contrastes formales (mínimo) con relevancia de negocio
**H1 de negocio (ejemplo):** ¿Los clientes que churnean reciben menos cashback?
$$H_0: \mu_{cashback}^{churn=1} = \mu_{cashback}^{churn=0} \quad vs \quad H_1: \mu^{churn=1} \neq \mu^{churn=0}$$
- Estadístico: **Welch t** (no asume varianzas iguales) + **Mann-Whitney U** como robustez ante asimetría. Reportar t, gl de Welch, p exacto, IC 95% de la diferencia (y por bootstrap 5.000 réplicas), **Cohen's d** (0,2 / 0,5 / 0,8).

**H2 de negocio:** ¿El reclamo (`Complain`) está asociado con el churn? → χ² de independencia + **V de Cramér** + diferencia de proporciones con IC 95%.

**H3 opcional (Explorar 3.1):** análisis de potencia: con el N observado, ¿qué d mínimo detectamos con 1−β = 0,80? (`statsmodels.stats.power.TTestIndPower`). Muestra conciencia de diseño muestral (rúbrica compl. 1.4).

### Paso 1.4 · ANOVA de 1 factor (≥ 3 grupos)
- Ejemplo: `CashbackAmount ~ PreferedOrderCat` (5-6 categorías) o `~ CityTier` (3).
- **Normalidad:** ⚠️ **Trampa Shapiro-Wilk con N ≥ 1.500:** la potencia del test → 1 con N grande; rechaza H₀ ante desvíos microscópicos irrelevantes para la inferencia sobre medias (el TCL ya garantiza normalidad aproximada de $\bar X$). Además, `scipy` advierte que el p-valor de Shapiro no es exacto para N > 5.000. **Decisión:** se reporta Shapiro solo para *exhibir* su hipersensibilidad; se decide con **asimetría ∈ [−1, 1], curtosis en exceso ∈ [−2, 2] y Q-Q plots por grupo**.
- **Homocedasticidad:** **Levene con `center='median'`** (Brown-Forsythe, robusto a no normalidad). Reportar ratio var_max/var_min.
- **Árbol de decisión:**
  - Levene no rechaza + simetría aceptable → ANOVA clásico + **Tukey HSD**.
  - Levene rechaza + distribuciones razonablemente simétricas → **Welch ANOVA** + **Games-Howell** (Tukey asume varianzas iguales).
  - Asimetría severa / outliers dominantes → **Kruskal-Wallis** + Dunn con **Bonferroni/Holm**.
- **Tamaño del efecto:** η² y **ω²** (menos sesgado). Interpretar intervalos de las diferencias post-hoc en unidades de negocio (\$ de cashback).
- **Control del error de familia:** con k grupos hay k(k−1)/2 comparaciones; con 6 grupos, 15 → P(≥1 falso positivo) ≈ 1 − 0,95¹⁵ ≈ 54% sin corrección. Tukey controla FWER exacto; Bonferroni es conservador (α/m).

**Entregables Fase 1:** tabla de faltantes con mecanismo, 2+ contrastes con tabla estándar (estadístico, gl, p, IC, efecto), ANOVA con árbol de decisión explícito.

---

## FASE 2 — Gauss-Markov, outliers, VIF y PCA (15 pts)

### Paso 2.1 · MCO de referencia
- Y continua: `CashbackAmount` (o `log1p(CashbackAmount)` si la asimetría lo amerita; justificar). Predictores: Z numéricas + dummies (categoría de referencia explícita).
- Reportar R², R² ajustado, F global y p, tabla de β con t, p, IC 95%.
- **Se ajusta en Train** (coherencia con todo el pipeline).

### Paso 2.2 · Auditoría de supuestos
| Supuesto | Diagnóstico | Si falla |
|---|---|---|
| Linealidad | residuos vs ŷ, partial residual plots | transformaciones / términos cuadráticos |
| Homocedasticidad | **Breusch-Pagan** y **White** | errores robustos **HC3**; WLS |
| Normalidad de ε | Q-Q de residuos + **Jarque-Bera** (mismo caveat de N grande) | inferencia asintótica / bootstrap |
| Independencia | **Durbin-Watson** (≈2). ⚠️ Solo tiene sentido con orden temporal/espacial: con datos transversales, se reporta aclarando que el orden de filas es arbitrario | — |
| No multicolinealidad | **VIF** | ver 2.4 |

### Paso 2.3 · Descomposición de observaciones extremas (distinguir los 3 conceptos)
- **Outlier en Y:** residuo estudentizado externo $|r_i| > 3$.
- **Leverage (outlier en X):** $h_{ii} > 2(p+1)/n$.
- **Influencia:** Cook $D_i = \frac{e_i^2}{(p+1)\,MSE}\cdot\frac{h_{ii}}{(1-h_{ii})^2}$; umbrales $D_i > 4/n$ (screening) y $D_i > 0{,}5$ (alarma).
- Gráfico **influence plot** (r vs h, burbuja ∝ D). Clasificar cada punto: ¿error de medición (→ corregir/eliminar con justificación) o caso real (→ conservar y comparar con **HuberRegressor / RANSAC**)? Reportar cómo cambian los β.
- ⚠️ **Trampa:** eliminar outliers "porque sí" (nivel Aprobado). Siempre ajustar con y sin, y mostrar la diferencia.

### Paso 2.4 · Multicolinealidad
- VIF con constante incluida; marcar > 5 y > 10. Esperado: `CouponUsed`–`OrderCount`–`CashbackAmount`.
- Remediación justificada: eliminar redundante, combinar (ratio cupones/pedido) o **Ridge** (puente a Fase 3). Ojo: VIF alto **no sesga** β, infla su varianza.

### Paso 2.5 · PCA sin leakage
- `Pipeline([("imputer", SimpleImputer(median)), ("scaler", StandardScaler()), ("pca", PCA())])` → `.fit(X_train_num)` → `.transform(X_test_num)`.
- ⚠️ **Trampa del leakage:** estandarizar o hacer PCA con el dataset completo filtra media, varianza y dirección de máxima varianza de Test a Train. El test `test_pca_fitted_only_on_train` lo vigila.
- Scree plot + varianza acumulada; criterio **Kaiser (λ > 1)** vs codo vs 80% acumulado: discutir si coinciden.
- **Loadings:** heatmap de cargas; nombrar PC1 y PC2 en lenguaje de negocio (p.ej. PC1 = "intensidad transaccional": OrderCount, CouponUsed, Cashback; PC2 = "antigüedad/fidelidad": Tenure, NumberOfAddress).
- Biplot PC1-PC2 coloreado por `Churn` (insumo para el módulo 1 de la app).
- PCA solo con numéricas (PCA sobre dummies es discutible; mencionarlo).

---

## FASE 3 — Torneo predictivo y calibración (20 pts)

### Paso 3.1 · Protocolo
- Split estratificado 80/20 (ya hecho en Fase 1) + `StratifiedKFold(5, shuffle=True, random_state=42)` **solo en Train**.
- Todo modelo = `Pipeline(preprocessor, estimator)` → la imputación/escalado se re-ajusta en cada fold.
- Métrica de selección en CV: **ROC-AUC** (y PR-AUC por el desbalance ≈17%). Test se usa **una sola vez**.
- ⚠️ **Trampa:** excluir de los features a `CustomerID`. Discutir si incluir `DaySinceLastOrder` (casi-leakage temporal): **sí** en el torneo predictivo (con la advertencia), **nunca** en el set de ajuste causal.

### Paso 3.2 · Las 4 familias
1. **Logística regularizada** (L1, L2, ElasticNet; grilla en C y l1_ratio) + **Logit statsmodels** sin penalizar para **OR = e^β con IC 95%**. Mostrar senda de coeficientes vs C. Interpretación: "cada reclamo multiplica las odds de churn por X (IC: a–b)".
2. **Naive Bayes:** Gaussian para continuas (o Categorical tras discretizar). Diagnosticar la **violación de independencia condicional** con la matriz de correlación **dentro de cada clase** (p.ej. `OrderCount`–`CouponUsed` correlacionadas | Churn). Consecuencia: el producto $\prod P(X_i\mid Y)$ "cuenta dos veces" la evidencia → probabilidades empujadas a 0/1 → **mala calibración** aunque el ranking (AUC) sea aceptable. Mostrarlo en el reliability diagram y corregir con `CalibratedClassifierCV` (isotónica / Platt) como experimento extra.
3. **CART:** (a) árbol sin restricciones → AUC Train ≈ 1 vs Test mucho menor (**overfit documentado**, candidato para Bitácora Entrada 3); (b) `cost_complexity_pruning_path` → CV sobre la grilla de α → α_opt (opcional regla 1-SE); graficar AUC CV vs α y nº de hojas vs α; `plot_tree(max_depth=3)` e interpretar la regla del nodo raíz.
4. **Random Forest:** `RandomizedSearchCV` (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features ∈ {sqrt, log2, 0.3, 0.5}) + `oob_score=True`. Graficar **error OOB vs n_estimators** y compararlo con el AUC/accuracy de CV (deberían converger; diferencias → explicar).

### Paso 3.3 · Evaluación en Test
- Tabla: Precision, Recall, F1 (macro y ponderado), ROC-AUC, PR-AUC, **Brier score**, + IC 95% por bootstrap del AUC.
- Curvas ROC y PR superpuestas; **elección del umbral** con costo de negocio (falso negativo = cliente perdido vs falso positivo = cupón regalado).
- **Reliability diagrams** (10 bins, estrategia `quantile`) de los 4 modelos + ECE.

### Paso 3.4 · MDI vs Permutation Importance
- **MDI (Gini)** se calcula en Train sobre la reducción de impureza: variables continuas/alta cardinalidad ofrecen **muchos más puntos de corte candidatos** → más oportunidades de reducir impureza por azar → sesgo sistemático a su favor frente a dummies binarias (Strobl et al., 2007). Además se mide sobre datos de entrenamiento (refleja también sobreajuste).
- **Permutation importance en Test** (`n_repeats ≥ 20`, scoring ROC-AUC): mide la caída real de desempeño fuera de muestra → insesgada respecto de cardinalidad. Limitación: con features correlacionadas reparte/diluye importancia (mencionarlo).
- Gráfico lado a lado con barras de error + experimento control: **agregar una columna de ruido aleatorio continua**; si MDI le da importancia > 0 y permutación ≈ 0, el sesgo queda demostrado empíricamente. 🔥 (Muy valorado: "demuestra matemáticamente").

---

## FASE 4 — Inferencia causal y Causal ML (25 pts)

### Paso 4.1 · DAG formal
- Construir con `networkx.DiGraph` y exportar DOT (para DoWhy y la app). Ver DAG de `docs/01`.
- Tabla de arcos: cada flecha con justificación de dominio + temporalidad (Bitácora Entrada 4). Incluir **nodos latentes** (U_engagement) para discutir confusión no observada.

### Paso 4.2 · Criterio backdoor (demostración analítica)
Un conjunto $Z$ satisface el criterio backdoor relativo a $(T, Y)$ si: (i) ningún nodo de $Z$ es descendiente de $T$; (ii) $Z$ bloquea todo camino entre $T$ e $Y$ que entra a $T$ por una flecha ($T \leftarrow \dots$). Entonces:
$$P(Y \mid do(T=t)) = \sum_z P(Y \mid T=t, Z=z)\,P(Z=z)$$
- Enumerar los caminos backdoor (`T ← Tenure → Y`, `T ← Complain → Y`, …) y mostrar que cada uno se bloquea condicionando en el confusor (cadena/bifurcación). Verificar con d-separación en `networkx` (grafo con aristas salientes de T removidas).
- ⚠️ **Trampa del mediador:** incluir `OrderCount` bloquea `T → M → Y` → se estima solo el **efecto directo** (CDE) y se subestima el efecto total que le importa al negocio. Mostrarlo numéricamente (ATE con y sin M).
- ⚠️ **Trampa del colisionador:** `T → DaySinceLastOrder ← Y`. Condicionar en C **abre** el camino y crea asociación espuria T–Y (sesgo de Berkson). Mostrarlo numéricamente (ATE con C incluida) + mini-simulación sintética de 10 líneas donde el efecto verdadero se conoce.
- ⚠️ Condicionar en un **descendiente de Y** es además casi-leakage temporal.

### Paso 4.3 · DoWhy en 4 pasos
1. **Model:** `CausalModel(data, treatment="coupon_any", outcome="Churn", graph=dot)`.
2. **Identify:** `identify_effect()` → leer el estimando backdoor y verificar que el set coincide con el derivado a mano.
3. **Estimate:** `backdoor.linear_regression` (modelo de probabilidad lineal → ATE en puntos porcentuales) y `backdoor.propensity_score_weighting` (IPW; revisar **overlap/positividad**: histograma de propensity por grupo, trimming 0,01-0,99, balance de covariables con SMD < 0,1 antes/después).
4. **Comparar** con la regresión ingenua `Churn ~ coupon_any`: sesgo = ATE_naive − ATE_ajustado; explicar dirección (¿los recurrentes usan más cupones y churnean menos → el ingenuo sobreestima? ¿los cupones compensatorios por reclamo → el ingenuo subestima?).

### Paso 4.4 · Causal ML con EconML
- `LinearDML(model_y=RandomForestRegressor, model_t=RandomForestClassifier, discrete_treatment=True, cv=5)` para CATE lineal en X interpretables (Tenure, CityTier, Complain, SatisfactionScore); `CausalForestDML` para heterogeneidad no lineal.
- Explicar la **ortogonalización (Neyman / FWL)**: se residualiza Y y T sobre Z con cross-fitting y se regresa residuo sobre residuo → robusto a errores de primer orden en los nuisance.
- Gráficos: CATE vs Tenure con banda IC; CATE por segmento (barras con IC); histograma de CATE individuales; **política**: "dar cupón solo si CATE < umbral de costo" y ganancia esperada vs "dar a todos".

### Paso 4.5 · Refutaciones (semáforo)
| Refutador | Expectativa | Semáforo |
|---|---|---|
| Random Common Cause | ATE ≈ original | 🟢 si Δ < 10% |
| Placebo Treatment (permute) | ATE ≈ 0, p > 0,05 | 🟢 si p > 0,05 |
| Data Subset (80%, ≥ 50 sims) | ATE dentro del IC 95% original | 🟢 si dentro |
| (Extra) Unobserved common cause | curva de sensibilidad | informativo |

### Paso 4.6 · Ensayo crítico "Capacidad predictiva vs palanca causal"
Estructura sugerida (1-1,5 páginas):
1. La variable top del RF (probablemente `Tenure` o `DaySinceLastOrder`) **no es intervenible** o es **consecuencia** de Y.
2. Condiciones bajo las que un gran predictor no tiene poder causal: confusión, causalidad inversa, colisionadores, proxies.
3. Inversión de signo (Simpson): cupón asociado a *más* churn en bruto por cupones compensatorios, pero efecto causal protector tras ajustar `Complain`.
4. Implicancia de negocio: targeting por **probabilidad de churn** (predictivo) ≠ targeting por **uplift/CATE** (causal). "Sure things", "lost causes", "persuadables", "sleeping dogs".

---

## 5. Catálogo de trampas (checklist del QA)

| # | Trampa | Cómo se detecta | Test / evidencia |
|---|---|---|---|
| 1 | Shapiro-Wilk con N grande | se decide por p de Shapiro | revisión de notebook |
| 2 | Imputar/escalar/PCA antes del split | media del scaler ≠ media Train | `test_scaler_fitted_only_on_train`, `test_pca_fitted_only_on_train` |
| 3 | Hiperparámetros tuneados mirando Test | Test usado más de una vez | revisión de notebook |
| 4 | Tukey con heterocedasticidad | Levene rechaza y se usa Tukey | revisión |
| 5 | Durbin-Watson en datos transversales sin aclarar | — | revisión |
| 6 | Eliminar outliers sin Cook/leverage | — | revisión |
| 7 | Mediador o colisionador en el set de ajuste | intersección no vacía | `test_adjustment_set_excludes_mediators_and_colliders` |
| 8 | Leer importancia MDI como causal | — | ensayo crítico |
| 9 | Interpretar coeficientes de la logística como efectos causales | — | revisión |
| 10 | IPW sin chequear positividad | propensities en 0/1 | histograma de overlap |

---

## 6. Mapeo con la Rúbrica complementaria (nivel Excelente = 4 pts)

| Sub-criterio | Cómo lo cubrimos |
|---|---|
| 1.1 Diagnóstico del problema | Entrada 1: causa raíz de negocio (costo del cupón vs valor de retener), preguntas de valor |
| 1.2 Estructura relacional | Tabla de roles T/Y/Z/M/C + DAG + asociaciones cruzadas no pedidas (Spearman vs Pearson) |
| 1.3 Auditoría forense (DGP) | Paso 1.1-1.2: anomalías, categorías duplicadas, mecanismo de NA, impacto antes/después |
| 1.4 Diseño muestral | Split, semilla, estratificación, cross-fitting, **análisis de potencia** (1−β) |
| 2.1 Abstracción vs código | Celda teórica (ecuaciones, H₀/H₁) antes de cada bloque |
| 2.2 Parámetros y supuestos | Potenciales resultados $Y(1), Y(0)$, SUTVA, positividad, ignorabilidad condicional |
| 2.3 Bordes y modelos de resguardo | Welch/Kruskal, HC3, Huber/RANSAC, calibración isotónica, trimming de propensity |
| 3.1 Escenarios alternativos | ATE con/sin M, con/sin C, T binaria vs continua, imputación alternativa (sensibilidad MNAR) |
| 3.2 Convergencia numérica | Curva OOB vs n_estimators, curvas de aprendizaje, estabilidad de semillas, IC bootstrap |
| 3.3 Rigor en contrastes | Shapiro, Levene, BP, White, JB, DW, VIF interpretados con su limitación |
| 3.4 Visualización avanzada | Paneles multi-eje, Q-Q por grupo, influence plot, reliability, CATE con bandas |
| 4.1 Vehículo editorial | Notebook con índice, numeración de figuras/tablas, export HTML/PDF |
| 4.2 Precisión terminológica | Glosario al final del notebook (ATE, CATE, CDE, backdoor, d-separación, OOB…) |
| 4.3 Traducción a negocio | Recomendación de política con \$ esperados y segmentos a targetear |
| 4.4 Reproducibilidad y gobernanza IA | `requirements.lock`, semilla, Restart & Run All, Bitácora Entrada 5 |

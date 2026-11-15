# Esqueleto de Slides para la Defensa Oral (12 slides · 12–15 min)

> **Estado:** Esqueleto v1.0 con títulos, mensajes clave y cifras verificadas (fases 1, 3 y 4).  
> **Pendiente:** insertar las figuras finales cuando estén integrados los notebooks de Fases 2, 3 y 4, más capturas reales de la Web App en Streamlit.  
> **Versión PDF preliminar:** [`docs/04_esqueleto_slides.pdf`](04_esqueleto_slides.pdf).  
> **Slides finales (2026-11-15):** [`docs/slides/slides_defensa.pdf`](slides/slides_defensa.pdf), con figuras y capturas. Las cifras de este guion que cambiaron están anotadas en la Sesión IA-15 de la bitácora.

---

## Estructura general (alineada con el enunciado y `docs/03_plan_accion_scrum_72hs.md`)

| Slide | Bloque | Responsable principal | Duración sugerida |
|---:|---|---|---:|
| **1–2** | Problema, preguntas $\mathbb{P}(Y \mid X)$ vs. $\mathbb{P}(Y \mid \text{do}(T))$, dataset y limpieza | Carola Nasif (PL) & Julián Lozano (ML) | ~2,5 min |
| **3–4** | Inferencia clásica, ANOVA, Gauss-Markov y PCA sin leakage | Carola Nasif (PL) & Emilia Mortarini (QA/F2) | ~2,5 min |
| **5–6** | Torneo predictivo de ML, calibración e importancia MDI vs. permutación | Julián Agustín Lozano (ML) | ~2,5 min |
| **7–8** | DAG estructural, trampas $M/C$, positividad, ATE/CATE y refutaciones | Julián Agustín Lozano (ML) & Carola Nasif (PL) | ~3,0 min |
| **9** | Demo en vivo de la Web App en Streamlit (3 módulos) | Emilia Mortarini (FE) | ~2,0 min |
| **10** | Gobernanza Scrum, Quality Gate y Bitácora (autopsia y trazabilidad IA) | Emilia Mortarini (QA) & Carola Nasif (PL) | ~1,0 min |
| **11** | Conclusiones ejecutivas, umbral $-\text{ATE}\cdot V > c$ y política comercial | Carola Nasif (PL) | ~1,5 min |
| **12** | *(Backup / Apéndice técnico)* Sensibilidad, TOT vs. ITT y reproducibilidad | Equipo completo | Preguntas |

---

### Slide 1 — Problema de negocio y pregunta causal: $\mathbb{P}(Y \mid X)$ vs. $\mathbb{P}(Y \mid \text{do}(T))$
- **Responsable:** Carola Nasif (Project Lead)
- **Mensaje principal:** Predecir quién abandonará no responde si regalar un cupón retiene clientes: el proyecto separa predicción (*What-Is*) de intervención causal (*What-If*).
- **Puntos clave:**
  - Paradoja comercial: clientes que usan cupones tienen alta autoselección; el churn bruto es similar (`18,05%` tratados vs. `18,40%` controles), impidiendo inferir retorno directo sin ajustar por confusores.
  - **Pregunta predictiva $\mathbb{P}(Y \mid X)$:** ¿qué clientes tienen alta probabilidad de abandonar el próximo período y qué tan bien calibradas están sus probabilidades?
  - **Pregunta causal $\mathbb{P}(Y \mid \text{do}(T))$:** ¿cuál es el efecto causal promedio ($\text{ATE}$) y heterogéneo ($\text{CATE}$) de usar cupones (`coupon_any`) sobre `Churn` tras bloquear confusores pre-tratamiento $Z$?
  - Objetivo de decisión: cuantificar un umbral de rentabilidad ($-\text{ATE} \cdot V > c$) para asignar o reasignar presupuesto promocional.
- **Figura / Tabla a insertar:** `[FIGURA: Esquema conceptual P(Y | X) vs. P(Y | do(T)) y mapa causal de BITACORA.md §1.4]`

---

### Slide 2 — Dataset (Opción A), contrato $T/Y/Z/M/C$ y limpieza forense
- **Responsables:** Carola Nasif (PL) & Julián Agustín Lozano (ML)
- **Mensaje principal:** La auditoría forense descubrió **2.638 copias casi duplicadas (`ID + 1470`)**: eliminarlas antes del split evitó *fold-leakage* y redujo la muestra de `5.630` a `2.982` clientes independientes.
- **Puntos clave:**
  - Fuente y licencia: Kaggle *Ecommerce Customer Churn Analysis and Prediction* (Ankit Verma) · Licencia **CC BY-NC-SA 4.0** · SHA-256 `db70f1e3...`.
  - Contrato DAG: $Y =$ `Churn` (`17,30%`), $T =$ `coupon_any` (`80,73%` tratados; `133` NA preservados sin imputar), $Z = 13$ confusores, $M = \{\text{OrderCount}, \text{CashbackAmount}\}$, $C = \{\text{DaySinceLastOrder}\}$.
  - Autopsia del DGP: `2.638` pares (`ID` vs `ID + 1470`) idénticos en `14–19/19` columnas y con `100%` igual `Churn`; `CouponUsed` variaba en `1.469` pares, atenuando el efecto de $T$.
  - Split estratificado único previo a toda imputación/escalado: `Train = 2.385` (`80%`) y `Test = 597` (`20%`), `RANDOM_STATE = 42`.
- **Figura / Tabla a insertar:** `[TABLA 0.1 cleaning_summary (5.630 → 2.982) + esquema del split Train/Test del notebook maestro]`

---

### Slide 3 — Inferencia clásica (Fase 1): faltantes MAR, contrastes, potencia y ANOVA
- **Responsable:** Carola Nasif (PL)
- **Mensaje principal:** Los faltantes no son MCAR, los reclamos triplican la tasa de churn y `PreferedOrderCat` explica el `82,7%` de la varianza de `CashbackAmount`, confirmando confusión estructural.
- **Puntos clave:**
  - Faltantes (`7` columnas, `4,39%–5,43%`): Test de Little EM rechaza MCAR ($d^2 = 2.875{,}89$, $p < 10^{-300}$); regresión logística MAR alcanza `AUC = 0,898–0,996` (`coupon_any` NA `AUC = 0,965`).
  - Contraste 1 (`CashbackAmount` por `Churn`): diferencia `-$19,36` (IC 95% `[-23,27; -15,44]`), Welch $t = -9{,}71$ ($p = 2{,}85 \times 10^{-21}$), Mann-Whitney $p = 4{,}34 \times 10^{-20}$, Cohen's $d = -0{,}3955$, potencia `1,0000` (`MDE 80% = 0,1357`).
  - Contraste 2 (`Complain` vs `Churn`): `32,63%` con reclamo vs `11,11%` sin reclamo (`+21,52 pp`), $\chi^2 = 197{,}84$ ($p = 6{,}19 \times 10^{-45}$), $V = 0{,}2576$, $\text{OR} = 3{,}88$.
  - ANOVA `CashbackAmount ~ PreferedOrderCat`: Levene $p = 1{,}14 \times 10^{-15}$ (heterocedástico) $\to$ Welch $F = 5.258{,}31$, Kruskal $H = 2.138{,}35$ ($p < 10^{-300}$), $\eta^2 = 0{,}8269$; `10/10` pares rechazan $H_0$ en Games-Howell.
- **Figura / Tabla a insertar:** `[FIGURAS 1.3 (missingno + AUC MAR), 1.4 (contrastes + potencia) y 1.5/1.6 (Q-Q y Games-Howell) del notebook maestro]`

---

### Slide 4 — Diagnóstico econométrico Gauss-Markov, influencia, VIF y PCA sin leakage (Fase 2)
- **Responsable:** Emilia Mortarini (QA / Fase 2)
- **Mensaje principal:** La auditoría MCO/HC3, el análisis de influencia, los VIF y el PCA ajustado exclusivamente en Train validan los supuestos lineales antes del modelado predictivo.
- **Puntos clave:**
  - Supuestos Gauss-Markov: contrastes de heterocedasticidad (Breusch-Pagan / White $\to$ errores estándar robustos `HC3`), normalidad de residuos (Jarque-Bera + Q-Q) e independencia (Durbin-Watson).
  - Observaciones influyentes y regresión robusta: distancia de Cook ($> 4/n$), *leverage* $h_{ii}$ y comparación de coeficientes MCO vs. `HuberRegressor`.
  - Multicolinealidad (VIF): impacto de la fuerte dependencia entre `OrderCount` y `CouponUsed` ($r = 0{,}741$, $\rho_s = 0{,}724$).
  - PCA *train-only* (`fit_pca_train_only` sobre `Train N = 2.385`): varianza acumulada, criterio de selección de componentes y biplot PC1–PC2 sin contaminación de Test.
- **Figura / Tabla a insertar:** `[FIGURAS A AGREGAR: Residuos MCO/HC3, Cook, tabla VIF y biplot PCA desde la sección Fase 2 integrada]`

---

### Slide 5 — Torneo predictivo (Fase 3): 4 familias con `StratifiedKFold(5)` y evaluación única en Test
- **Responsable:** Julián Agustín Lozano (ML)
- **Mensaje principal:** Sin *leakage* de copias duplicadas, **Random Forest** alcanza **`ROC-AUC = 0,9461`** en Test (`N = 597`) y supera con claridad a Logística ElasticNet, CART podado y Naive Bayes.
- **Puntos clave:**
  - Selección en CV sobre Train (`N = 2.385`): `Random Forest` **`0,9320 ± 0,0147`** (`OOB AUC = 0,9411`) $>$ `Logística ElasticNet` (`0,8935`) $>$ `CART podado` (`0,8477`) $>$ `Naive Bayes` (`0,7857`).
  - Autopsia de CART sin podar: `Train ROC-AUC = 1,0000` vs `Test = 0,8066` (sobreajuste severo); poda por costo-complejidad (`ccp_alpha = 0,0019`) eleva Test a `0,8355`.
  - Evaluación única en hold-out Test (`N = 597`): `Random Forest + isotónica` logra **`ROC-AUC = 0,9461`** (IC 95% `[0,9218; 0,9676]`), **`PR-AUC = 0,8062`**, `Accuracy = 90,62%`, `F1 = 0,6989`.
  - Impacto de la depuración forense: la limpieza de pares `ID + 1470` corrigió un ROC-AUC inflado de `0,993` (con copias entre folds) a `0,948/0,932`.
- **Figura / Tabla a insertar:** `[TABLA 6.1 + FIGURAS A AGREGAR: Curvas ROC/PR en Test y trayectoria ccp_alpha desde borrador Fase 3]`

---

### Slide 6 — Calibración isotónica e importancia MDI vs. permutación en Test
- **Responsable:** Julián Agustín Lozano (ML)
- **Mensaje principal:** La calibración isotónica reduce el error de calibración (`ECE`) en **`77%`** (`0,079 → 0,018`), mientras la permutación en Test desnuda el sesgo por cardinalidad del MDI.
- **Puntos clave:**
  - Calibración en Test: `RF + isotónica` reduce `Brier` de `0,0716 → 0,0643` y `ECE` de `0,0790 → 0,0179` (`Logística ElasticNet` logra `ECE = 0,0225`).
  - Autopsia de `Naive Bayes`: `Brier = 0,3223` y `ECE = 0,3650` al violar independencia condicional entre `OrderCount` y `CouponUsed`.
  - Top-4 predictores por permutación en Test ($\Delta\text{ROC-AUC}$): **`Tenure`** (`0,2024`), **`Complain`** (`0,0365`), **`PreferedOrderCat`** (`0,0251`) y **`DaySinceLastOrder`** (`0,0170`).
  - Experimento MDI vs. Permutación (Strobl et al., 2007): `CashbackAmount` aparece `#2` en MDI (`0,0857`) pero cae al **`#18` (último, `-0,0010`)** por permutación en Test; `CouponUsed` queda `#16` (`+0,0003`).
- **Figura / Tabla a insertar:** `[FIGURAS A AGREGAR: Reliability diagrams (RF base vs isotónica vs NB) y ranking MDI vs Permutación con ruido_continuo desde Fase 3]`

---

### Slide 7 — Estructura causal (Fase 4): DAG, criterio *backdoor*, positividad y trampas $M/C$
- **Responsables:** Julián Agustín Lozano (ML) & Carola Nasif (PL)
- **Mensaje principal:** Solo el set pre-tratamiento $Z$ cumple $d$-separación sin latentes; además, **`808` clientes** quedan fuera de soporte común (`100%` tratados) y ajustar $M$ o $C$ sesga el efecto.
- **Puntos clave:**
  - DAG estructural: `Z` (13 covariables) bloquea los caminos backdoor observables entre `coupon_any` y `Churn`; `U_engagement` permanece como latente no observado.
  - Violación estructural de positividad: `808` clientes en estratos donde el `100%` usó cupón (`HourSpendOnApp = 4`: `605`; `WarehouseToHome = 7`: `202`; `NumberOfDeviceRegistered = 6`: `87`; `NumberOfAddress = 11`: `49`; `Tenure = 31`: `23`), más `29` fuera del rango de propensity `[0,05; 0,95]`.
  - Población con soporte común: **`N = 2.012`** (`1.463` tratados / `549` controles); ponderación IPW reduce $|\text{SMD}|_{\max}$ de `0,483` (`11` variables $>0{,}10$) a `0,107` (`1` $>0{,}10$).
  - Trampas causales verificadas: controlar por mediadores $Z + M$ (`−2,39 pp`) o por colisionador $Z + C$ (`−0,74 pp`); simulación de Berkson colapsa un efecto real `+0,10` a **`−0,46`**.
- **Figura / Tabla a insertar:** `[FIGURAS A AGREGAR: DAG renderizado (artifacts/dag.dot), histograma overlap propensity score y Love plot SMD desde Fase 4]`

---

### Slide 8 — Estimaciones ATE/CATE, dosis-respuesta y semáforo de refutaciones DoWhy
- **Responsable:** Julián Agustín Lozano (ML)
- **Mensaje principal:** Todos los estimadores ajustados incluyen el `0` en su IC 95% (**no hay evidencia de que usar cupón reduzca el churn**) y las `3` refutaciones de DoWhy pasan en verde.
- **Puntos clave:**
  - Estimadores ATE en soporte común (`N = 2.012`, en puntos porcentuales de churn): Backdoor regresión = **`−1,16 pp`** (`[-4,60; +2,28]`), IPW = **`−1,09 pp`** (`[-5,56; +3,14]`), LinearDML = **`+1,11 pp`** (`[-2,03; +4,24]`), CausalForestDML = **`+1,42 pp`** (`[-1,46; +4,29]`).
  - Dosis-respuesta continua (`CouponUsed`): **`+1,07 pp` de churn por cupón adicional** (IC 95% `[+0,14; +2,00]`, $p = 0{,}024$).
  - Heterogeneidad (CATE): ningún segmento (`Tenure`, `Complain`, `SatisfactionScore`) es significativo; en `Complain = 1`, LinearDML estima **`+7,1 pp`** ($p = 0{,}065$).
  - Refutaciones DoWhy **3/3 verdes** (🟢): *Random Common Cause* (`−1,16 pp`, $p = 0{,}90$), *Placebo Treatment* (`+0,09 pp`, $p = 0{,}96$), *Data Subset 80%* (`−0,99 pp`, $p = 0{,}88$). Sensibilidad latente: `[−1,75; +6,17] pp`.
- **Figura / Tabla a insertar:** `[TABLA 6.2 + FIGURAS A AGREGAR: Forest plot de estimadores ATE, curva CATE por Tenure y semáforo de refutaciones desde Fase 4]`

---

### Slide 9 — Demo en vivo: Web App interactiva en Streamlit (3 módulos)
- **Responsable:** Emilia Mortarini (Frontend & UI/UX)
- **Mensaje principal:** La aplicación productiva desacopla diagnóstico exploratorio, *scoring* predictivo individual y simulación causal interactiva sobre artefactos auditados.
- **Puntos clave:**
  - **Módulo 1 — Diagnóstico / PCA:** métricas del dataset depurado (`N = 2.982`), auditoría de faltantes y proyección PCA ajustada solo en Train.
  - **Módulo 2 — What-Is (Predictivo):** probabilidad individual de churn calibrada (`Random Forest + isotónica` desde `artifacts/best_model.joblib`) e importancia global por permutación.
  - **Módulo 3 — What-If (Causal):** grafo `dag.dot`, tabla/comparador de ATE con IC, semáforo de refutaciones (`causal_results.json`) y selector de perfiles CATE (`cate_grid.parquet`).
  - Reproducibilidad en Streamlit Community Cloud garantizada mediante `app/requirements.txt` (`scikit-learn==1.5.2`) y pines `pandas<3` / `networkx<3.5` en la raíz (`URL deploy: https://tpi-eamr-grupo-atomos.streamlit.app/`).
- **Figura / Tabla a insertar:** `[CAPTURAS A AGREGAR: Screenshots de los 3 módulos de app/app.py desplegados + video backup MP4 para la defensa]`

---

### Slide 10 — Gobernanza Scrum, Quality Gate y Bitácora de ingeniería
- **Responsables:** Emilia Mortarini (QA) & Carola Nasif (PL)
- **Mensaje principal:** El flujo de PRs apilados bajo *Quality Gate* determinístico (`ruff + pytest + bitácora`) permitió interceptar errores de dependencias, *leakage* por copias y sesgos de medición.
- **Puntos clave:**
  - Organización Scrum (3 integrantes / 4 roles): orden estricto de merge `#1 → #2 → #11 → #12 → #13 → #14 → #15` usando únicamente *merge commits* tras `[APROBADO]` de QA.
  - Quality Gate automático (`tests/` + `ruff`): verificación de no-leakage (split 80/20 primero, preprocesador y PCA ajustados en Train), exclusión de $M/C$ del set backdoor y tests sintéticos con efecto conocido (`-0,10` en causal, `2` en smoke test).
  - Data Autopsy (Entrada 3): Exp. `#1` (`2.638` copias casi duplicadas `ID + 1470`) y Exp. `#2` (descalibración extrema de Naive Bayes / sobreajuste CART sin podar / trampas de colisionador).
  - Trazabilidad IA (Entrada 5): verificación V1–V8 contra datos/docs reales (detectando incompatibilidad `networkx 3.6` / `pandas 3.1` con `dowhy 0.12` y API de `pingouin 0.6.1`).
- **Figura / Tabla a insertar:** `[DIAGRAMA/CAPTURA A AGREGAR: Diagrama Mermaid src → artifacts → app + evidencia pytest/ruff en verde]`

---

### Slide 11 — Conclusiones ejecutivas, umbral $-\text{ATE}\cdot V > c$ y recomendación de política
- **Responsable:** Carola Nasif (Project Lead)
- **Mensaje principal:** No hay sustento para escalar cupones masivos: utilizar Random Forest para detectar riesgo, reasignar subsidios a experiencia/reclamos y validar incentivos vía A/B test.
- **Puntos clave:**
  - Regla de rentabilidad cuantificada: con costo del cupón $c$ y valor de retención $V$, conviene dar cupón solo si **`−ATE · V > c`**. Con el límite más favorable del IC 95% (`−4,60 pp`), se requiere **`c < 0,046 · V`** (y puntual, `c < 0,0116 · V`).
  - Separar detección de intervención: usar **`RF + isotónica`** (`ROC-AUC = 0,946`, `ECE = 0,018`) para **DETECTAR riesgo**, jamás como gatillo automático de cupones.
  - Reasignación operativa: reducir subsidios inframarginales (`80,73%` tratados), reforzar *onboarding* temprano (`Tenure` `#1` en permutación) y resolver causa raíz de **`Complain`** (`#2` en permutación; `+7,1 pp` en DML).
  - Próximo paso: **experimento A/B aleatorio (`RCT`)** asignando la oferta del cupón (`ITT` / `TOT`) para eliminar positividad violada (`808` excluidos) y confusión oculta.
- **Figura / Tabla a insertar:** `[DIAGRAMA/TABLA A AGREGAR: Matriz de decisión P(Y|X) vs P(Y|do(T)) y recta de equilibrio c / V <= -ATE]`

---

### Slide 12 — *(Backup / Apéndice técnico)* Limitaciones, TOT vs. ITT y sensibilidad latente
- **Responsables:** Equipo completo (PL / ML / QA-FE)
- **Mensaje principal:** Material de respaldo para preguntas cruzadas del tribunal sobre validez interna, diseño longitudinal y reproducibilidad.
- **Puntos clave:**
  - Redención vs. exposición: `CouponUsed >= 1` mide uso efectivo (**TOT**) y no oferta asignada (**ITT**), conservando `133` NA bajo mecanismo MAR (`AUC = 0,965`).
  - Diseño transversal ("último mes"): `SatisfactionScore` y `HourSpendOnApp` comparten ventana temporal con $T$ e $Y$ (sensibilidad excluyéndolas en Fase 4).
  - Análisis de sensibilidad a confusores ocultos (`U_engagement`): `add_unobserved_common_cause` produce un rango `[−1,75; +6,17] pp` en el ATE según fuerza del confusor.
  - Reproducción íntegra desde línea de comandos: `src.preprocessing` $\to$ `src.models` $\to$ `src.causal` $\to$ `ruff` + `pytest` $\to$ `nbconvert --execute` $\to$ `streamlit run`.
- **Figura / Tabla a insertar:** `[TABLA A AGREGAR: Resumen cuantitativo de sensibilidad DAG y reproducibilidad de entorno]`

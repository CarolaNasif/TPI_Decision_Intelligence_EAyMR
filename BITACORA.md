# 📓 BITÁCORA DE INGENIERÍA — Decision Intelligence Capstone

> Cuaderno de laboratorio y registro de decisiones (no es un resumen del notebook ni un log de commits).
> **Proyecto:** Retorno causal de un motor de recomendaciones / incentivo personalizado sobre la retención · EAyMR 2026 · UCA
> **Formato de cada registro:** `[AAAA-MM-DD HH:MM] · Autor (Rol) · Decisión · Evidencia · Alternativas descartadas`

| Rol | Integrante | Firma |
|---|---|---|
| 👑 Project Lead & Scrum Master (PO) | **Carola Nasif** | ☐ |
| ⚙️ Backend & ML / Causal Engineer | **Julián Agustín Lozano** | ☐ |
| 🎨 Frontend & UI/UX Engineer | **Emilia Mortarini** | ☐ |
| 🔍 QA & Validation Engineer | **Emilia Mortarini** (co-auditoría cruzada con Carola Nasif) | ☐ |

---

## Entrada 1 — Marco Estratégico e Hipótesis
**Responsable:** Project Lead · **Estado:** 🟢 CERRADA v1.0 (2026-11-14 15:00)

### 1.1 Problema de negocio
Las plataformas de e-commerce asignan cupones, cashback y recomendaciones personalizadas (generadas por un motor de recomendación / CRM) con el objetivo de retener clientes. Los dashboards muestran que **los clientes que usan incentivos churnean menos**, y la dirección concluye que "los cupones retienen". Pero esa correlación está contaminada por **autoselección**: los usuarios recurrentes, antiguos y comprometidos son, a la vez, los que más usan incentivos y los que menos abandonan. Si la empresa escala la política sobre esa correlación, puede estar **regalando margen a clientes que se iban a quedar igual** ("sure things") y no alcanzando a los "persuadibles".

### 1.2 Preguntas fundacionales
1. **Predictiva — $P(Y \mid X)$:** ¿Qué clientes tienen alta probabilidad de churn el próximo período, y qué tan bien calibradas están esas probabilidades?
2. **Causal — $P(Y \mid do(T=t))$:** ¿Cuál es el efecto causal promedio (ATE) del incentivo $T$ sobre la retención $Y$ una vez bloqueados los caminos de puerta trasera por $Z$?
3. **Heterogeneidad — CATE $\tau(x)$:** ¿En qué segmentos (antigüedad, ciudad, categoría, reclamo previo) el incentivo es costo-efectivo y en cuáles es neutro o contraproducente?
4. **Choque:** ¿La variable más predictiva del churn es una palanca accionable?

### 1.3 Hipótesis de trabajo (a contrastar)
- **H-causal-1:** el ATE ajustado del incentivo sobre el churn es negativo (protege) pero de **menor magnitud** que la asociación ingenua (sesgo de autoselección por `Tenure`).
- **H-causal-2:** el sesgo puede invertir su signo por **cupones compensatorios** otorgados tras un reclamo (`Complain` como confusor).
- **H-predictiva-1:** las variables top del Random Forest (`Tenure`, `DaySinceLastOrder`, `Complain`) no coinciden con las palancas causales.
- **H-inferencial-1:** el cashback medio difiere entre clientes que churnean y que no (Welch t, Cohen's d).
- **H-inferencial-2:** el cashback medio difiere entre categorías de producto preferidas (ANOVA / Welch / Kruskal).

### 1.4 Mapa conceptual

```mermaid
flowchart LR
    Z["Z · Autoselección: antigüedad, reclamos, satisfacción, segmento"] --> T["T · Incentivo / recomendación personalizada"]
    Z --> Y["Y · Retención / Churn"]
    T --> M["M · Pedidos y cashback"]
    M --> Y
    T --> Y
    T --> C["C · Días desde la última compra"]
    Y --> C
```

### 1.5 Selección fundamentada del dataset
- Candidatos evaluados: **A** E-Commerce Customer Churn (Kaggle, N=5.630, 18 explicativas) · **B** Olist (≈96k clientes, 9 tablas) · **C** Lenta Uplift (687k, RCT). Detalle y matriz ponderada en [`docs/01_propuestas_datasets_y_dags.md`](docs/01_propuestas_datasets_y_dags.md).
- **Propuesta del PL:** Opción A (puntaje 4,50/5) por tener T, Y, Z, M y C **observables** en una sola tabla, 7 columnas con faltantes reales y viabilidad en 72 hs.
- **Decisión del equipo:** ☑ A ☐ B ☐ C — fecha/hora: 2026-11-14 15:00 — votos: 3/3 (unánime: Carola Nasif, Julián Agustín Lozano, Emilia Mortarini)
- Procedencia, licencia y hash SHA-256 del archivo crudo: Kaggle (`ankitverma2010/ecommerce-customer-churn-analysis-and-prediction`, archivo `E Commerce Dataset.xlsx`, hoja `E Comm`) · licencia pública de dataset Kaggle (uso académico/no comercial) · SHA-256: `db70f1e34bc5326887ea3fb3d2a0854e82c577d55ef3a76fdfb25328631bd44d`
- Amenaza reconocida desde el día 0: procedencia corporativa anónima y corte transversal (orden temporal de T, M, C e Y asumido, no observado).

### 1.6 Criterios de éxito
Notebook "Restart & Run All" sin errores · ATE con IC 95% y 3 refutaciones en verde · App pública con 3 módulos · Quality Gate en verde en `main` · Rúbrica autoevaluada ≥ 90.

---

## Entrada 2 — Matriz de Decisiones Estadísticas y Supuestos
**Responsable:** QA + Project Lead · **Estado:** 🟡 Fase 1 completa (Carola Nasif · pendiente Fase 2 por QA y Fase 4 por ML)

| # | Fase | Supuesto / decisión | Test / evidencia | Resultado (estadístico, p) | Violación | Remediación elegida | Alternativa descartada y por qué |
|---|---|---|---|---|---|---|---|
| 1 | F1 | α a priori | — | α = 0,05 | — | Fijado antes del análisis; se reporta p exacto y magnitud | α = 0,01 (reduciría potencia innecesariamente) |
| 2 | F1 | Mecanismo de NA en las 7 columnas con faltantes (4,39%–5,43% en N=2.982) | Test de Little (EM) + logit indicador NA + χ² vs Churn | Little d²=2875,89 (gl=77, p<1e-300); logit MAR AUC=0,898–0,996 (p_LRT≤2,22e-64); χ² vs Churn p≤0,00067 en 5/7 (HourSpendOnApp p=0,075, DaySinceLastOrder p=0,836) | Sí (no es MCAR) | Imputación por mediana/moda + indicador binario de NA dentro del Pipeline (fit solo en Train); T=`coupon_any` preserva 133 NA | Eliminación por lista (sesgo bajo MAR) o imputación previa al split (leakage) |
| 3 | F1 | Normalidad por grupo (ANOVA `CashbackAmount ~ PreferedOrderCat`) | asimetría, curtosis, Q-Q (Shapiro solo informativo) | Fashion (skew 0,10; kurt -0,29; Shapiro p=3,74e-4) y Others (-0,03; -1,49; p=8,12e-8) simétricos; Mobile (-0,92; 7,20), Laptop (-1,53; 11,55) y Grocery (-4,26; 30,99) con colas/curtosis | Sí (Grocery, Laptop y Mobile exceden \|kurt\|≤2; Shapiro rechaza en los 5 por N grande) | Decisión por asimetría/curtosis + Q-Q; Kruskal-Wallis (H=2138,35, p<1e-300, ε²=0,717) como no paramétrico y Welch ANOVA como robusto de medias | Decidir por Shapiro-Wilk (hipersensible con N=2.982) o usar solo ANOVA clásico |
| 4 | F1 | Homocedasticidad (`CashbackAmount ~ PreferedOrderCat`) | Levene (center=median) + ratio var_max/var_min | Levene=19,28 (p=1,14e-15); ratio var_max/var_min=5,98 | Sí (heterocedasticidad) | Welch ANOVA (F=5258,31, gl1=4, gl2=613,21, p<1e-300) y Kruskal-Wallis (H=2138,35, p<1e-300, η²=0,827, ω²=0,827); se rechaza H0 | ANOVA clásico de Fisher sin ajuste (asume varianzas iguales; queda solo como referencia) |
| 5 | F1 | Control FWER post-hoc (k=5 grupos, m=10 pares, FWER sin ajuste=40,13%) | Tukey HSD / Games-Howell / Bonferroni | 10/10 pares rechazan H0 en Games-Howell (p_adj≤4,53e-13), Tukey HSD (p_adj=2,43e-12) y Bonferroni (p_adj≤1,73e-38) | No (controlado) | Games-Howell (heterocedástico) + Bonferroni (no paramétrico) y Tukey HSD como contraste | 10 comparaciones sin corrección (inflaría FWER a 40,13%) |
| 6 | F2 | Homocedasticidad MCO | Breusch-Pagan / White | | | HC3 | |
| 7 | F2 | Normalidad residuos | Jarque-Bera + Q-Q | | | | |
| 8 | F2 | Independencia | Durbin-Watson (aclarar orden de filas) | | | | |
| 9 | F2 | Influyentes | Cook > 4/n, h > 2(p+1)/n, \|r\| > 3 | | | Huber / RANSAC | |
| 10 | F2 | Multicolinealidad | VIF > 5 / > 10 | | | | |
| 11 | F2 | PCA: nº de componentes | Kaiser vs codo vs 80% | | | | |
| 12 | F4 | Positividad / overlap | histograma de propensity, trimming | | | | |
| 13 | F4 | Balance post-IPW | SMD < 0,1 | | | | |

### 2.1 Resumen de contrastes formales y potencia (Fase 1, `make_clean_dataset()`, N = 2.982; consistentes en Train N = 2.385)
- **Contraste 1 (`CashbackAmount` según `Churn`):** `Churn=1` (n=516, media \$160,77, DE 38,89, mediana \$149,76, IQR \$41,12) vs `Churn=0` (n=2.466, media \$180,12, DE 50,79, mediana \$165,83, IQR \$53,33). Diferencia = −\$19,36 (IC 95% analítico [−23,27; −15,44], bootstrap 5.000 réplicas [−23,32; −15,44]). Welch $t = -9,71$ ($\text{gl}=923,66$, $p = 2,85 \times 10^{-21}$), Mann-Whitney $U = 472.974,0$ ($p = 4,34 \times 10^{-20}$), Cohen's $d = -0,3955$ (efecto pequeño). Se rechaza $H_0$. **Potencia (`TTestIndPower`):** $1-\beta = 1,0000$; tamaño de efecto mínimo detectable al 80% de potencia: $d_{\min} = 0,1357$. *(En Train N=2.385: diff = −\$19,86, $t = -9,20$, $\text{gl}=776,39$, $p = 3,24 \times 10^{-19}$, Mann-Whitney $p = 4,82 \times 10^{-16}$, $d = -0,4062$, potencia = 1,0000, $d_{\min} = 0,1517$).*
- **Contraste 2 (`Complain` vs `Churn`):** tasa de churn con reclamo = 32,63% (280/858) vs sin reclamo = 11,11% (236/2.124); diferencia = +21,52 pp (IC 95% [+18,11 pp; +24,93 pp]), $\chi^2 = 197,84$ ($\text{gl}=1$, $p = 6,19 \times 10^{-45}$), $V$ de Cramér $= 0,2576$, OR $= 3,88$. Se rechaza $H_0$. *(En Train N=2.385: diff = +23,43 pp, $\chi^2 = 187,15$, $p = 1,33 \times 10^{-42}$, $V = 0,2801$, OR $= 4,35$).*

---

## Entrada 3 — Autopsia de Experimentos Fallidos (Data Autopsy)
**Responsable:** Backend & ML + QA · **Estado:** 🟡 en curso (1/2) · **Mínimo: 2 experimentos**

### Experimento fallido #1 — Torneo predictivo "demasiado bueno": copias casi duplicadas del dataset
`[2026-11-14 16:22] · Julián Agustín Lozano (Backend & ML)`
- **Hipótesis / qué intentamos:** correr el torneo de Fase 3 (4 familias, StratifiedKFold(5) en Train) sobre el dataset limpio de la Sesión IA-03 (N = 5.073: sin los 557 duplicados exactos).
- **Qué pasó (evidencia numérica):** Random Forest con ROC-AUC 0,973 en CV y 0,988 en Test; OOB 0,985. El árbol CART **sin podar** obtuvo AUC 0,917 en Test, **mejor** que el podado por CV (0,857): lo opuesto a lo esperable si sobreajusta.
- **Diagnóstico de la causa:** comparando cada cliente con el de ID + 1470 aparecen **2.638 pares** que coinciden en 14-19 de las 19 columnas, con el **mismo Churn en el 100%** de los pares; entre filas no emparejadas coinciden 0-10 columnas, sin casos intermedios (14-19 vs 0-10). Las columnas que más difieren dentro del par son `CouponUsed` (1.469 pares), `MaritalStatus` (827) y `SatisfactionScore` (568). El archivo es un dataset **aumentado con ruido**: 2.638 originales + 2.638 copias perturbadas + 354 clientes sin par. Los 557 duplicados exactos eran el subconjunto de copias sin perturbar. Con las copias, un cliente de Test tiene a su "gemelo" en Train → **leakage**: el árbol sin podar memoriza al gemelo y acierta.
- **Cuantificación:** RF con 300 árboles, ROC-AUC en StratifiedKFold(5): **0,993 con copias vs 0,948 sin copias**. Además, las copias violan la independencia de las observaciones (Fases 1-2: errores estándar subestimados) y, al tener `CouponUsed` perturbado con el mismo Churn, **diluyen el efecto causal de T hacia 0** (Fase 4).
- **Qué aprendimos:** un desempeño excepcional es un síntoma a auditar, no un logro. Los duplicados exactos no agotan el problema: hay que buscar **casi duplicados** con una métrica de coincidencia por columnas.
- **Corrección aplicada y resultado:** `near_duplicate_copies()` en `src/preprocessing.py` (ID + 1470 y ≥ 14/19 columnas iguales; se conserva el ID más bajo), aplicada antes del split junto con los duplicados exactos restantes (10). **5.630 → 2.982 filas** · churn 16,84% → 17,30% · T tratados 80,73% (133 NA) · Train 2.385 / Test 597. Reversible con `load_raw_data(drop_near_duplicates=False)` para sensibilidad. Tests: `test_near_duplicate_copies_detected`, `test_real_raw_dataset_copy_structure`. La Fase 1 (PR #11) debe re-ejecutarse sobre el dataset corregido.

### Experimento fallido #2 — _título_
- (misma estructura)

> Candidatos probables: árbol CART sin podar (AUC Train≈1 vs Test), Naive Bayes descalibrado por dependencia `OrderCount`–`CouponUsed`, logística inestable por VIF alto, ATE "inflado" al incluir el colisionador.

---

## Entrada 4 — Formalización del Debate Causal
**Responsable:** Backend & ML / Causal + Project Lead · **Estado:** 🟡 4.1–4.3 completados por Project Lead (Carola Nasif, 2026-11-14) · pendiente 4.4 por Backend & ML (Julián Agustín Lozano)

**Contrato estructural del DAG (Opción A confirmada):**
- **Tratamiento ($T$):** `coupon_any` $= \mathbb{I}(\text{CouponUsed} \ge 1)$ (entero nullable; `133` NA preservados en $N = 2.982$).
- **Resultado ($Y$):** `Churn` $\in \{0, 1\}$ (abandono en el último mes; retención $= 1 - Y$).
- **Set de ajuste backdoor ($Z$, 13 confusores pre-tratamiento):** `Tenure`, `Complain`, `SatisfactionScore`, `CityTier`, `PreferedOrderCat`, `HourSpendOnApp`, `PreferredLoginDevice`, `PreferredPaymentMode`, `Gender`, `MaritalStatus`, `WarehouseToHome`, `NumberOfAddress`, `NumberOfDeviceRegistered` (cada uno con flechas $Z_j \to T$ y $Z_j \to Y$).
- **Mediadores ($M$, excluidos de $Z$):** `OrderCount`, `CashbackAmount` ($T \to M \to Y$).
- **Colisionador ($C$, excluido de $Z$):** `DaySinceLastOrder` ($T \to C \leftarrow Y$).
- **Latente no observada ($U$):** $U_{\text{engagement}} \to \{\text{HourSpendOnApp},\, T,\, Y\}$.

### 4.1 Justificación arco por arco
| Arco | Mecanismo de dominio | Temporalidad | Confianza (alta/media/baja) |
|---|---|---|---|
| `Tenure → T` | Clientes con mayor antigüedad conocen mejor los cupones y promociones del CRM, lo que eleva la probabilidad de redención (autoselección de usuarios recurrentes). | Pre-tratamiento genuino (meses desde el alta preceden al uso de cupones del último mes). | Alta |
| `Tenure → Y` | La antigüedad consolida hábito de compra, confianza y costos de cambio (*switching costs*), reduciendo la tasa de abandono (`Churn`). | Pre-resultado (antigüedad acumulada antecede a la decisión de baja). | Alta |
| `Complain → T` | Los reclamos disparan la emisión de cupones compensatorios (*recovery/apology coupons*) por parte de atención al cliente y reglas automáticas de retención. | Pre-tratamiento asumido dentro de la ventana mensual (reclamo precede al cupón de recuperación). | Alta |
| `Complain → Y` | Un reclamo refleja una falla operativa o de calidad que incrementa directamente la propensión al abandono (Fase 1: `32,63%` con reclamo vs `11,11%` sin reclamo; $\Delta = +21{,}52\text{ pp}$, $\text{OR} = 3{,}88$). | Pre-resultado (el incidente ocurre antes de la baja efectiva). | Alta |
| `SatisfactionScore → T` | Usuarios con menor satisfacción reciben/buscan más incentivos compensatorios o promocionales, mientras que los muy conformes aprovechan beneficios vigentes. | **Discutible (ambigua):** medida en el mismo corte "último mes" que $T$; podría estar afectada por el cupón (sesgo post-tratamiento si $T \to \text{Satisfaction}$). Se incluye en $Z$ base y se evalúa sensibilidad excluyéndola. | Baja / Media |
| `SatisfactionScore → Y` | Menor satisfacción general anticipa desafección con el servicio y mayor probabilidad de abandono (`Churn`). | Ambigua en corte transversal (incluida en $Z$ por contrato; contrastar sensibilidad). | Media |
| `HourSpendOnApp → T` | Mayor exposición horaria en la app aumenta la visibilidad de banners, notificaciones push y cupones disponibles. | **Discutible (simultánea):** promedio de horas en el último mes es coetáneo con `CouponUsed` y refleja también la latente $U_{\text{engagement}}$. | Media |
| `HourSpendOnApp → Y` | Mayor uso activo de la app traduce hábito digital e involucramiento que reducen el riesgo de abandono (`Churn`). | Simultánea dentro del corte mensual; proxy parcial de $U_{\text{engagement}}$. | Media |
| `CityTier → T` | La segmentación geográfica por tamaño de ciudad condiciona campañas promocionales regionales, intensidad competitiva y alianzas de cupones. | Pre-tratamiento estructural (ubicación del cliente). | Alta |
| `CityTier → Y` | Los tiempos de envío, cobertura logística y alternativas de competidores varían según el nivel urbano (`CityTier`), afectando la retención. | Pre-resultado estructural. | Alta |
| `PreferedOrderCat → T` | La categoría preferida segmenta las campañas del motor de recomendaciones y el calendario de descuentos por vertical (Fase 1: `CashbackAmount` varía con $\eta^2 = 0{,}827$). | Pre-tratamiento (preferencia consolidada de compra). | Alta |
| `PreferedOrderCat → Y` | La frecuencia de recompra y fidelidad natural difieren entre verticales (p. ej. `Grocery` y `Fashion` frente a electrónica como `Mobile Phone` o `Laptop & Accessory`). | Pre-resultado (patrón de consumo por categoría). | Alta |
| `WarehouseToHome → T` | Distancias mayores al centro de distribución encarecen o demoran la entrega, incentivando cupones promocionales/flete gratis para sostener conversión. | Pre-tratamiento estructural (distancia geográfica). | Media / Alta |
| `WarehouseToHome → Y` | Mayor distancia logística aumenta fricciones de entrega y riesgo de insatisfacción o abandono. | Pre-resultado estructural. | Media / Alta |
| `PreferredLoginDevice`, `PreferredPaymentMode` $\to T$ y $\to Y$ | Canal de acceso (`Mobile Phone` vs `Computer`) y medio de pago (`Credit Card`, `Debit Card`, `UPI`, `Cash on Delivery`, `E wallet`) activan promos bancarias/app exclusivas y reflejan fricción de checkout asociada al churn. | Pre-tratamiento estructural / perfil de cuenta. | Media / Alta |
| `Gender`, `MaritalStatus`, `NumberOfAddress`, `NumberOfDeviceRegistered` $\to T$ y $\to Y$ | Rasgos demográficos y huella del hogar/dispositivos que capturan estabilidad del perfil, uso familiar compartido y adopción multidispositivo ligados tanto al uso de beneficios como a la permanencia. | Pre-tratamiento (atributos de perfil). | Media |
| `T → OrderCount → Y` | **Mediador $M_1$:** usar un cupón reduce el precio efectivo e induce compras adicionales en el mes ($T \to \text{OrderCount}$); mayor volumen de pedidos consolida retención ($\text{OrderCount} \to Y$). Condicionar en `OrderCount` bloquea el efecto mediado y subestima el ATE total. | Post-tratamiento y pre-resultado (asumido en ventana mensual). | Media / Alta |
| `T → CashbackAmount → Y` | **Mediador $M_2$:** la redención de incentivos dispara reintegros monetarios acoplados ($T \to \text{CashbackAmount}$), que retienen al usuario en la plataforma (Fase 1: `Churn=1` recibe $-\$19{,}36$ menos de cashback). Excluido de $Z$ por ser post-tratamiento. | Post-tratamiento (consecuencia económica del incentivo usado). | Media |
| `T → Y` | **Efecto directo:** impacto causal directo del incentivo sobre la retención (percepción de valor y lealtad hacia el motor CRM) no canalizado por pedidos adicionales o cashback acumulado. | Ventana del último mes ($T$ precede a la decisión de churn). | Media |
| `T → DaySinceLastOrder ← Y` | **Colisionador $C$:** redimir un cupón implica actividad reciente y acorta los días desde la última compra ($T \to C$, signo $-$); a la inversa, el proceso de abandono detiene compras y alarga esa recencia ($Y \to C$, signo $+$). Condicionar en `DaySinceLastOrder` abre la ruta no causal $T \to C \leftarrow Y$ (sesgo de Berkson / casi-leakage temporal). | Post-tratamiento y post/co-resultado (descendiente común). | Alta |
| `U_engagement → {HourSpendOnApp, T, Y}` | **Latente no observada:** afinidad intrínseca con la marca impulsa horas en app, cacería de incentivos y menor abandono; no se observa directamente en el dataset. | Latente continua (confusión residual). | Alta |

### 4.2 Variables latentes no observadas
1. **`U_engagement` (Afinidad intrínseca con la marca / intención latente de compra):**
   - **A qué afecta:** incrementa `HourSpendOnApp` ($+$) y la probabilidad de usar cupones $T$ ($+$), mientras reduce directamente el abandono $Y = \text{Churn}$ ($-$).
   - **Dirección del sesgo esperado sobre el ATE:** como $U_{\text{engagement}}$ correlaciona positivamente con $T$ y negativamente con `Churn`, si `Tenure` y `HourSpendOnApp` no la absorben por completo, genera un **sesgo negativo** (hace que el estimador parezca *más protector* —más negativo en puntos porcentuales de churn— de lo que realmente es, confundiendo a usuarios *"sure things"* con eficacia del cupón).
2. **`U_severidad_incidente` (Gravedad no observada de fallas de servicio):**
   - **A qué afecta:** `Complain` es un indicador binario (`0/1`) que oculta la intensidad del problema (demora leve vs. pedido dañado o contracargo). Una falla severa dispara cupones compensatorios de mayor esfuerzo ($U \to T$, $+$) y eleva fuertemente `Churn` ($U \to Y$, $+$).
   - **Dirección del sesgo esperado sobre el ATE:** asociación positiva tanto con $T$ como con $Y$, introduciendo **sesgo positivo sobre `Churn`** que atenúa el efecto protector o incluso invierte su signo hacia $\text{ATE} > 0$ (mecanismo de paradoja de Simpson residual dentro de `Complain = 1`).
3. **`U_sensibilidad_precio` (Perfil *deal-hunter* / elasticidad al descuento):**
   - **A qué afecta:** usuarios oportunistas buscan activamente cupones ($U \to T$, $+$) pero cambian de plataforma apenas terminan las promociones ($U \to Y$, $+$).
   - **Dirección del sesgo esperado sobre el ATE:** introduce **sesgo positivo sobre `Churn`** (subestima la retención genuina generada por el incentivo).
- **Estrategia de análisis de sensibilidad (`add_unobserved_common_cause` de DoWhy):** En Fase 4 se evalúa la robustez del estimando backdoor mediante `refute_estimate(..., method_name="add_unobserved_common_cause")`, inyectando un confusor sintético con distintas magnitudes de efecto sobre $T$ y sobre $Y$ para determinar el umbral de confusión oculta necesario para anular o revertir el signo del ATE; se complementa re-estimando el modelo causal al excluir covariables de temporalidad discutible (`SatisfactionScore` y `HourSpendOnApp`).

### 4.3 Amenazas a la validez interna
- **Positividad / soporte común (*overlap*):** En `make_clean_dataset()` ($N = 2.982$), sobre las `2.849` filas con tratamiento observado (`133` NA en `coupon_any`, $4{,}46\%$), el **$80{,}73\%$ ($n = 2.300$) corresponde a tratados ($T=1$) y solo el $19{,}27\%$ ($n = 549$) a controles ($T=0$)**. En subgrupos de alta propensión ($\hat{e}(Z) \to 1$), los pesos IPW del grupo control $1 / (1 - \hat{e}(Z))$ pueden volverse inestables, exigiendo inspección de histogramas de propensión, *trimming* y verificación de balance ($\text{SMD} < 0{,}1$). Asimismo, en Fase 1 se comprobó que los `133` NA de `CouponUsed` siguen un mecanismo MAR ($\text{AUC} = 0{,}9652$, $\chi^2$ vs `Churn` $= 19{,}88$, $p = 8{,}23 \times 10^{-6}$), por lo que imputarlos ingenuamente como `0` contaminaría el grupo control.
- **Error de medición en $T$ (TOT vs. ITT y ruido en el archivo crudo):** La variable observada `CouponUsed` mide **redención efectiva del cupón**, no asignación o envío del incentivo por el motor de recomendaciones. Por lo tanto, el estimando captura un efecto tipo *Treatment on the Treated* (TOT) mezclado con autoselección de canje, no un *Intention-to-Treat* (ITT) de campaña. Adicionalmente, la autopsia forense de la Entrada 3 (Experimento #1) demostró que en las **`2.638` copias casi duplicadas (`ID + 1470`)** eliminadas en preprocesamiento, `CouponUsed` era **la columna más perturbada** (cambiaba en **`1.469` pares** con idéntico `Churn`), lo que introducía error de medición artificial en $T$ y atenuación causal hacia cero en los datos crudos.
- **Orden temporal asumido (corte transversal "último mes"):** `CouponUsed`, `OrderCount`, `CashbackAmount`, `DaySinceLastOrder`, `HourSpendOnApp`, `Complain` y `SatisfactionScore` corresponden a una misma ventana mensual agregada sin marcas de tiempo; la dirección $Z \to T \to M \to Y$ y $T \to C \leftarrow Y$ depende de supuestos sustantivos del dominio.
- **Causalidad inversa y asignación reactiva (`Complain → T`):** El CRM asigna cupones de retención de manera reactiva ante reclamos recientes (`Complain = 1` presenta `32,63%` de churn frente a `11,11%` sin reclamo) o frente a señales de desvinculación inminente no registradas en $Z$.
- **Confusión residual:** Variables potencialmente relevantes (nivel de ingresos, promociones de la competencia, antigüedad de suscripción o precios relativos) no están disponibles en la tabla.
- **SUTVA (interferencia y múltiples versiones del tratamiento):** Puede haber interferencia entre clientes del mismo hogar o dispositivos compartidos, y la binarización $T = \mathbb{I}(\text{CouponUsed} \ge 1)$ agrupa intensidades heterogéneas (`CouponUsed` varía de `1` a `16` entre tratados) y distintos valores monetarios de incentivo bajo una misma etiqueta `T=1`.
- **Selección muestral y validez externa:** El archivo crudo de Kaggle (`5.630` filas de procedencia corporativa anónima) había sido aumentado con ruido (`2.638` pares `ID + 1470` más `10` duplicados exactos adicionales; `557` duplicados exactos en bruto), dejando **$N = 2.982$** observaciones únicas (`Train = 2.385`, `Test = 597`). Los hallazgos son válidos para esa cartera histórica y no deben extrapolarse sin recalibración temporal.

### 4.4 Resultados y refutaciones
| Estimador | ATE | IC 95% | Comentario |
|---|---|---|---|
| Ingenuo Y ~ T | | | |
| Backdoor regresión lineal | | | |
| IPW | | | |
| LinearDML | | | |
| Con M incluido (trampa) | | | |
| Con C incluido (trampa) | | | |

| Refutador | Resultado | Semáforo |
|---|---|---|
| Random Common Cause | | |
| Placebo Treatment | | |
| Data Subset | | |

---

## Entrada 5 — Trazabilidad y Uso Ético de Asistentes de IA
**Responsable:** QA & Validation Engineer (con aportes de todos) · **Estado:** 🟢 abierta (registro vivo)

**Política del equipo:** la IA se usa como *tutor y acelerador*, nunca como fuente de verdad. Toda afirmación numérica o de API generada por IA se verifica contra (a) los datos, (b) la documentación oficial o (c) un test determinístico antes de entrar a `main`.

### Registro de sesiones

#### Sesión IA-01 · 2026-11-12 17:30 · Diseño arquitectónico (Sprint 0)
- **Herramienta:** Google Antigravity (agente de IA con acceso al workspace) · modelo Claude Opus.
- **Operador:** Carola Nasif (Project Lead)
- **Equipo registrado:** Carola Nasif, Julián Agustín Lozano, Emilia Mortarini (3 integrantes).
- **Prompt utilizado (resumen fiel):**
  > "Actuá como Principal Data Scientist, Arquitecto de Software y Tutor Metodológico para nuestro TPI 'Decision Intelligence Capstone' (EAyMR, UCA 2026). REGLA CRÍTICA: todavía NO escribas código de modelado, notebooks ni la app. [Contexto: entrega técnica dom 15/11 23:59, defensa mié 18/11, 4 roles Scrum, tema 'Retorno causal de algoritmo de recomendación', requisitos del dataset y rúbrica de 5 fases]. Tareas: (1) proponer 3 datasets reales de e-commerce/recomendación con roles T, Y, Z, M, C; (2) crear la estructura de carpetas `/src`, `/data`, `/notebooks`, `/app`, `/tests` con esqueletos, `README.md` y `requirements.txt`; (3) crear en `/docs` las propuestas de datasets y DAGs, la guía metodológica paso a paso y el plan Scrum de 72 hs; crear `BITACORA.md` con las 5 entradas, la Entrada 1 redactada y esta sesión registrada en la Entrada 5."
  > Insumos adjuntos: `Enunciado_TP_Integrador_EAyMR_2026.pdf`, `Rubrica_Evaluacion_TP_Integrador.pdf`, captura de la *Rúbrica complementaria* y PDF "Dataset TPI" del aula virtual EVA.
- **Artefactos generados:** estructura del repo, esqueletos `src/*.py` (solo firmas + `NotImplementedError`), `app/app.py` (layout sin lógica), `tests/test_quality_gate.py`, `docs/01-03`, este archivo.
- **Discrepancias detectadas entre el prompt y los documentos oficiales (verificadas por la IA contra los PDF):**
  1. El prompt agrupa "Fase 5 = 25 pts"; la Rúbrica Detallada lo separa en **D5 Scrum/Bitácora (10 pts) + D6 Deploy/Código/Defensa (15 pts)**. La suma coincide, pero se adopta el desglose oficial.
  2. El enunciado fija la **entrega oficial el miércoles 18/11 23:59** (EVA); el domingo 15/11 es un *code freeze* interno del equipo, no la fecha de la cátedra.
  3. La defensa dura **12-15 minutos** (enunciado), dato que el prompt no explicitaba.
- **Afirmaciones de la IA que el equipo DEBE verificar (posibles alucinaciones):**

| # | Afirmación generada | Cómo se verifica | Responsable | Estado |
|---|---|---|---|---|
| V1 | Dataset A: 5.630 filas, 20 columnas, ≈17% churn, ~4-6% NA en 7 columnas | `df.shape`, `df.isna().mean()`, `df.Churn.mean()` + `test_real_dataset_dimensions` | QA | ✅ 5.630 × 20 · churn 16,84% · NA 4,46-5,45% en las 7 columnas citadas (Sesión IA-03) |
| V2 | Dataset A tiene categorías duplicadas (`Phone`/`Mobile Phone`, `CC`/`Credit Card`, `COD`) | `value_counts()` | PL | ⚠️ correcta pero incompleta: faltaba `Mobile`/`Mobile Phone` en `PreferedOrderCat` (Sesión IA-03) |
| V3 | Olist: voucher ≈5% de pagos, recompra ≈3%, `review_comment_message` ≈58% vacío | conteos sobre las tablas | PL | ☐ |
| V4 | Lenta: 687.029 × 193 + `group` + `response_att`; nombres de columnas citados (`response_sms`, etc.) | `fetch_lenta().data.columns` | ML | ☐ |
| V5 | APIs: `CausalModel`, `identify_effect`, métodos `backdoor.linear_regression` / `backdoor.propensity_score_weighting`, refutadores `random_common_cause` / `placebo_treatment_refuter` / `data_subset_refuter`; `LinearDML`, `CausalForestDML` | Docs oficiales de DoWhy/EconML en la versión instalada + smoke test | ML | ☐ |
| V6 | Rangos de versiones de `requirements.txt` compatibles entre sí | `pip install` limpio + `pytest` | QA | ❌ → ✅ corregido (ver Sesión IA-02) |
| V7 | Argumento de hipersensibilidad de Shapiro-Wilk y sesgo MDI (Strobl et al., 2007) | Cita bibliográfica + experimento con columna de ruido | ML | ☐ |
| V8 | Signo del sesgo de confusión esperado (H-causal-1/2) | Es una **hipótesis**, no un resultado: se contrasta en Fase 4 | ML | ☐ |

- **Pruebas determinísticas creadas para neutralizar errores de IA:** `tests/test_quality_gate.py` (split 80/20 disjunto y estratificado, escalador y PCA ajustados solo en Train, N ≥ 1.500 y ≥ 10 explicativas, M y C fuera del set de ajuste).
- **Errores / alucinaciones detectadas en esta sesión:** _completar por el equipo tras la verificación V1-V8_.

#### Sesión IA-02 · 2026-11-14 14:47 · Entorno reproducible y verificación V6
- **Herramienta:** Claude Code (Claude Opus 5.5), con acceso al repositorio local.
- **Operador:** Julián Agustín Lozano (Backend & ML / Causal Engineer).
- **Prompt (resumen fiel):** "Cloná el repo e instalá lo que falte." Luego: "Fijá las versiones en `requirements.txt` en una rama aparte con PR."
- **Qué se hizo:** entorno con Python 3.11 (vía `uv`) e instalación limpia de `requirements.txt`, seguida de un smoke test de DoWhy y EconML sobre datos sintéticos con ATE verdadero = 2.
- **Error detectado (alucinación de la Sesión IA-01, punto V6):** los rangos abiertos de `requirements.txt` resolvían a versiones incompatibles entre sí:
  1. `networkx 3.6` eliminó `nx.algorithms.d_separated`, que `dowhy 0.12` usa en `identify_effect` → `AttributeError`.
  2. `pandas 3.1` rompe la indexación posicional que usa `dowhy 0.12` en `backdoor.linear_regression` → `KeyError: 0`.
- **Corrección:** `pandas>=2.1,<3` y `networkx>=3.1,<3.5`. Con eso: ingenuo 5,35 · regresión 2,08 · IPW 2,29 · LinearDML 2,04 · placebo 0,007 · random common cause 2,08 · data subset 2,07.
- **Qué se aceptó:** los pines de versión y el smoke test. **Qué se rechazó:** subir a `dowhy 0.14` (obligaba a bajar `scipy` y no se había probado con el resto del stack).
- **Test asociado:** `tests/test_causal_stack.py` (DoWhy backdoor y EconML LinearDML deben recuperar ATE = 2 ± 0,3; el ingenuo debe estar sesgado).

#### Sesión IA-03 · 2026-11-14 15:10 · Carga, limpieza, contrato de datos y split
- **Herramienta:** Claude Code (Claude Opus 5.5), con acceso al repositorio local.
- **Operador:** Julián Agustín Lozano (Backend & ML / Causal Engineer).
- **Prompt (resumen fiel):** "Arrancá con la carga, limpieza y split del dataset A en una rama nueva."
- **Qué se hizo:** `src/preprocessing.py` (contrato T/Y/Z/M/C según `docs/01`, carga, normalización de categorías, eliminación de duplicados, `coupon_any`, split 80/20 estratificado persistido en `artifacts/split_idx.json`, `build_preprocessor`) y `tests/test_preprocessing.py`.
- **Hallazgos sobre el dataset (verificación V1/V2 y auditoría):**
  1. V1 confirmada. V2 incompleta: además de las 3 duplicaciones citadas, `PreferedOrderCat` tiene `Mobile` y `Mobile Phone`.
  2. **No detectado por la IA en la Sesión IA-01:** 556 pares de filas idénticas salvo el `CustomerID`, con IDs separados exactamente por 1470 (p. ej. 51046 ↔ 52516); 557 tras unificar categorías. Sin etiquetas de churn contradictorias. Se interpretan como copias del proceso de generación del dataset.
  3. T = `coupon_any`: 81,25% tratados y 210 NA (de `CouponUsed`) tras la limpieza → la positividad/overlap será crítica en Fase 4.
- **Decisión:** eliminar los duplicados **antes** del split (si no, el mismo cliente puede quedar en Train y en Test: leakage). Antes → después: 5.630 → 5.073 filas; churn 16,84% → 16,58%. Se conserva `load_raw_data(drop_duplicates=False)` para análisis de sensibilidad. Los NA de T no se imputan en el preprocesamiento: se decide en Fase 4.
- **Actualización `[2026-11-14 16:22]`:** la limpieza resultó incompleta. El torneo de Fase 3 (con la misma herramienta) reveló además **2.638 copias casi duplicadas** (ID + 1470, ≥ 14/19 columnas iguales) que la IA no había detectado en la auditoría inicial. Se eliminan: N final **2.982**. Detalle, evidencia y tests en Entrada 3, Experimento fallido #1.
- **Qué se rechazó:** imputar o binarizar `CouponUsed` con NA = 0 (supone sin evidencia que el faltante significa "no usó cupón").
- **Test asociado:** `tests/test_preprocessing.py` (categorías normalizadas, duplicados eliminados, NA de T preservados, dataset sin imputar, split 80/20 estratificado y disjunto, round-trip del split, preprocesador ajustado solo en Train, categorías nuevas en Test).

#### Sesión IA-04 · 2026-11-14 15:32 · Fase 1: auditoría de faltantes, contrastes formales, potencia y ANOVA
- **Herramienta:** Google Antigravity (agente de IA con acceso al workspace).
- **Operador:** Carola Nasif (Project Lead & Scrum Master).
- **Prompt (resumen fiel):** "Revisión cruzada de PR #1 y #2 sin mergear; cerrar Entrada 1 de BITACORA.md con decisión A y SHA-256; crear milestones e issues por integrante en GitHub; en rama feat/fase1-inferencia (desde feat/preprocesamiento) implementar solo missingness_report en src/preprocessing.py y las 6 funciones de Fase 1 en src/inference.py (sin tocar Fase 2), con tests en tests/test_inference.py y Entrada 2 F1 en la bitácora."
- **Qué se hizo:**
  - `src/preprocessing.py`: `missingness_report` con conteo/% de NA, $\chi^2$ y $V$ de Cramér vs `Churn`, regresión logística del indicador de NA sobre covariables observadas (LRT en `statsmodels` + ROC-AUC en CV estratificada 5-fold) y test global MCAR de Little por algoritmo EM.
  - `src/inference.py`: `cohens_d`, `two_sample_test` (Welch t, IC analítico y bootstrap 5.000 réplicas, Mann-Whitney U, Cohen's d, potencia y MDE con `TTestIndPower`), `chi2_independence` ($\chi^2$, $V$ de Cramér, diferencia de proporciones con IC 95%), `normality_diagnostics` (asimetría, curtosis en exceso, Shapiro/D'Agostino informativos y Q-Q), `one_way_anova_suite` (Levene `center='median'`, ANOVA clásico, Welch ANOVA, Kruskal-Wallis, $\eta^2$, $\omega^2$, $\varepsilon^2$ y árbol de decisión) y `posthoc` (`games-howell`, `tukey`, `bonferroni`).
  - `BITACORA.md`: Entrada 1 cerrada (v1.0, Opción A unánime 3/3, SHA-256 `db70f1e34bc5326887ea3fb3d2a0854e82c577d55ef3a76fdfb25328631bd44d`) y Entrada 2 filas F1 completadas con los resultados reales sobre `make_clean_dataset()`.
- **Errores / incompatibilidades detectadas y corregidas:**
  1. `pingouin 0.6.1` renombró columnas respecto a 0.5.x (`p_unc` en lugar de `p-unc` en `welch_anova`; `p_tukey` en lugar de `p-tukey` y `mean_A`/`mean_B` en lugar de `mean(A)`/`mean(B)` en `pairwise_tukey`/`pairwise_gameshowell`). Se agregó `_pick_col` para compatibilidad con ambas versiones.
  2. La separación casi completa de algunos indicadores de NA producía warnings de convergencia/Hessiano en `statsmodels.Logit`; se aislaron dentro de `warnings.catch_warnings()` usando BFGS primero y respaldando la evidencia MAR con ROC-AUC out-of-fold.
- **Qué se aceptó / qué se rechazó:** Se aceptaron los cálculos exactos en `src/` sin duplicar lógica ni tocar las funciones asignadas a Emilia (Fase 2). Se rechazó cualquier redacción que dijera "se acepta H0", usando estrictamente "Se rechaza H0" / "No se rechaza H0".
- **Test asociado:** `tests/test_inference.py` (tests unitarios con datos sintéticos + verificación de fronteras de rol + integración sobre `make_clean_dataset()`).
- **Actualización [2026-11-14 16:50]:** re-ejecutado sobre N = 2.982 por la Entrada 3, Experimento #1 (eliminación de 2.638 copias casi duplicadas con `ID + 1470` en `src/preprocessing.py`); se actualizaron la fecha real en Entrada 1 §1.5, la Entrada 2 (filas F1 y §2.1) y la descripción del PR #11 sin cambios en el código de Fase 1.

#### Sesión IA-05 · 2026-11-14 17:18 · Formalización del debate causal (Entrada 4 §4.1–4.3) y notebook maestro (Sección 0, Fase 1 y esqueleto)
- **Herramienta:** Google Antigravity (agente de IA con acceso al workspace).
- **Operador:** Carola Nasif (Project Lead & Scrum Master).
- **Prompt (resumen fiel):** "En rama nueva docs/entrada4-notebook desde origin/feat/fase1-inferencia, completar BITACORA.md Entrada 4 §4.1–4.3 sin tocar §4.4 (de Julián) ni poner resultados numéricos de Fase 4; armar el esqueleto del notebook maestro notebooks/TP_Integrador_Final.ipynb con portada, Sección 0 (setup e imports desde src), Sección 1 (Fase 1 completa con Markdown teórico previo, gráficos y tablas numeradas), encabezados pendientes de Secciones 2–6, referencias bibliográficas verificadas y declaración de uso de IA; ejecutar Restart & Run All, pasar ruff/pytest y abrir PR con 'Depende de #11'."
- **Qué se hizo:**
  - `BITACORA.md` (Entrada 4 §4.1–4.3): formalización del DAG Opción A, tabla de justificación arco por arco (`Z`, `M`, `C`, `U_engagement`), discusión de temporalidad ambigua en `SatisfactionScore` y `HourSpendOnApp`, 3 latentes no observadas con dirección esperada del sesgo sobre `Churn` y protocolo de sensibilidad (`add_unobserved_common_cause`), y 7 amenazas a la validez interna cuantificadas sobre `N = 2.982` (`80,73%` tratados / `549` controles / `133` NA MAR, TOT vs ITT, `1.469` pares perturbados en `CouponUsed`).
  - `notebooks/TP_Integrador_Final.ipynb`: portada institucional + índice general, Sección 0 (`cleaning_summary()` y `load_split()`), Sección 1 completa invocando exclusivamente funciones de `src.preprocessing` y `src.inference` (auditoría univariada/bivariada, Pearson vs Spearman, faltantes MCAR/MAR/MNAR, Welch t + Mann-Whitney + χ² + potencia, ANOVA/Welch/Kruskal + Q-Q por grupo + post-hoc Games-Howell/Tukey/Bonferroni), encabezados pendientes de Secciones 2–6 con responsable, bibliografía verificada por búsqueda web y remisión a Entrada 5 para uso de IA.
- **Qué se aceptó / qué se rechazó:** Se aceptó mantener intacto el apartado §4.4 (`Resultados y refutaciones`) para evitar conflictos con la rama de Fase 4 de Julián y prohibir lógica estadística duplicada en el notebook. Se rechazó incluir estimaciones numéricas anticipadas de ATE/CATE o citas bibliográficas sin verificación previa de autor, año, título y revista.
- **Test y verificación asociada:** Ejecución integral con `jupyter nbconvert --to notebook --execute --inplace notebooks/TP_Integrador_Final.ipynb` (sin errores, salidas persistidas) + Quality Gate (`ruff check .` y `pytest -q`).

#### Sesión IA-06 · 2026-11-14 18:28 · Conclusiones ejecutivas (Sección 6), README editorial y esqueleto de slides
- **Herramienta:** Google Antigravity (agente de IA con acceso al workspace).
- **Operador:** Carola Nasif (Project Lead & Scrum Master).
- **Prompt (resumen fiel):** "Monitorear merges en orden #1 → #2 → #11 → #12 → #13 → #14 tras [APROBADO] de QA sin usar squash/rebase; en rama de #14 completar Sección 6 del notebook maestro (con resultados verificados de model_results.json y causal_results.json, umbral -ATE·V > c, recomendación y limitaciones); en nueva rama docs/readme crear README.md editorial completo (licencia Kaggle verificada, diagrama Mermaid, comandos, tabla de resultados) y esqueleto de 10-12 slides PDF; respetar fronteras (no tocar src/inference.py, app/, notebooks/borradores/ ni filas 6-13 de Entrada 2)."
- **Qué se hizo:**
  - `notebooks/TP_Integrador_Final.ipynb` (Sección 6 completa + Índice actualizado): síntesis predictiva (`Random Forest + isotónica` ROC-AUC Test `0,9461`, Brier `0,0643`, ECE `0,0179`, top permutación `Tenure`/`Complain`/`PreferedOrderCat`/`DaySinceLastOrder` vs sesgo MDI en `CashbackAmount`) vs evidencia causal (`N = 2.012` en soporte común, `808` excluidos por positividad, ATE backdoor `−1,16 pp` IC 95% `[−4,60; +2,28]`, IPW `−0,96 pp`, LinearDML `+1,11 pp`, CausalForestDML `+1,42 pp`, dosis-respuesta `+1,07 pp/cupón`, 3/3 refutaciones verdes, sensibilidad `[−1,75; +6,17] pp`, CATE `Complain` `+7,1 pp`, p = `0,065`), regla económica `$-\text{ATE}\cdot V > c$` (`c < 0,046·V` en cota optimista) y recomendación de experimento A/B (`RCT`).
  - Rama `docs/readme` (PR nuevo): `README.md` editorial con licencia **CC BY-NC-SA 4.0** verificada en Kaggle, diagrama Mermaid, comandos de reproducción y tablas clave; más esqueleto de 12 slides (`docs/04_esqueleto_slides.md` y `docs/04_esqueleto_slides.pdf`).
- **Qué se aceptó / qué se rechazó:** Se aceptó leer los JSON reales de `origin/feat/fase3-modelos` y `origin/feat/fase4-causal` vía `git show` para transcribir cifras exactas sin extrapolación y mantener estrictamente la frontera de archivos asignados. Se rechazó modificar `src/inference.py`, `app/`, `notebooks/borradores/`, filas 6–13 de Entrada 2 o mergear PRs sin `[APROBADO]` explícito de QA (`@EmiMortarini`).
- **Test y verificación asociada:** Re-ejecución completa del notebook (`nbconvert --execute`, 0 errores) y Quality Gate (`ruff check .` + `pytest -q`).

#### Sesión IA-07 · _fecha_ · _tema_
- Herramienta / modelo · Operador · Prompt · Qué se aceptó · Qué se rechazó y por qué · Test asociado.

# 📦 `/data` — Instrucciones de descarga

> Los datos crudos **no se versionan** (ver `.gitignore`). Cada integrante los descarga con estos pasos.
> El dataset final se elige el **jueves 12/11 a las 21:00** (ver `docs/03_plan_accion_scrum_72hs.md`).

```
data/
├── raw/         # archivo original tal cual se descargó (inmutable, nunca se edita)
└── processed/   # dataset_clean.parquet (salida de src/preprocessing.py, SIN imputar: la imputación va dentro del Pipeline)
```

## Prerrequisito: Kaggle API

```bash
pip install kaggle
# Kaggle → Settings → API → "Create New Token" → mover kaggle.json a ~/.kaggle/ y:
chmod 600 ~/.kaggle/kaggle.json
```

## Opción A — E-Commerce Customer Churn (Kaggle) ⭐ recomendada para 72 hs

- Fuente: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
- 5.630 clientes × 20 columnas · archivo `E Commerce Dataset.xlsx` (hoja `E Comm` + hoja de diccionario).

```bash
kaggle datasets download -d ankitverma2010/ecommerce-customer-churn-analysis-and-prediction -p data/raw --unzip
```

## Opción B — Olist Brazilian E-Commerce (Kaggle)

- Fuente: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- ~99k pedidos en 9 tablas relacionales (orders, customers, order_items, order_payments, order_reviews, products, sellers, geolocation, category translation).

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw/olist --unzip
```

## Opción C — Lenta Uplift (BigTarget Hackathon, vía scikit-uplift)

- Fuente: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html
- 687.029 clientes × 193 features + `group` (tratamiento SMS) + `response_att` (respuesta). **Experimento aleatorizado.**

```python
# pip install scikit-uplift
from sklift.datasets import fetch_lenta
ds = fetch_lenta()                          # descarga y cachea (~cientos de MB)
df = ds.data.assign(group=ds.treatment, response_att=ds.target)
df.sample(n=30_000, random_state=42).to_parquet("data/raw/lenta_sample_30k.parquet")
```

## Checklist de integridad (lo valida QA)

- [ ] Hash SHA-256 del archivo crudo registrado en `BITACORA.md` (Entrada 1): `shasum -a 256 data/raw/<archivo>`
- [ ] N ≥ 1.500 y ≥ 10 explicativas (test `test_real_dataset_dimensions`)
- [ ] Licencia y procedencia citadas en el README principal

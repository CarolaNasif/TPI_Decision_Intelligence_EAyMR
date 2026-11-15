"""Reentrena el torneo EN MEMORIA (no guarda artefactos) y exporta las curvas para las slides."""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
out = sys.argv[2]

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src import models as mdl  # noqa: E402

t0 = time.time()
res = mdl.run_tournament()
X_te, y_te = res["X_test"], res["y_test"]
models = {**res["models"], f"{res['best_name']} + isotónica": res["best_calibrated"]}

payload = {"elapsed_s": None, "roc": {}, "auc": {}, "reliability": {}}
for name, m in models.items():
    p = m.predict_proba(X_te[mdl.FEATURES])[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    payload["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    payload["auc"][name] = float(roc_auc_score(y_te, p))

rel = mdl.reliability_curves(models, X_te, y_te)
payload["reliability_columns"] = list(rel.columns)
payload["reliability"] = rel.to_dict(orient="records")
payload["elapsed_s"] = time.time() - t0
json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("ok", round(payload["elapsed_s"]), "s", {k: round(v, 4) for k, v in payload["auc"].items()})
print(rel.head(3).to_string())

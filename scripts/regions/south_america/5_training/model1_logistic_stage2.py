#!/usr/bin/env python3
"""W5: Logistic regression baseline on Stage 2 formulation (South America)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.evaluation.stage2_metrics import compute_stage2_metrics
from scripts.regions.shared.training.stage2_lgbm_core import (
    Stage2Config,
    _expansion_groups_from_batches,
    load_stage2_arrays,
    resolve_parquet_file,
)

OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ml_models"


def main() -> None:
    region = "south_america"
    train_path = resolve_parquet_file(region, "train.parquet")
    test_path = resolve_parquet_file(region, "test.parquet")
    expansion = _expansion_groups_from_batches(train_path, region, (2001, 2016))

    import pyarrow.parquet as pq
    from scripts.regions.shared.training.stage2_lgbm_core import STAGE2_EXCLUDE_COLS

    schema = pq.ParquetFile(train_path).schema_arrow
    feature_cols = [n for n in schema.names if n not in STAGE2_EXCLUDE_COLS and n not in ("row", "col")]

    X_tr, y_tr, _, _, g_tr = load_stage2_arrays(
        train_path, region, feature_cols, expansion, (2001, 2013)
    )
    test_exp = _expansion_groups_from_batches(test_path, region, (2017, 2019))
    X_te, y_te, _, _, g_te = load_stage2_arrays(
        test_path, region, feature_cols, test_exp, (2017, 2019)
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    metrics = compute_stage2_metrics(y_te.astype(np.float64), scores, g_te)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {"model": "logistic_regression", "task": "stage2_baseline", "timestamp": ts},
        "test_performance": metrics,
        "coefficients": {f: float(c) for f, c in zip(feature_cols, clf.coef_.ravel())},
        "intercept": float(clf.intercept_[0]),
    }
    path = OUT_DIR / f"model1_logistic_stage2_metrics_{ts}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Logistic Stage 2 NDCG@1%: {metrics['ndcg_at_1pct']:.4f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

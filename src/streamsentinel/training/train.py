"""Train, evaluate and register the detector.

1. Simulate separate train / validation / test streams with labelled anomalies.
2. Fit an Isolation Forest on the (unlabelled) training z-scores.
3. Pick the detector mode and thresholds with the best F1 on validation.
4. Report precision, recall, F1 and false alarms per hour on the untouched test stream.
5. Log everything to MLflow, register the model and promote it to ``champion`` if it beats
   the current champion on the test set.

    python -m streamsentinel.training.train
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from streamsentinel.features import MODEL_FEATURES
from streamsentinel.training.evaluate import (
    build_dataset,
    classification_metrics,
    predict,
    recall_by_type,
    select_thresholds,
)

log = logging.getLogger("streamsentinel.train")


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "window_seconds": args.window_seconds,
        "anomaly_rate": args.anomaly_rate,
        "history": args.history,
        "min_history": args.min_history,
    }
    t0 = time.time()
    train = build_dataset(args.train_hours, args.seed, **common)
    val = build_dataset(args.val_hours, args.seed + 1, **common)
    test = build_dataset(args.test_hours, args.seed + 2, **common)
    log.info(
        "Datasets: %d train / %d val / %d test windows (%.1fs)",
        len(train.z),
        len(val.z),
        len(test.z),
        time.time() - t0,
    )

    forest = IsolationForest(
        n_estimators=args.n_estimators, max_samples=256, random_state=args.seed
    )
    forest.fit(train.z)  # unsupervised: labels are never shown to the model

    def if_scores(data):
        return -forest.decision_function(data.z)

    best = select_thresholds(val, if_scores(val))
    test_zmax = np.abs(test.z).max(axis=1)
    test_if = if_scores(test)
    results = {}
    for mode, cfg in best.items():
        pred = predict(mode, test_zmax, test_if, cfg["z_threshold"], cfg["if_threshold"])
        results[mode] = {
            **cfg,
            "test": classification_metrics(pred, test.y, test.hours),
            "test_recall_by_type": recall_by_type(pred, test.labels),
        }

    chosen = max(results, key=lambda m: (results[m]["val"]["f1"], -results[m]["val"]["fp"]))
    return {"forest": forest, "results": results, "chosen": chosen, "sample": val.z[:5]}


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 1e9


def log_to_mlflow(args: argparse.Namespace, out: dict[str, Any]) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = MlflowClient()
    results, chosen = out["results"], out["chosen"]
    cfg = results[chosen]

    with mlflow.start_run(run_name=f"detector-{chosen}") as run:
        mlflow.log_params(
            {
                "mode": chosen,
                "z_threshold": _finite(cfg["z_threshold"]),
                "if_threshold": _finite(cfg["if_threshold"]),
                "history": args.history,
                "min_history": args.min_history,
                "window_seconds": args.window_seconds,
                "n_estimators": args.n_estimators,
                "train_hours": args.train_hours,
                "anomaly_rate": args.anomaly_rate,
                "features": ",".join(MODEL_FEATURES),
                "seed": args.seed,
            }
        )
        for mode, res in results.items():
            for name in ("precision", "recall", "f1", "false_alarms_per_hour"):
                mlflow.log_metric(f"{mode}_test_{name}", res["test"][name])
                mlflow.log_metric(f"{mode}_val_{name}", res["val"][name])
        for name in ("precision", "recall", "f1", "false_alarms_per_hour"):
            mlflow.log_metric(f"test_{name}", cfg["test"][name])
        for kind, recall in cfg["test_recall_by_type"].items():
            mlflow.log_metric(f"test_recall_{kind}", recall)
        mlflow.log_dict(
            {"chosen": chosen, "results": json.loads(json.dumps(results, default=float))},
            "evaluation.json",
        )
        sample = out["sample"]
        mlflow.sklearn.log_model(
            out["forest"],
            artifact_path="model",
            signature=infer_signature(sample, out["forest"].decision_function(sample)),
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)

    new_f1 = cfg["test"]["f1"]
    try:
        current = client.get_model_version_by_alias(args.model_name, args.alias)
        current_f1 = client.get_run(current.run_id).data.metrics.get("test_f1", -1.0)
    except MlflowException:
        current, current_f1 = None, -1.0

    promoted = args.force_promote or current is None or new_f1 >= current_f1
    if promoted:
        client.set_registered_model_alias(args.model_name, args.alias, version.version)
        log.info("Promoted %s v%s to @%s", args.model_name, version.version, args.alias)
    else:
        log.info(
            "Kept v%s as @%s (test F1 %.3f >= %.3f)",
            current.version,
            args.alias,
            current_f1,
            new_f1,
        )
    return {"run_id": run.info.run_id, "version": str(version.version), "promoted": promoted}


def print_report(out: dict[str, Any]) -> None:
    results, chosen = out["results"], out["chosen"]
    print("\nTest set (unseen during training and threshold selection)")
    print(f"{'mode':<10}{'precision':>10}{'recall':>9}{'F1':>8}{'false alarms/h':>16}")
    for mode, res in results.items():
        t = res["test"]
        marker = "  <- selected" if mode == chosen else ""
        print(
            f"{mode:<10}{t['precision']:>10.3f}{t['recall']:>9.3f}{t['f1']:>8.3f}"
            f"{t['false_alarms_per_hour']:>16.2f}{marker}"
        )
    print("\nRecall by anomaly type (selected detector)")
    for kind, recall in results[chosen]["test_recall_by_type"].items():
        print(f"  {kind:<18}{recall:>6.2f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-hours", type=float, default=8)
    p.add_argument("--val-hours", type=float, default=4)
    p.add_argument("--test-hours", type=float, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-seconds", type=int, default=int(os.getenv("WINDOW_SECONDS", "10")))
    p.add_argument("--anomaly-rate", type=float, default=0.02)
    p.add_argument("--history", type=int, default=180)
    p.add_argument("--min-history", type=int, default=30)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    p.add_argument("--experiment", default="streamsentinel")
    p.add_argument("--model-name", default=os.getenv("MODEL_NAME", "streamsentinel-detector"))
    p.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "champion"))
    p.add_argument("--force-promote", action="store_true")
    p.add_argument("--no-mlflow", action="store_true", help="only print the evaluation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    out = train_and_evaluate(args)
    print_report(out)
    if not args.no_mlflow:
        out["registry"] = log_to_mlflow(args, out)
    return out


if __name__ == "__main__":
    main()

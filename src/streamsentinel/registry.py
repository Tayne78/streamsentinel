"""Loading the champion detector from the MLflow model registry."""

from __future__ import annotations

import logging
from typing import Any

from streamsentinel.detector import DetectorConfig

log = logging.getLogger(__name__)


def load_champion(
    tracking_uri: str, model_name: str, alias: str
) -> tuple[DetectorConfig, Any] | None:
    """Return (config, isolation forest) for ``models:/<name>@<alias>``, or None if unavailable.

    Thresholds and the detector mode were chosen on validation data during training and are
    stored as run params, so the registry is the single source of truth for what runs live.
    """
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, alias)
        params = client.get_run(version.run_id).data.params
        forest = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    except (MlflowException, OSError) as exc:
        log.info("No %s@%s in registry yet (%s)", model_name, alias, type(exc).__name__)
        return None

    config = DetectorConfig(
        mode=params.get("mode", "zscore"),
        z_threshold=float(params.get("z_threshold", 6.0)),
        if_threshold=float(params.get("if_threshold", 0.0)),
        history=int(params.get("history", 180)),
        min_history=int(params.get("min_history", 30)),
        model_version=str(version.version),
    )
    return config, forest

import numpy as np
import pytest

pytest.importorskip("mlflow")

from streamsentinel.detector import StreamingDetector  # noqa: E402
from streamsentinel.registry import load_champion  # noqa: E402
from streamsentinel.training.evaluate import (  # noqa: E402
    Dataset,
    build_dataset,
    classification_metrics,
    select_thresholds,
)
from streamsentinel.training.train import main  # noqa: E402

TINY = ["--train-hours", "0.5", "--val-hours", "0.5", "--test-hours", "0.5", "--n-estimators", "20"]


def test_classification_metrics():
    pred = np.array([True, True, False, False])
    y = np.array([True, False, True, False])
    m = classification_metrics(pred, y, hours=2.0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["false_alarms_per_hour"] == 0.5


def test_threshold_selection_prefers_best_f1():
    z = np.array([[1.0], [2.0], [9.0], [10.0]])
    data = Dataset(z=z, labels=np.array([None, None, "pump", "pump"], dtype=object), hours=1)
    best = select_thresholds(data, if_scores=np.zeros(4))
    assert best["zscore"]["val"]["f1"] == 1.0
    assert 2.0 < best["zscore"]["z_threshold"] <= 9.0


def test_dataset_excludes_warmup_windows():
    data = build_dataset(hours=0.2, seed=1, min_history=10)
    windows_per_symbol = int(0.2 * 3600 / 10)
    assert len(data.z) == 3 * (windows_per_symbol - 10)
    assert data.z.shape[1] == 5


def test_end_to_end_train_register_and_load(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    out = main([*TINY, "--tracking-uri", uri, "--model-name", "test-detector"])
    assert out["registry"]["promoted"]

    loaded = load_champion(uri, "test-detector", "champion")
    assert loaded is not None
    config, forest = loaded
    assert config.mode == out["chosen"]
    assert config.model_version == out["registry"]["version"]

    # The registered model plugs straight into the streaming detector.
    detector = StreamingDetector(config, forest)
    assert detector.iforest_score(np.zeros(5)) == pytest.approx(
        -forest.decision_function(np.zeros((1, 5)))[0]
    )

    # A second, worse model is registered but does not replace the champion.
    worse = main(
        [
            *TINY,
            "--tracking-uri",
            uri,
            "--model-name",
            "test-detector",
            "--seed",
            "3",
            "--n-estimators",
            "2",
            "--anomaly-rate",
            "0.001",
        ]
    )
    current = load_champion(uri, "test-detector", "champion")[0].model_version
    expected = (
        worse["registry"]["version"] if worse["registry"]["promoted"] else config.model_version
    )
    assert current == expected


def test_missing_model_returns_none(tmp_path):
    assert load_champion(f"sqlite:///{tmp_path / 'empty.db'}", "nope", "champion") is None

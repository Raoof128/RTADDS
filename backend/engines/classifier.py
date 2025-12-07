"""Lightweight deepfake classifier stub using Keras."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = Path("models/deepfake_classifier.keras")
_MODEL_CACHE: Dict[int, Any] = {}


def _import_tf() -> tuple[Any, Any, Any]:
    """Import TensorFlow lazily to avoid heavy startup costs during testing.

    Raises:
        RuntimeError: If TensorFlow is not available in the runtime environment.
    """

    try:  # Local import keeps module import cheap for unit tests.
        import tensorflow as tf  # type: ignore
        from tensorflow.keras import layers, models  # type: ignore
    except Exception as exc:  # noqa: BLE001 - want to wrap any loading failure
        raise RuntimeError(
            "TensorFlow is required for ML classification; install tensorflow-cpu"
        ) from exc

    return tf, layers, models


def build_model(input_dim: int) -> Any:
    """Construct a small MLP classifier."""

    _, layers, models = _import_tf()
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_or_load(input_dim: int) -> Any:
    """Load model from disk or train quickly on synthetic data."""

    tf, _, _ = _import_tf()
    if MODEL_PATH.exists():
        logger.info("Loading classifier from %s", MODEL_PATH)
        return tf.keras.models.load_model(MODEL_PATH)

    logger.info("Training lightweight classifier on synthetic data")
    model = build_model(input_dim)
    rng = np.random.default_rng(42)
    x_real = rng.normal(0.0, 1.0, size=(128, input_dim))
    x_fake = rng.normal(0.3, 0.5, size=(128, input_dim))
    X = np.vstack([x_real, x_fake]).astype(np.float32)
    y = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_fake))])
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Saved classifier to %s", MODEL_PATH)
    return model


def _get_model(input_dim: int) -> Any:
    """Return a cached classifier instance for the requested input dimension."""

    if input_dim not in _MODEL_CACHE:
        _MODEL_CACHE[input_dim] = train_or_load(input_dim)
    return _MODEL_CACHE[input_dim]


def _heuristic_probability(feature_vector: np.ndarray) -> float:
    """Fallback probability estimation when TensorFlow is unavailable."""

    normalized_mean = np.clip((float(np.mean(feature_vector)) + 1.5) / 3, 0, 1)
    normalized_variance = np.clip(float(np.var(feature_vector)), 0, 1)
    return float(np.clip(0.4 * normalized_mean + 0.6 * normalized_variance, 0, 1))


def predict_deepfake(feature_vector: np.ndarray) -> Tuple[float, str]:
    """Predict deepfake probability and label.

    Falls back to a deterministic heuristic if TensorFlow is missing to keep the
    educational demo operable in constrained environments.
    """

    try:
        model = _get_model(len(feature_vector))
        probs = model.predict(feature_vector[np.newaxis, :], verbose=0)[0]
        deepfake_prob = float(probs[1])
    except Exception as exc:  # noqa: BLE001 - converting to heuristic fallback
        logger.warning("TensorFlow classifier unavailable; using heuristic: %s", exc)
        deepfake_prob = _heuristic_probability(feature_vector)

    label = "FAKE" if deepfake_prob > 0.6 else "REAL" if deepfake_prob < 0.4 else "SUSPECT"
    score = float(deepfake_prob * 100)
    logger.debug("Classifier deepfake probability %.2f", deepfake_prob)
    return score, label

import numpy as np
import torch

from reentry_mpc.phase9 import (
    MLPResidualModel,
    evaluate_predictions,
    fit_normalizer,
    normalize_X,
    normalize_y,
    predict,
)


def test_phase9_model_forward_shape():
    model = MLPResidualModel(input_dim=9, output_dim=3, hidden_sizes=[8])
    X = torch.zeros((5, 9), dtype=torch.float32)
    y = model(X)
    assert tuple(y.shape) == (5, 3)


def test_phase9_normalization_round_trip_prediction_shape():
    X = np.ones((6, 9), dtype=np.float32)
    y = np.zeros((6, 3), dtype=np.float32)
    normalizer = fit_normalizer(X, y)
    assert normalize_X(X, normalizer).shape == X.shape
    assert normalize_y(y, normalizer).shape == y.shape
    model = MLPResidualModel(input_dim=9, output_dim=3, hidden_sizes=[8])
    prediction = predict(model, X, normalizer)
    assert prediction.shape == y.shape


def test_phase9_zero_baseline_metrics_are_finite():
    y_true = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    y_pred = np.zeros_like(y_true)
    metrics = evaluate_predictions(y_true, y_pred, "zero_residual")
    assert np.isfinite(metrics["mse"])
    assert metrics["mse"] > 0.0

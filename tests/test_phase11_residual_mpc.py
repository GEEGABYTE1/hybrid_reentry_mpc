import casadi as ca
import numpy as np

from reentry_mpc.longitudinal import load_phase1_config
from reentry_mpc.nmpc import nmpc_derivatives_numeric
from reentry_mpc.phase2 import build_reference_profile, load_phase2_config
from reentry_mpc.residual_mpc import (
    fit_residual_surrogate,
    predict_residual_qdot_numpy,
    predict_residual_qdot_symbolic,
    residual_augmented_derivatives_numeric,
)


def test_phase11_surrogate_fit_is_deterministic():
    first = fit_residual_surrogate(
        dataset_dir="outputs/phase8_residual_dataset",
        ridge_lambda=1.0e-6,
        feature_mode="quadratic",
    )
    second = fit_residual_surrogate(
        dataset_dir="outputs/phase8_residual_dataset",
        ridge_lambda=1.0e-6,
        feature_mode="quadratic",
    )
    np.testing.assert_allclose(first.coefficients, second.coefficients)


def test_phase11_surrogate_prediction_shape():
    surrogate = fit_residual_surrogate(
        dataset_dir="outputs/phase8_residual_dataset",
        ridge_lambda=1.0e-6,
        feature_mode="quadratic",
    )
    features = np.zeros((3, 9), dtype=float)
    prediction = predict_residual_qdot_numpy(features=features, surrogate=surrogate)
    assert prediction.shape == (3,)


def test_phase11_symbolic_residual_can_be_evaluated():
    surrogate = fit_residual_surrogate(
        dataset_dir="outputs/phase8_residual_dataset",
        ridge_lambda=1.0e-6,
        feature_mode="quadratic",
    )
    profile = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    row = profile.iloc[0]
    state = ca.MX.sym("x", 3)
    control = ca.MX.sym("u")
    expression = predict_residual_qdot_symbolic(
        state=state, delta_flap_rad=control, row=row, surrogate=surrogate
    )
    fn = ca.Function("residual", [state, control], [expression])
    value = fn(np.array([0.1, 0.0, 0.0]), 0.0)
    assert np.isfinite(float(value))


def test_phase11_zero_gain_matches_nominal_dynamics():
    surrogate = fit_residual_surrogate(
        dataset_dir="outputs/phase8_residual_dataset",
        ridge_lambda=1.0e-6,
        feature_mode="quadratic",
    )
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    profile = build_reference_profile(
        load_phase2_config("configs/phase2_reference.yaml")
    )
    row = profile.iloc[3]
    state = np.array([0.15, 0.01, 0.04])
    control = 0.02
    nominal = nmpc_derivatives_numeric(
        state=state,
        delta_flap_rad=control,
        row=row,
        vehicle=plant.vehicle,
        aero=plant.aero,
    )
    augmented = residual_augmented_derivatives_numeric(
        state=state,
        delta_flap_rad=control,
        row=row,
        vehicle=plant.vehicle,
        aero=plant.aero,
        surrogate=surrogate,
        residual_gain=0.0,
    )
    np.testing.assert_allclose(nominal, augmented)

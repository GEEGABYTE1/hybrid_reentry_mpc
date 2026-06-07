import numpy as np
import pandas as pd

from reentry_mpc.longitudinal import AeroParams, load_phase1_config
from reentry_mpc.phase10 import aero_with_equivalent_qdot_bias, load_residual_model


def test_phase10_loads_residual_model_checkpoint():
    loaded = load_residual_model(
        "outputs/phase9_residual_model/residual_model_checkpoint.pt"
    )
    assert loaded.normalizer["x_mean"].shape[0] == 9
    assert loaded.normalizer["y_mean"].shape[0] == 3


def test_phase10_equivalent_qdot_bias_changes_cm0():
    plant = load_phase1_config("configs/phase1_open_loop.yaml")
    row = pd.Series({"dynamic_pressure_pa": 5000.0})
    corrected = aero_with_equivalent_qdot_bias(
        aero=plant.aero,
        residual_q_dot=0.01,
        row=row,
        plant_config=plant,
        gain=1.0,
    )
    assert isinstance(corrected, AeroParams)
    assert not np.isclose(corrected.cm0, plant.aero.cm0)

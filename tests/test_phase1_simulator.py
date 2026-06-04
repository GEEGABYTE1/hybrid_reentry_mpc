import numpy as np

from reentry_mpc.atmosphere import dynamic_pressure_pa, standard_atmosphere
from reentry_mpc.longitudinal import (
    flap_effectiveness,
    load_phase1_config,
    longitudinal_derivatives,
    scheduled_pitching_moment,
    simulate_open_loop,
)


def test_dynamic_pressure_matches_definition() -> None:
    density = 0.02
    velocity = 2500.0

    assert dynamic_pressure_pa(velocity, density) == 62500.0


def test_standard_atmosphere_density_decreases_with_altitude() -> None:
    sea_level = standard_atmosphere(0.0)
    high_altitude = standard_atmosphere(50000.0)

    assert high_altitude.density_kgm3 < sea_level.density_kgm3
    assert high_altitude.speed_of_sound_mps > 0.0


def test_open_loop_state_propagation_is_finite() -> None:
    config = load_phase1_config("configs/phase1_open_loop.yaml")
    trajectory = simulate_open_loop(config)

    assert not trajectory.empty
    assert np.isfinite(trajectory[["alpha_rad", "q_radps", "theta_rad"]]).all().all()
    assert trajectory["time_s"].is_monotonic_increasing


def test_alpha_derivative_uses_pitch_rate_not_pitch_acceleration() -> None:
    config = load_phase1_config("configs/phase1_open_loop.yaml")
    low_q_state = np.array([0.08, 0.0, 0.0])
    high_q_state = np.array([0.08, 0.025, 0.0])

    low_q_derivative = longitudinal_derivatives(
        time_s=4.0,
        state=low_q_state,
        delta_flap_rad=0.0,
        config=config,
    )
    high_q_derivative = longitudinal_derivatives(
        time_s=4.0,
        state=high_q_state,
        delta_flap_rad=0.0,
        config=config,
    )

    assert np.isclose(high_q_derivative[0] - low_q_derivative[0], 0.025)


def test_pitching_moment_sign_conventions() -> None:
    config = load_phase1_config("configs/phase1_open_loop.yaml")
    schedule = config.reference.sample(6.0)
    neutral_state = np.array([0.0, 0.0, 0.0])
    positive_alpha_state = np.array([0.08, 0.0, 0.0])

    neutral_moment, _neutral_cm, _neutral_effectiveness = scheduled_pitching_moment(
        state=neutral_state,
        delta_flap_rad=0.0,
        schedule=schedule,
        vehicle=config.vehicle,
        aero=config.aero,
    )
    alpha_moment, _alpha_cm, _alpha_effectiveness = scheduled_pitching_moment(
        state=positive_alpha_state,
        delta_flap_rad=0.0,
        schedule=schedule,
        vehicle=config.vehicle,
        aero=config.aero,
    )
    flap_moment, _flap_cm, _flap_effectiveness = scheduled_pitching_moment(
        state=neutral_state,
        delta_flap_rad=0.04,
        schedule=schedule,
        vehicle=config.vehicle,
        aero=config.aero,
    )

    assert alpha_moment < neutral_moment
    assert flap_moment < neutral_moment


def test_flap_effectiveness_decreases_at_high_mach_and_altitude() -> None:
    config = load_phase1_config("configs/phase1_open_loop.yaml")

    low_energy = flap_effectiveness(mach=8.0, altitude_m=35000.0, aero=config.aero)
    high_energy = flap_effectiveness(mach=28.0, altitude_m=80000.0, aero=config.aero)

    assert config.aero.min_effectiveness <= high_energy <= config.aero.max_effectiveness
    assert config.aero.min_effectiveness <= low_energy <= config.aero.max_effectiveness
    assert high_energy < low_energy

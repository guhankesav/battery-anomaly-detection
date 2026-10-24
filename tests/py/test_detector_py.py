import ctypes

import numpy as np
import pytest

from bad import detector
from bad.evaluate import ocv_table
from bad.sim import FleetConfig, load_model, simulate_batch

MODEL = load_model()
SHORT = FleetConfig(cells=24, minutes=120)


def cfg():
    return detector.default_config(ocv_table=ocv_table(MODEL), capacity_ah=1.6)


def test_python_and_c_agree_on_the_config_layout():
    assert detector.LIB.cm_config_size() == ctypes.sizeof(detector.Config)


def test_simulation_is_reproducible():
    a = simulate_batch(MODEL, SHORT, 2, "isc", np.random.default_rng(3))
    b = simulate_batch(MODEL, SHORT, 2, "isc", np.random.default_rng(3))
    assert np.array_equal(a["v"], b["v"]) and np.array_equal(a["target"], b["target"])


def test_internal_short_drains_its_cell_against_peers():
    b = simulate_batch(MODEL, SHORT, 1, "isc", np.random.default_rng(4))
    v, tgt, on = b["v"][0], b["target"][0], b["onset"][0]
    gap_before = v[on, tgt] - np.median(v[on])
    gap_end = v[-1, tgt] - np.median(v[-1])
    assert gap_end < gap_before


def test_stuck_sensor_holds_its_value():
    b = simulate_batch(MODEL, SHORT, 1, "sensor_stuck", np.random.default_rng(5))
    v, tgt, on = b["v"][0], b["target"][0], b["onset"][0]
    assert np.all(v[on:, tgt] == v[on, tgt]) and np.ptp(np.median(v[on:], axis=1)) > 0.02


@pytest.mark.parametrize("fault, flag", [("sensor_stuck", 4), ("sensor_spike", 8), ("thermal", 2)])
def test_detector_flags_the_faulted_cell_only(fault, flag):
    b = simulate_batch(MODEL, SHORT, 1, fault, np.random.default_rng(11))
    first, flags = detector.run(cfg(), b["v"][0], b["t"][0], b["current"][0])
    tgt = b["target"][0]
    assert flags[tgt] & flag
    assert first[tgt] >= b["onset"][0]
    assert not np.delete(flags, tgt).any()


def test_healthy_packs_raise_nothing():
    b = simulate_batch(MODEL, SHORT, 4, None, np.random.default_rng(21))
    for p in range(4):
        _, flags = detector.run(cfg(), b["v"][p], b["t"][p], b["current"][p])
        assert not flags.any()

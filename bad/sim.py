"""Fleet simulator: series packs of NASA-calibrated cells driving a shared load, with injected faults.

Each pack gets an aging state, an ambient temperature, cell-to-cell manufacturing spread, and
BMS-grade sensor noise. Packs are simulated in batches (arrays of shape [packs, cells]) so a
fleet of thousands runs in minutes.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAULTS = ["isc", "high_resistance", "thermal", "sensor_stuck", "sensor_spike", "sensor_drift"]
EXPECTED_FLAG = {"isc": 1 | 32, "high_resistance": 16, "thermal": 2, "sensor_stuck": 4, "sensor_spike": 8, "sensor_drift": 1 | 32}


@dataclass
class FleetConfig:
    cells: int = 96
    minutes: int = 120
    dt_s: float = 1.0
    capacity_spread: float = 0.035      # NASA cells: ~3.5% capacity spread at equal age
    resistance_spread: float = 0.05
    soc_spread: float = 0.01
    v_noise: float = 0.001              # BMS-grade: 1 mV (NASA lab instruments: 0.09 mV)
    v_lsb: float = 0.001
    t_noise: float = 0.2
    t_offset_sd: float = 0.3            # thermistor tolerance
    t_lsb: float = 0.1
    heat_capacity_j_per_k: float = 45.0  # 18650, ~46 g
    ambient_range: tuple = (-5.0, 40.0)
    life_range: tuple = (0.0, 1.0)      # fraction of cycle life used (NASA aging curves)


def load_model(path: str = "docs/cell_model.json") -> dict:
    return json.loads(Path(path).read_text())


def drive_profile(rng: np.random.Generator, steps: int, dt: float) -> np.ndarray:
    """Rest, drive, rest, drive, rest; driving current is an AR(1) process with acceleration peaks and regen."""
    i = np.zeros(steps)
    m = int(60 / dt)
    segments = [(10 * m, False), (50 * m, True), (15 * m, False), (35 * m, True), (steps, False)]
    k, level = 0, 0.8
    for length, driving in segments:
        end = min(steps, k + length)
        for j in range(k, end):
            if driving:
                level = 0.8 + 0.97 * (level - 0.8) + rng.normal(0, 0.25)
                i[j] = np.clip(level + (rng.random() < 0.01) * rng.uniform(1, 2), -1.5, 4.0)  # <= 2C peaks
        k = end
        if k >= steps:
            break
    return i


def simulate_batch(model: dict, cfg: FleetConfig, packs: int, fault: str | None, rng: np.random.Generator) -> dict:
    n, steps, dt = cfg.cells, int(cfg.minutes * 60 / cfg.dt_s), cfg.dt_s
    grid, ocv = np.array(model["ocv_soc_grid"]), np.array(model["ocv_v"])
    aging = model["aging"]
    life = rng.uniform(*cfg.life_range, packs)
    cap0 = np.interp(life, aging["life_fraction"], aging["capacity_ah"])
    r0 = np.interp(life, aging["life_fraction"], aging["resistance_ohm"])
    cap = cap0[:, None] * rng.normal(1, cfg.capacity_spread, (packs, n))
    r25 = r0[:, None] * rng.normal(1, cfg.resistance_spread, (packs, n))
    r_th = model["thermal_resistance_k_per_w"] * rng.normal(1, 0.1, (packs, n))
    soc = rng.uniform(0.8, 1.0, packs)[:, None] + rng.normal(0, cfg.soc_spread, (packs, n))
    ambient = rng.uniform(*cfg.ambient_range, packs)
    temp = ambient[:, None] + rng.normal(0, 0.2, (packs, n))
    t_offset = rng.normal(0, cfg.t_offset_sd, (packs, n))
    v_rc = np.zeros((packs, n))
    current = np.stack([drive_profile(rng, steps, dt) for _ in range(packs)])

    target = rng.integers(0, n, packs)
    onset = rng.integers(int(20 * 60 / dt), int(80 * 60 / dt), packs)
    rows = np.arange(packs)
    severity = np.zeros(packs)
    i_sc_r = np.full(packs, np.inf)
    extra_r = np.zeros(packs)
    extra_heat = np.zeros(packs)
    if fault == "isc":
        severity = np.exp(rng.uniform(np.log(10), np.log(300), packs))  # short resistance, ohm
        i_sc_r = severity
    elif fault == "high_resistance":
        severity = extra_r = rng.uniform(0.02, 0.08, packs)
    elif fault == "thermal":
        severity = extra_heat = rng.uniform(0.1, 0.6, packs)
    elif fault == "sensor_drift":
        severity = rng.uniform(0.5, 5.0, packs) / 1000 / 60  # V per second (0.5-5 mV/min)
    elif fault == "sensor_spike":
        severity = rng.uniform(0.2, 0.6, packs)

    v_out = np.empty((packs, steps, n), np.float32)
    t_out = np.empty((packs, steps, n), np.float32)
    stuck_value = np.zeros(packs)
    cutoff = np.zeros(packs, bool)
    for k in range(steps):
        active = k >= onset
        current[cutoff, k] = 0.0  # the BMS stops discharge once any cell reaches cutoff
        i = current[:, k][:, None]
        r = r25 * np.exp(3000 * (1 / (temp + 273.15) - 1 / 298.15))
        r_cell = r.copy()
        r_cell[rows, target] += np.where(active, extra_r, 0)
        v_ocv = np.interp(np.clip(soc, 0, 1), grid, ocv)
        v = v_ocv - i * r_cell - v_rc
        i_sc = np.zeros((packs, n))
        i_sc[rows, target] = np.where(active, v[rows, target] / i_sc_r, 0)
        heat = i * i * r_cell + i_sc * v
        heat[rows, target] += np.where(active, extra_heat, 0)
        soc = soc - (i + i_sc) * dt / 3600 / cap
        v_rc += dt / 30.0 * (i * 0.4 * r - v_rc)
        temp += dt / cfg.heat_capacity_j_per_k * (heat - (temp - ambient[:, None]) / r_th)

        vm = np.round((v + rng.normal(0, cfg.v_noise, v.shape)) / cfg.v_lsb) * cfg.v_lsb
        tm = np.round((temp + t_offset + rng.normal(0, cfg.t_noise, temp.shape)) / cfg.t_lsb) * cfg.t_lsb
        if fault == "sensor_stuck":
            stuck_value = np.where(k == onset, vm[rows, target], stuck_value)
            vm[rows, target] = np.where(active, stuck_value, vm[rows, target])
        elif fault == "sensor_drift":
            vm[rows, target] -= np.where(active, (k - onset) * dt * severity, 0)
        elif fault == "sensor_spike":
            glitch = active & (rng.random(packs) < 0.003)
            vm[rows, target] += np.where(glitch, severity * rng.choice([-1, 1], packs), 0)
        cutoff |= soc.min(axis=1) < 0.03  # empty, not voltage sag under a cold load
        v_out[:, k] = vm
        t_out[:, k] = tm
    return {"v": v_out, "t": t_out, "current": current.astype(np.float32), "fault": fault, "target": target,
            "onset": onset, "severity": severity, "life": life, "ambient": ambient, "steps": steps}

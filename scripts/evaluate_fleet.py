"""Final evaluation on a fleet never used for tuning (thresholds were chosen on seed 100).

Writes docs/results.json: false alarms on healthy packs, detection per fault type, detection versus
fault severity (logistic regression), stress tests, and detector cost on this machine.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from bad import detector
from bad.evaluate import ocv_table, run_fleet
from bad.sim import FleetConfig, load_model, simulate_batch

SEED = 2027  # v1 was scored on 2026 (docs/results_v1_seed2026.json); its stress test led to v2's changes
SEVERITY_UNITS = {"isc": ("short resistance", "ohm", True), "high_resistance": ("extra resistance", "ohm", False),
                  "thermal": ("extra heat", "W", False)}


def severity_curve(results, fault: str) -> dict:
    """P(detect) vs log severity. For a short, lower resistance is more severe."""
    rs = [r for r in results if r.fault == fault]
    x = np.log([r.severity for r in rs]).reshape(-1, 1)
    y = np.array([bool(r.target_flags) for r in rs])
    model = LogisticRegression().fit(x, y)
    a, b = model.intercept_[0], model.coef_[0][0]
    at = {p: float(np.exp((np.log(p / (1 - p)) - a) / b)) for p in (0.5, 0.9)}
    name, unit, lower_is_worse = SEVERITY_UNITS[fault]
    return {"quantity": f"{name} ({unit})", "severity_for_50pct_detection": at[0.5],
            "severity_for_90pct_detection": at[0.9], "more_severe_is": "lower" if lower_is_worse else "higher",
            "range_simulated": [float(np.min(np.exp(x))), float(np.max(np.exp(x)))]}


def main() -> None:
    model = load_model()
    cfg = detector.default_config(ocv_table=ocv_table(model), capacity_ah=1.6)
    out: dict = {"seed": SEED, "tuned_on_seed": 100, "detector_state_bytes_128_cells": detector.state_bytes()}

    t0 = time.time()
    packs = {None: 1000, "isc": 200, "high_resistance": 200, "thermal": 200,
             "sensor_stuck": 100, "sensor_spike": 100, "sensor_drift": 100}
    score = run_fleet({"final": cfg}, packs, seed=SEED, model=model)["final"]
    out["test_fleet"] = score.summary(1.0, 120)
    out["severity"] = {f: severity_curve(score.results, f) for f in SEVERITY_UNITS}
    out["test_fleet_seconds"] = round(time.time() - t0, 1)

    stress = {
        "baseline": FleetConfig(),
        "sensor_noise_x2": FleetConfig(v_noise=0.002, t_noise=0.4),
        "manufacturing_spread_x2": FleetConfig(capacity_spread=0.07, resistance_spread=0.10),
        "cold_-5_to_5C": FleetConfig(ambient_range=(-5.0, 5.0)),
        "end_of_life_80_100pct": FleetConfig(life_range=(0.8, 1.0)),
    }
    out["stress"] = {}
    for name, fleet in stress.items():
        sc = run_fleet({"final": cfg}, {None: 300, "isc": 100, "thermal": 100}, seed=SEED + 1, fleet=fleet,
                       model=model)["final"]
        out["stress"][name] = sc.summary(1.0, 120)
        print(name, json.dumps({k: (v.get("false_alarms_per_1000_pack_hours", v.get("detected"))) for k, v in out["stress"][name].items()}), flush=True)

    # Cost of the C detector: one 2-hour, 96-cell pack recording.
    b = simulate_batch(model, FleetConfig(), 1, None, np.random.default_rng(1))
    reps = 20
    t1 = time.perf_counter()
    for _ in range(reps):
        detector.run(cfg, b["v"][0], b["t"][0], b["current"][0])
    per_step = (time.perf_counter() - t1) / reps / b["steps"]
    out["host_cost"] = {"microseconds_per_pack_step_96_cells": round(per_step * 1e6, 2),
                        "nanoseconds_per_cell_step": round(per_step / 96 * 1e9, 1)}
    Path("docs/results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stress"}, indent=2))


if __name__ == "__main__":
    main()

"""Runs the C detector over simulated fleets and scores it against the injected ground truth.

Several detector configurations can be scored on the same simulated packs in one pass, which is
how thresholds are tuned on one fleet (seed) and then reported on a separate one.
"""
from dataclasses import dataclass, field

import numpy as np

from bad import detector
from bad.sim import EXPECTED_FLAG, FAULTS, FleetConfig, load_model, simulate_batch


@dataclass
class PackResult:
    fault: str | None
    target: int
    onset: int
    severity: float
    life: float
    ambient: float
    alarmed_cells: list
    target_flags: int
    target_first: int
    other_flags: int


@dataclass
class Score:
    results: list = field(default_factory=list)

    def summary(self, dt_s: float, minutes: int) -> dict:
        out = {}
        normal = [r for r in self.results if r.fault is None]
        if normal:
            hours = len(normal) * minutes / 60
            fa = sum(1 for r in normal if r.alarmed_cells)
            out["normal"] = {"packs": len(normal), "pack_hours": hours, "packs_with_false_alarm": fa,
                             "false_alarms_per_1000_pack_hours": 1000 * fa / hours}
        for f in FAULTS:
            rs = [r for r in self.results if r.fault == f]
            if not rs:
                continue
            hit = [r for r in rs if r.target_flags]
            delays = [(r.target_first - r.onset) * dt_s / 60 for r in hit if r.target_first >= r.onset]
            out[f] = {"packs": len(rs), "detected": len(hit) / len(rs),
                      "right_type": sum(1 for r in hit if r.target_flags & EXPECTED_FLAG[f]) / len(rs),
                      "other_cells_flagged": sum(1 for r in rs if r.other_flags) / len(rs),
                      "median_delay_min": float(np.median(delays)) if delays else None}
        return out


def score_pack(batch: dict, p: int, first: np.ndarray, flags: np.ndarray) -> PackResult:
    t = int(batch["target"][p])
    others = np.delete(flags, t)
    return PackResult(batch["fault"], t, int(batch["onset"][p]), float(batch["severity"][p]), float(batch["life"][p]),
                      float(batch["ambient"][p]), np.flatnonzero(flags).tolist(),
                      int(flags[t]) if batch["fault"] else 0, int(first[t]) if batch["fault"] else -1,
                      int(np.bitwise_or.reduce(others)) if len(others) else 0)


def run_fleet(configs: dict, packs: dict, seed: int, fleet: FleetConfig | None = None, batch_size: int = 50,
              model: dict | None = None) -> dict:
    """configs: name -> detector.Config. packs: fault (None for healthy) -> number of packs."""
    fleet = fleet or FleetConfig()
    model = model or load_model()
    rng = np.random.default_rng(seed)
    scores = {name: Score() for name in configs}
    for fault, count in packs.items():
        for start in range(0, count, batch_size):
            b = simulate_batch(model, fleet, min(batch_size, count - start), fault, rng)
            if fault is None:
                b["target"][:] = -1
            for p in range(len(b["life"])):
                for name, cfg in configs.items():
                    first, flags = detector.run(cfg, b["v"][p], b["t"][p], b["current"][p])
                    if fault is None:
                        r = PackResult(None, -1, -1, 0.0, float(b["life"][p]), float(b["ambient"][p]),
                                       np.flatnonzero(flags).tolist(), 0, -1, int(np.bitwise_or.reduce(flags)))
                    else:
                        r = score_pack(b, p, first, flags)
                    scores[name].results.append(r)
    return scores


def ocv_table(model: dict) -> list[float]:
    return list(np.interp(np.linspace(0, 1, 21), model["ocv_soc_grid"], model["ocv_v"]))

"""Cell parameters measured from the NASA PCoE Li-ion aging data (18650 cells, 2 A discharges, 24 degC).

Everything the fleet simulator assumes about a cell comes from here: the voltage curve, how capacity
and resistance change with age, how noisy the voltage and temperature sensors are, and how fast a
cell heats.
"""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

RAW = Path("data/raw/fy08")
CELLS = ["B0005", "B0006", "B0007", "B0018"]


def load(cell: str) -> dict:
    cycles = loadmat(RAW / f"{cell}.mat", simplify_cells=True)[cell]["cycle"]
    discharges, impedance = [], []
    for c in cycles:
        d = c["data"]
        if c["type"] == "discharge":
            discharges.append({k: np.asarray(d[k], dtype=float) for k in
                               ("Time", "Voltage_measured", "Current_measured", "Temperature_measured")}
                              | {"capacity": float(d["Capacity"])})
        elif c["type"] == "impedance":
            impedance.append(float(np.real(d["Re"])) + float(np.real(d["Rct"])))
    return {"discharges": discharges, "resistance": np.array(impedance)}


def robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def fit() -> dict:
    data = {c: load(c) for c in CELLS}
    life, capacity, resistance = [], [], []
    v_noise, t_noise, r_th = [], [], []
    for cell, d in data.items():
        dis, res = d["discharges"], d["resistance"]
        res = res[(res > 0.05) & (res < 0.5)]
        for k, cyc in enumerate(dis):
            frac = k / (len(dis) - 1)
            life.append(frac)
            capacity.append(cyc["capacity"])
            resistance.append(float(np.interp(frac, np.linspace(0, 1, len(res)), res)))
            v, temp = cyc["Voltage_measured"], cyc["Temperature_measured"]
            mid = slice(len(v) // 5, 4 * len(v) // 5)  # avoid the start step and the end knee
            if mid.stop - mid.start > 15:
                v_noise.append(robust_sigma(v[mid] - savgol_filter(v[mid], 9, 2)))
                t_noise.append(robust_sigma(temp[mid] - savgol_filter(temp[mid], 9, 2)))
            i = -np.median(cyc["Current_measured"][mid])
            heat = i * i * resistance[-1]
            rise = temp.max() - temp[0]
            if heat > 0 and rise > 0:
                r_th.append(rise / heat)

    # Voltage curve vs state of charge from the first full discharge of B0005, shifted to open circuit.
    first = data["B0005"]["discharges"][0]
    t, v, i = first["Time"], first["Voltage_measured"], -first["Current_measured"]
    on = i > 1.0
    ah = np.concatenate([[0], np.cumsum(np.diff(t) * i[1:] / 3600)])
    soc = 1 - ah[on] / first["capacity"]
    ocv = v[on] + i[on] * data["B0005"]["resistance"][0]
    grid = np.linspace(0, 1, 101)
    order = np.argsort(soc)
    ocv_grid = np.interp(grid, soc[order], np.maximum.accumulate(ocv[order]))

    life, capacity, resistance = map(np.array, (life, capacity, resistance))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(life, bins[1:-1])
    return {
        "source": "NASA Ames PCoE Battery Data Set, cells " + ", ".join(CELLS),
        "nominal_capacity_ah": float(capacity[life == 0].mean()),
        "ocv_soc_grid": grid.round(2).tolist(), "ocv_v": ocv_grid.round(4).tolist(),
        "aging": {"life_fraction": [float(b) for b in (bins[:-1] + bins[1:]) / 2],
                  "capacity_ah": [float(capacity[idx == k].mean()) for k in range(10)],
                  "capacity_sd_ah": [float(capacity[idx == k].std()) for k in range(10)],
                  "resistance_ohm": [float(resistance[idx == k].mean()) for k in range(10)]},
        "sensor_noise": {"voltage_v": float(np.median(v_noise)), "temperature_c": float(np.median(t_noise))},
        "thermal_resistance_k_per_w": float(np.median(r_th)),
        "discharge_cycles": int(len(life)),
    }


if __name__ == "__main__":
    params = fit()
    Path("docs").mkdir(exist_ok=True)
    Path("docs/cell_model.json").write_text(json.dumps(params, indent=2))
    print(json.dumps({k: v for k, v in params.items() if k not in ("ocv_soc_grid", "ocv_v")}, indent=2))
    print("OCV at SOC 0, 0.2, 0.5, 0.8, 1:", [params["ocv_v"][j] for j in (0, 20, 50, 80, 100)])

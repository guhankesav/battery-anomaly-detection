# Battery Cell Anomaly Detection

Real-time detection of failing cells in a series battery pack, written in C the way BMS firmware is
written: no heap, no recursion, fixed-size state, one call per 1 Hz sample. It catches internal
shorts, abnormal heating, rising resistance, and stuck, spiking or drifting sensors, and it is
evaluated on simulated fleets whose cells are calibrated from real aging data
(**NASA Ames PCoE Li-ion data set**, 636 discharge cycles of four 18650 cells).

```
NASA aging data ──► cell model (OCV curve, capacity and resistance vs age, heating) ──► fleet simulator
                                                                                        │ 96-cell packs, drive cycles,
                                                                                        │ manufacturing spread, sensor noise,
                                                                                        ▼ injected faults (ground truth)
                              src/cellmon.c  (same code a microcontroller would run, called through ctypes)
                                                                                        │
                                                                                        ▼
                                      detection rate, delay, false alarms per 1,000 pack-hours, stress tests
```

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
make && make test                 # C library and unit tests (make test-sanitize: ASan + UBSan)
scripts/download_nasa.sh          # ~210 MB, only needed to refit the cell model
.venv/bin/python -m bad.nasa      # writes docs/cell_model.json
PYTHONPATH=. .venv/bin/python scripts/evaluate_fleet.py   # docs/results.json, ~4 min
.venv/bin/python -m pytest
make arm                          # Cortex-M4F build and footprint (arm-none-eabi-gcc)
```

## How each fault is detected

Every check compares a cell with the **median of its pack**, so load, state of charge and ambient
temperature cancel; what is left is modeled per cell.

| Fault | Signal | Method |
|---|---|---|
| Internal short | Cell drains against its peers **while the pack rests** | After polarization settles, an online regression of the cell's residual voltage on time gives a slope; through the OCV curve's slope and the cell capacity it becomes a leakage current in amps, alarmed above 15 mA at 5 standard errors |
| Abnormal heating | Hotter than peers beyond what its own heating explains | Per-cell RLS of temperature residual on pack I² filtered at the cell's thermal time constant; CUSUM on the excess over a 1.75 °C margin, with noise estimated from sample differences |
| Rising resistance | Larger voltage step than peers when current steps | Pulse resistance: ΔV/ΔI across current steps (slow effects cancel), Arrhenius-corrected to 25 °C, alarmed above max(25 mΩ, pack median + 5 robust spreads) |
| Stuck sensor | Reading frozen while the pack moves | Unchanged for 120 s while the pack median moved 20 mV |
| Spike | Jump in model error that comes straight back | Reversal within 5 samples, ignored during current steps |
| Sensor drift | Same as an internal short in voltage | Caught by the rest-period leakage check (temperature can separate the two: a short heats its cell) |

The firmware needs only what a BMS already has: cell voltages and temperatures, pack current, and
an OCV table (here, 21 points from the NASA curve).

## Results

Thresholds were tuned on fleet seed 100 and reported on seed **2027**, which was never used for
tuning ([`docs/results.json`](docs/results.json)). 1,000 healthy packs and 800 faulted packs, 96 cells
each, 2-hour drive cycles with rests, ambient −5 to 40 °C, random age across the NASA aging range.

| | Detected | Correct type | Other cells flagged | Median time to detect |
|---|---|---|---|---|
| **Healthy packs** | 2 of 1,000 alarmed: **1.0 false alarm per 1,000 pack-hours** | | | |
| Internal short (10–290 Ω) | **94.5%** | 94% | 0% | 6.8 min |
| Abnormal heating (0.1–0.6 W) | **96%** | 96% | 0% | 7.1 min |
| Extra resistance (20–80 mΩ) | 57% | 56.5% | 0.5% | 15.2 min |
| Stuck sensor | 100% | 100% | 0% | 2.0 min |
| Sensor spikes | 100% | 100% | 0% | 3.6 min |
| Sensor drift (0.5–5 mV/min) | 100% | 100% | 1% | 22.5 min |

**Detection versus severity** (logistic regression of detected on log severity, `scripts/evaluate_fleet.py`):

| Fault | 50% detected at | 90% detected at |
|---|---|---|
| Internal short | 375 Ω (≈10 mA leak) | **140 Ω (≈26 mA)** |
| Abnormal heating | 0.07 W | **0.16 W** |
| Extra resistance | 42 mΩ | 94 mΩ (extrapolated past the 80 mΩ simulated) |

Resistance faults below ~40 mΩ sit inside normal cell-to-cell spread (5% of a 0.13–0.16 Ω cell),
which is why the overall rate is 57%.

**Stress tests** (300 healthy + 200 faulted packs each, seed 2028):

| Condition | False alarms per 1,000 pack-hours | Shorts detected | Heating detected |
|---|---|---|---|
| Baseline | 3.3 | 94% | 97% |
| Sensor noise ×2 (2 mV, 0.4 °C) | 3.3 | 90% | 97% |
| Manufacturing spread ×2 (7% capacity, 10% resistance) | 0.0 | 96% | 97% |
| Cold only (−5 to 5 °C) | 5.0 | 94% | 97% |
| End of life (80–100% of cycle life) | 1.7 | 99% | 97% |

**Cost:** 2.6 µs per 96-cell sample on an Apple M-series laptop (27 ns per cell). Built for a
Cortex-M4F with `-Os` in CI, the detector (`cm_step`, `cm_init`, selection, defaults) is **2.9 KB of
code**, and its state is 24 KB of RAM at the 128-cell maximum, all static. (`cm_run_ex`, the
whole-recording harness for Python, adds its own static copy of that state.)

## What did not work, and what changed

- **CUSUM on the voltage residual while driving** flagged 40% of healthy packs at any useful setting:
  capacity mismatch moves cells apart nonlinearly along the OCV curve and the online model lags it.
  It is kept in the code, off by default. Internal shorts are found during rests instead, where
  only a leak can move a cell against its peers, which is how production BMS shorts detection works.
- **Estimating leakage while driving** (regressing on time and on charge throughput together) could not
  separate a short from a smaller cell: over a drive, time and amp-hours rise together.
- **Noise learned from the residual** grew along with slow faults, so they never looked unusual; the
  temperature check now estimates noise from sample-to-sample differences and alarms on a physical margin.
- **v1 was scored on seed 2026** ([`docs/results_v1_seed2026.json`](docs/results_v1_seed2026.json)):
  4.5 false alarms per 1,000 pack-hours, but **373** with manufacturing spread doubled, almost all from
  the fixed resistance threshold. v2 adds the pack-relative threshold and fixes a one-sample ordering bug
  where cells could be checked against the fixed floor on the step their estimates became valid. The
  simulator also gained a BMS discharge cutoff at empty. Because those changes were prompted by seed 2026,
  v2 is reported on a new seed.

## Limits

- Faults are simulated, from a cell model fitted to NASA data. The NASA lab sensors are far quieter
  (0.09 mV) than a BMS front end, so the simulation uses 1 mV and 0.2 °C and stresses at double that.
- One cell chemistry and format, 1 Hz sampling, 2-hour episodes; parked-overnight rests would make
  short detection more sensitive than shown.
- The detector is compiled for a Cortex-M4F in CI but has not run on hardware.

## Tests

C unit tests on synthetic packs with known answers (leak estimate within 8 mA of 50 mA injected, pulse
resistance within 6 mΩ of 60 mΩ, stuck, spike, temperature rise, sensor offset not flagged, healthy pack
silent), run plain and under AddressSanitizer + UndefinedBehaviorSanitizer in CI. Python tests check the
ctypes struct layout against C, simulator reproducibility and physics, and end-to-end detection.

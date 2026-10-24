"""ctypes binding to the C detector, so Python runs exactly the code a microcontroller would."""
import ctypes
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

LIB = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "build" / "libcellmon.so"))


class Config(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "dt_s", "rls_lambda", "noise_alpha", "min_sigma_v", "cusum_k", "cusum_h", "temp_alpha", "min_sigma_t",
        "temp_margin_c", "temp_cusum_h", "stuck_s", "stuck_pack_mv", "spike_v", "max_extra_r_ohm")] + [
        ("warmup_steps", ctypes.c_uint32), ("min_excitation", ctypes.c_float), ("i_tau_s", ctypes.c_float),
        ("r_nominal_ohm", ctypes.c_float), ("ocv_v", ctypes.c_float * 21)] + [
        (name, ctypes.c_float) for name in ("rest_current_a", "rest_settle_s", "rest_min_s", "capacity_ah",
                                            "leak_alarm_a", "leak_z", "r_activation_k", "pulse_min_di_a",
                                            "pulse_alpha")] + [("pulse_min_events", ctypes.c_uint32),
                                                               ("thermal_tau_s", ctypes.c_float),
                                                               ("r_outlier_z", ctypes.c_float)]


LIB.cm_default_config.argtypes = [ctypes.POINTER(Config)]
LIB.cm_state_size.restype = ctypes.c_uint32
LIB.cm_config_size.restype = ctypes.c_uint32
if ctypes.sizeof(Config) != LIB.cm_config_size():
    raise ImportError(f"Config layout mismatch: Python {ctypes.sizeof(Config)} bytes, C {LIB.cm_config_size()}")
F32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
U32 = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
LIB.cm_run.argtypes = [ctypes.POINTER(Config), ctypes.c_uint16, ctypes.c_uint32, F32, F32, F32, U32, U32]
LIB.cm_run_ex.argtypes = LIB.cm_run.argtypes + [F32, F32]

NO_ALARM = np.iinfo(np.uint32).max


def default_config(ocv_table: list[float] | None = None, **overrides) -> Config:
    """ocv_table: 21 open-circuit voltages at SOC 0, 5, ..., 100 %."""
    cfg = Config()
    LIB.cm_default_config(ctypes.byref(cfg))
    if ocv_table is not None:
        for i, v in enumerate(ocv_table):
            cfg.ocv_v[i] = v
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(cfg: Config, v: np.ndarray, t: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v, t: [steps, cells] float32; returns (first alarm step per cell, latched flags per cell)."""
    steps, n = v.shape
    first = np.empty(n, np.uint32)
    flags = np.empty(n, np.uint32)
    LIB.cm_run(ctypes.byref(cfg), n, steps, np.ascontiguousarray(v, np.float32), np.ascontiguousarray(t, np.float32),
               np.ascontiguousarray(current, np.float32), first, flags)
    return first, flags


def state_bytes() -> int:
    return LIB.cm_state_size()


def run_with_estimates(cfg: Config, v: np.ndarray, t: np.ndarray, current: np.ndarray) -> dict:
    """Like run(), plus each cell's final pulse resistance (ohm above median, 25 degC) and leakage (A)."""
    steps, n = v.shape
    out = {k: np.empty(n, np.uint32) for k in ("first", "flags")} | {k: np.empty(n, np.float32) for k in ("pulse_r", "leak_a")}
    LIB.cm_run_ex(ctypes.byref(cfg), n, steps, np.ascontiguousarray(v, np.float32), np.ascontiguousarray(t, np.float32),
                  np.ascontiguousarray(current, np.float32), out["first"], out["flags"], out["pulse_r"], out["leak_a"])
    return out

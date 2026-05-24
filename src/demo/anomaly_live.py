"""
Sync.ai Demo — Live Anomaly Detector

Strategy: rolling z-score against the most-recent 12-row (1-hour) baseline.
A minimum-std floor prevents z-score explosion when the signal is very flat
(e.g. the chiller idling at ~0.55 W before a sudden power surge).

This approach detects:
  - Sudden spikes / drops  (July 14 power event, pressure transients)
  - Rapid temperature departures  (Aug 7 outlet_temp event)

It intentionally does NOT flag gradual diurnal drift (chiller heating up
over the day) — those are normal operating patterns, not faults.

Returns the highest-z anomaly in the current row, or None.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

WATCH_COLS  = ["active_power", "high_pressure_1", "low_pressure_1", "outlet_temp"]
Z_THRESHOLD = 2.5
_BASELINE_ROWS = 12   # 1 hour of 5-min data

# Minimum std floor per sensor — prevents near-zero std from generating
# meaninglessly large z-scores on flat baselines.
_MIN_STD: dict[str, float] = {
    "active_power":    0.5,   # W
    "high_pressure_1": 0.2,   # bar
    "low_pressure_1":  0.2,   # bar
    "outlet_temp":     0.4,   # degC
}

# Root cause map for instant diagnosis
FAILURE_MODE_MAP = {
    "active_power": {
        "failure_mode": "Compressor Stress / Electrical Fault",
        "action":       "Inspect compressor windings and capacitors. Check supply voltage.",
        "fault_code":   "EL-020",
        "priority":     "High",
    },
    "high_pressure_1": {
        "failure_mode": "Refrigerant Overcharge / Condenser Fouling",
        "action":       "Inspect and clean condenser coils. Verify refrigerant charge.",
        "fault_code":   "RF-010",
        "priority":     "High",
    },
    "low_pressure_1": {
        "failure_mode": "Refrigerant Leak / Evaporator Issue",
        "action":       "Perform leak detection. Check filter-drier. Inspect evaporator.",
        "fault_code":   "RF-005",
        "priority":     "Critical",
    },
    "outlet_temp": {
        "failure_mode": "Coil Fouling / Airflow Restriction",
        "action":       "Clean or replace air filters. Clean coil surfaces.",
        "fault_code":   "MT-030",
        "priority":     "Medium",
    },
}

PRIORITY_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}


def _severity(z: float) -> str:
    if z >= 3.5:
        return "Critical"
    if z >= 2.8:
        return "Warning"
    return "Watch"


def detect(window_df: pd.DataFrame) -> dict | None:
    """
    Analyse the latest row in window_df against a 1-hour rolling baseline.
    Returns the highest-z anomaly dict, or None if nothing exceeds threshold.

    Alert dict keys:
        timestamp, sensor, value, z_score, severity,
        failure_mode, action, fault_code, priority
    """
    if len(window_df) < _BASELINE_ROWS + 2:
        return None

    latest = window_df.iloc[-1]
    best: dict | None = None
    best_z = Z_THRESHOLD  # minimum bar

    for col in WATCH_COLS:
        if col not in window_df.columns:
            continue
        series = window_df[col].dropna()
        if len(series) < _BASELINE_ROWS + 2:
            continue

        # 1-hour local baseline (excludes the current row)
        baseline = series.iloc[:-1].tail(_BASELINE_ROWS)
        mu  = float(baseline.mean())
        sig = float(baseline.std())

        # Apply min-std floor
        min_std = _MIN_STD.get(col, 0.1)
        if np.isnan(sig) or sig < min_std:
            sig = min_std

        z = abs((float(latest[col]) - mu) / sig)

        if z > best_z:
            best_z = z
            fm = FAILURE_MODE_MAP.get(col, {
                "failure_mode": "Sensor Anomaly",
                "action":       "Inspect sensor and recent maintenance logs.",
                "fault_code":   "GN-001",
                "priority":     "Medium",
            })
            best = {
                "timestamp":    latest["timestamp"],
                "sensor":       col,
                "value":        round(float(latest[col]), 3),
                "z_score":      round(min(float(z), 15.0), 2),  # cap at 15σ for display
                "severity":     _severity(z),
                "failure_mode": fm["failure_mode"],
                "action":       fm["action"],
                "fault_code":   fm["fault_code"],
                "priority":     fm["priority"],
            }

    return best


def is_duplicate(alert: dict, event_log: list, cooldown_rows: int = 12) -> bool:
    """
    Suppress re-firing the same alert within cooldown_rows ticks
    to avoid event log spam on a sustained anomaly.
    """
    if not event_log:
        return False
    last = event_log[0]
    if last["sensor"] != alert["sensor"]:
        return False
    try:
        delta = abs(
            (pd.Timestamp(alert["timestamp"]) - pd.Timestamp(last["timestamp"])).total_seconds()
        )
        return delta < cooldown_rows * 300  # 300s = 5 min per row
    except Exception:
        return False

"""
Sync.ai Demo — 3-Agent Intelligence Pipeline

Agent 1  Sensor Intelligence  — cross-correlates sensors, explains the anomaly
Agent 2  Diagnosis             — ranks root causes with confidence scores
Agent 3  Action / Work Order   — generates a structured, actionable work order

Each agent uses Claude tool calling. Pipeline runs sequentially;
each agent's output feeds the next. One agent per Streamlit rerun so the UI
shows progressive results without blocking the full pipeline at once.
"""
from __future__ import annotations
import json
import uuid
import os
from datetime import datetime

import numpy as np
import pandas as pd
import anthropic

MODEL = "claude-sonnet-4-6"

try:
    _client = anthropic.Anthropic()
except Exception:
    _client = None  # API key missing — handled gracefully in each agent

# ── Static asset profile ──────────────────────────────────────────────────────
ASSET_PROFILE = {
    "asset_id":    "A1-CHILLER-MAIN",
    "name":        "Chiller Unit A1",
    "site":        "MAIN",
    "installed":   "2018-03-15",
    "model":       "Carrier 30XA-1002",
    "refrigerant": "R-134a",
    "capacity_kW": 350,
    "last_service": "2024-11-20",
    "open_tickets": 0,
}

# ── HVAC fault library ────────────────────────────────────────────────────────
FAULT_LIBRARY = {
    "EL-020": {
        "name": "Compressor Electrical Fault",
        "signatures": ["active_power spike", "supply voltage deviation", "high winding temp"],
        "severity": "High",
        "typical_causes": ["capacitor failure", "winding insulation breakdown", "contactor wear"],
        "mttr_hours": 4,
    },
    "RF-010": {
        "name": "Refrigerant Overcharge / Condenser Fouling",
        "signatures": ["high_pressure_1 elevated", "reduced efficiency", "head pressure alarm"],
        "severity": "High",
        "typical_causes": ["dirty condenser coils", "refrigerant overcharge", "non-condensables"],
        "mttr_hours": 3,
    },
    "RF-005": {
        "name": "Refrigerant Leak / Low Charge",
        "signatures": ["low_pressure_1 dropping", "superheat deviation", "reduced capacity"],
        "severity": "Critical",
        "typical_causes": ["Schrader valve leak", "brazed joint failure", "filter-drier saturation"],
        "mttr_hours": 6,
    },
    "MT-030": {
        "name": "Coil Fouling / Airflow Restriction",
        "signatures": ["outlet_temp elevated", "delta-T reduction", "increased runtime"],
        "severity": "Medium",
        "typical_causes": ["dirty air filters", "fouled evaporator coil", "fan belt slippage"],
        "mttr_hours": 2,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared agentic loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_agent_loop(
    tools: list[dict],
    tool_map: dict,
    messages: list[dict],
    system: str,
    max_iter: int = 8,
) -> str:
    """Standard tool-calling loop. Returns final text response."""
    if _client is None:
        return "ANTHROPIC_API_KEY not configured. Add it to your .env file."

    for _ in range(max_iter):
        try:
            response = _client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=tools,
                messages=messages,
            )
        except anthropic.BadRequestError as e:
            msg = str(e)
            if "credit balance is too low" in msg:
                return ("API credits exhausted. Please top up at "
                        "console.anthropic.com/settings/billing, then retry.")
            return f"API request error: {msg}"
        except anthropic.AuthenticationError:
            return "Invalid ANTHROPIC_API_KEY. Check your .env file."
        except Exception as e:
            return f"Unexpected error calling Claude API: {e}"

        if response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if hasattr(b, "text")),
                "(no response)",
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = tool_map.get(block.name)
                    out = fn(block.input) if fn else '{"error":"unknown tool"}'
                    results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     out,
                    })
            messages.append({"role": "user", "content": results})

    return "(agent loop exceeded max iterations)"


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Sensor Intelligence
# ══════════════════════════════════════════════════════════════════════════════

_SENSOR_SYSTEM = """\
You are the Sensor Intelligence Agent for Sync.ai's Industrial AI platform.

When an anomaly fires on the HVAC chiller system you:
1. Call get_sensor_snapshot for the key sensors to see current values + recent stats.
2. Call get_anomaly_context for the statistical deviation details.
3. Call get_asset_profile for asset metadata.
4. Write a clear technical narrative (under 200 words) for a senior maintenance engineer:
   WHAT happened, WHICH sensors co-moved, and WHAT the combined pattern suggests.
   Reference actual values and z-scores. Do not use bullet points — write flowing prose.\
"""

_SENSOR_TOOLS = [
    {
        "name": "get_sensor_snapshot",
        "description": "Current value + 1-hour and 12-hour stats for requested sensors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sensor_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sensor column names e.g. active_power, high_pressure_1",
                }
            },
            "required": ["sensor_names"],
        },
    },
    {
        "name": "get_anomaly_context",
        "description": "Statistical context of the detected anomaly: z-score, baseline, pressure ratio.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_asset_profile",
        "description": "Static asset information: model, site, install date.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_sensor_agent(alert: dict, window_df: pd.DataFrame, current: dict) -> str:
    """Agent 1: cross-correlate sensors, explain the anomaly. Returns analysis text."""

    def _snapshot(inp: dict) -> str:
        out = {}
        for col in inp.get("sensor_names", []):
            if col not in window_df.columns:
                continue
            s = window_df[col].dropna()
            b = s.iloc[:-1].tail(12)
            out[col] = {
                "current":  round(float(current.get(col, 0)), 3),
                "1hr_mean": round(float(b.mean()), 3),
                "1hr_std":  round(float(b.std()), 3),
                "12hr_min": round(float(s.min()), 3),
                "12hr_max": round(float(s.max()), 3),
            }
        return json.dumps(out)

    def _context(_: dict) -> str:
        hp = float(current.get("high_pressure_1", 0))
        lp = max(float(current.get("low_pressure_1", 0.01)), 0.01)
        return json.dumps({
            "triggered_sensor": alert["sensor"],
            "current_value":    alert["value"],
            "z_score":          alert["z_score"],
            "severity":         alert["severity"],
            "fault_code":       alert["fault_code"],
            "sim_timestamp":    str(alert["timestamp"]),
            "pressure_ratio":   round(hp / lp, 3),
        })

    tool_map = {
        "get_sensor_snapshot": _snapshot,
        "get_anomaly_context": _context,
        "get_asset_profile":   lambda _: json.dumps(ASSET_PROFILE),
    }
    messages = [{
        "role": "user",
        "content": (
            f"Anomaly detected on Chiller Unit A1 at {alert['timestamp']}.\n"
            f"Sensor: {alert['sensor']}, value: {alert['value']}, "
            f"z-score: {alert['z_score']}sigma, severity: {alert['severity']}.\n"
            f"Analyse the sensor data and explain what is happening."
        ),
    }]
    return _run_agent_loop(_SENSOR_TOOLS, tool_map, messages, _SENSOR_SYSTEM)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

_DIAGNOSIS_SYSTEM = """\
You are the Diagnosis Agent for Sync.ai's Industrial AI platform.

You receive sensor analysis from the Sensor Intelligence Agent. Your job:
1. Call get_fault_library for the relevant fault codes.
2. Call get_system_health_metrics for real-time efficiency data.
3. Output a ranked list of the top 2-3 root causes with confidence percentages (sum <= 100%).
4. State: Urgency (Critical / High / Medium), estimated time-to-failure if unaddressed,
   and the single most important immediate action.

Be direct. Under 220 words. Use numbered root causes, then urgency + action as a final paragraph.\
"""

_DIAGNOSIS_TOOLS = [
    {
        "name": "get_fault_library",
        "description": "Retrieve HVAC failure mode signatures for given fault codes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fault_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "e.g. ['EL-020', 'RF-005']",
                }
            },
            "required": ["fault_codes"],
        },
    },
    {
        "name": "get_system_health_metrics",
        "description": "Real-time pressure ratio, power deviation, runtime percentage.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_diagnosis_agent(
    alert: dict,
    sensor_text: str,
    current: dict,
    window_df: pd.DataFrame,
) -> str:
    """Agent 2: rank root causes. Returns diagnosis text."""

    def _fault_lib(inp: dict) -> str:
        codes = inp.get("fault_codes", [])
        result = {c: FAULT_LIBRARY[c] for c in codes if c in FAULT_LIBRARY}
        return json.dumps(result if result else FAULT_LIBRARY)

    def _health(_: dict) -> str:
        hp  = float(current.get("high_pressure_1", 0))
        lp  = max(float(current.get("low_pressure_1", 0.01)), 0.01)
        pwr = float(current.get("active_power", 0))
        avg = float(window_df["active_power"].tail(12).mean())
        return json.dumps({
            "pressure_ratio":            round(hp / lp, 3),
            "pressure_ratio_normal":     "1.0 - 1.4",
            "current_power_W":           round(pwr, 2),
            "1hr_avg_power_W":           round(avg, 2),
            "power_deviation_pct":       round((pwr - avg) / max(avg, 0.01) * 100, 1),
            "high_pressure_bar":         round(hp, 2),
            "low_pressure_bar":          round(lp, 2),
            "runtime_on_pct_12hr":       round(float(window_df["on_off"].mean() * 100), 1),
        })

    tool_map = {
        "get_fault_library":         _fault_lib,
        "get_system_health_metrics": _health,
    }
    messages = [{
        "role": "user",
        "content": (
            f"Sensor Intelligence Analysis:\n{sensor_text}\n\n"
            f"Primary fault code: {alert['fault_code']}, priority: {alert['priority']}.\n"
            f"Diagnose the root cause and rank failure modes with confidence percentages."
        ),
    }]
    return _run_agent_loop(_DIAGNOSIS_TOOLS, tool_map, messages, _DIAGNOSIS_SYSTEM)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Action / Work Order
# ══════════════════════════════════════════════════════════════════════════════

_ACTION_SYSTEM = """\
You are the Action Agent for Sync.ai's Industrial AI platform.

You receive a fault diagnosis and generate a complete maintenance work order.
1. Call get_maintenance_history to check asset service context.
2. Call create_work_order with ALL required fields — this creates the official WO in the CMMS.
3. Confirm in under 80 words: WO number, one-line summary, estimated hours, priority.

Work order steps must be ordered: Safety -> Diagnostic -> Repair -> Verification.
Parts must be realistic for the fault code.\
"""

_ACTION_TOOLS = [
    {
        "name": "get_maintenance_history",
        "description": "Recent maintenance history and open tickets for the asset.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_work_order",
        "description": "Create a structured maintenance work order in the CMMS.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fault_code":      {"type": "string"},
                "priority":        {"type": "string", "enum": ["Emergency", "High", "Medium", "Low"]},
                "title":           {"type": "string"},
                "steps":           {"type": "array", "items": {"type": "string"}},
                "parts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":    {"type": "string"},
                            "qty":     {"type": "integer"},
                            "part_no": {"type": "string"},
                        },
                    },
                },
                "estimated_hours": {"type": "number"},
                "safety_notes":    {"type": "array", "items": {"type": "string"}},
            },
            "required": ["fault_code", "priority", "title", "steps",
                         "parts", "estimated_hours", "safety_notes"],
        },
    },
]


def run_action_agent(
    alert: dict,
    diagnosis_text: str,
    current: dict,
) -> tuple[str, dict]:
    """Agent 3: generate work order. Returns (confirmation_text, work_order_dict)."""
    captured: dict = {}

    def _history(_: dict) -> str:
        return json.dumps({
            "asset_id":    ASSET_PROFILE["asset_id"],
            "last_service": ASSET_PROFILE["last_service"],
            "open_tickets": ASSET_PROFILE["open_tickets"],
            "recent_work_orders": [
                {"date": "2024-11-20", "type": "Preventive Maintenance", "tech": "J. Martinez"},
                {"date": "2024-08-05", "type": "Refrigerant top-up",    "tech": "S. Patel"},
            ],
        })

    def _create_wo(inp: dict) -> str:
        wo_num = f"WO-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
        wo = {
            "wo_number":       wo_num,
            "fault_code":      inp.get("fault_code", ""),
            "priority":        inp.get("priority", "High"),
            "title":           inp.get("title", ""),
            "asset":           ASSET_PROFILE["name"],
            "site":            ASSET_PROFILE["site"],
            "created_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "steps":           inp.get("steps", []),
            "parts":           inp.get("parts", []),
            "estimated_hours": inp.get("estimated_hours", 0),
            "safety_notes":    inp.get("safety_notes", []),
            "status":          "Pending Approval",
        }
        captured.update(wo)
        return json.dumps({"success": True, "wo_number": wo_num})

    tool_map = {
        "get_maintenance_history": _history,
        "create_work_order":       _create_wo,
    }
    messages = [{
        "role": "user",
        "content": (
            f"Diagnosis:\n{diagnosis_text}\n\n"
            f"Fault code: {alert['fault_code']}, priority: {alert['priority']}.\n"
            f"Generate the maintenance work order now."
        ),
    }]
    text = _run_agent_loop(_ACTION_TOOLS, tool_map, messages, _ACTION_SYSTEM)
    return text, captured

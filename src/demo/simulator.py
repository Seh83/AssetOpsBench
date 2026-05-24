"""
Sync.ai Demo — Simulation Engine
Manages replay state, scenario selection, and stream cursor.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[2] / "src" / "dataset" / "hvac_demo_ready.csv"

SCENARIOS = {
    "July 14 — Power Spike (Critical)": 12640,   # 2022-07-14 21:20 — 30 min before 9.7σ spike
    "June 12 — Heat Soak Event":        3342,    # 2022-06-12 14:30 — 6 hrs before outlet_temp peak
    "August 7 — Pressure Rise Event":     19473,   # 2022-08-07 14:45 — pressure Watch fires ~55 min in
    "Full Replay (from June 1)":        0,
}

SPEED_OPTIONS = {
    "1x  — Real-time feel": 1,
    "5x  — Fast demo":      5,
    "20x — Quick scan":     20,
    "50x — Rapid review":   50,
}

WINDOW_ROWS = 144  # 12 hours of 5-min data
ALERT_TTL    = 30  # ticks before alert clears


@st.cache_data
def load_full_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def init_state(df: pd.DataFrame) -> None:
    """Initialise session state on first load — idempotent."""
    if "sim_initialized" not in st.session_state:
        scenario = list(SCENARIOS.keys())[0]
        st.session_state.sim_initialized  = True
        st.session_state.sim_row          = SCENARIOS[scenario]
        st.session_state.sim_scenario     = scenario
        st.session_state.sim_speed        = list(SPEED_OPTIONS.keys())[1]  # 5x default
        st.session_state.sim_playing      = False
        st.session_state.sim_event_log    = []
        st.session_state.sim_active_alert   = None
        st.session_state.sim_alert_tick    = 0
        st.session_state.sim_max_row       = len(df) - 1
        # Agent pipeline state
        st.session_state.sim_agent_status   = "idle"   # idle|pending|agent1_done|agent2_done|done
        st.session_state.sim_agent_results  = {}
        st.session_state.sim_agent_alert_id = None
        st.session_state.sim_agents_visible = True


def tick(df: pd.DataFrame) -> None:
    """Advance the cursor by speed rows if playing. Wrap at end of data."""
    if not st.session_state.sim_playing:
        return
    speed = SPEED_OPTIONS[st.session_state.sim_speed]
    new_row = st.session_state.sim_row + speed
    if new_row >= st.session_state.sim_max_row:
        # Loop back to scenario start
        new_row = SCENARIOS[st.session_state.sim_scenario]
        st.session_state.sim_event_log    = []
        st.session_state.sim_active_alert = None
        st.session_state.sim_alert_tick   = 0
    st.session_state.sim_row = new_row

    # Count down alert TTL
    if st.session_state.sim_active_alert:
        st.session_state.sim_alert_tick -= 1
        if st.session_state.sim_alert_tick <= 0:
            st.session_state.sim_active_alert = None


def get_window(df: pd.DataFrame, n: int = WINDOW_ROWS) -> pd.DataFrame:
    """Return the last n rows up to and including the current cursor."""
    end   = st.session_state.sim_row + 1
    start = max(0, end - n)
    return df.iloc[start:end].copy()


def get_current(df: pd.DataFrame) -> dict:
    """Return the current row as a dict."""
    return df.iloc[st.session_state.sim_row].to_dict()


def push_alert(alert: dict) -> None:
    """Register a new anomaly alert, add to event log."""
    st.session_state.sim_active_alert = alert
    st.session_state.sim_alert_tick   = ALERT_TTL
    st.session_state.sim_event_log.insert(0, alert)
    # Keep log bounded
    if len(st.session_state.sim_event_log) > 100:
        st.session_state.sim_event_log = st.session_state.sim_event_log[:100]


def reset(scenario: str | None = None) -> None:
    """Reset cursor to scenario start and clear event log."""
    if scenario:
        st.session_state.sim_scenario = scenario
    st.session_state.sim_row          = SCENARIOS[st.session_state.sim_scenario]
    st.session_state.sim_playing      = False
    st.session_state.sim_event_log    = []
    st.session_state.sim_active_alert   = None
    st.session_state.sim_alert_tick    = 0
    st.session_state.sim_agent_status  = "idle"
    st.session_state.sim_agent_results = {}
    st.session_state.sim_agent_alert_id = None

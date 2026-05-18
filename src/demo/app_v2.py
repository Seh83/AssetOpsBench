"""
Sync.ai — Real-Time HVAC Asset Intelligence Simulation
Run: streamlit run src/demo/app_v2.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from simulator   import (load_full_data, init_state, tick, get_window,
                          get_current, push_alert, reset,
                          SCENARIOS, SPEED_OPTIONS, WINDOW_ROWS)
from anomaly_live import detect, is_duplicate
from charts      import (gauge_chart, scrolling_chart,
                          temperature_panel, pressure_history_panel,
                          system_status_panel, humidity_co2_panel)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sync.ai — Live Asset Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .syncai-header {
    background: linear-gradient(90deg, #0A1628 0%, #0D2348 100%);
    padding: 14px 24px; border-radius: 10px; margin-bottom: 16px;
    border-left: 4px solid #00C2FF;
    display: flex; justify-content: space-between; align-items: center;
  }
  .syncai-logo { font-size: 24px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
  .syncai-logo span { color: #00C2FF; }
  .syncai-tag  { font-size: 12px; color: #8BA3C7; margin-top: 2px; }
  .sim-ticker  {
    text-align: right; font-size: 12px; color: #8BA3C7;
    font-family: monospace;
  }
  .sim-ticker b { color: #00C2FF; font-size: 14px; }

  /* Alert banners */
  .alert-critical {
    background: linear-gradient(90deg,#3D0A0A,#2A0A0A);
    border: 1px solid #FF4444; border-left: 5px solid #FF4444;
    border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;
    animation: pulse-red 1.5s infinite;
  }
  .alert-warning {
    background: linear-gradient(90deg,#2A1500,#1A0E00);
    border: 1px solid #FF8C00; border-left: 5px solid #FF8C00;
    border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;
  }
  .alert-watch {
    background: linear-gradient(90deg,#1A1600,#111000);
    border: 1px solid #FFD700; border-left: 5px solid #FFD700;
    border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;
  }
  @keyframes pulse-red {
    0%,100% { border-color: #FF4444; box-shadow: 0 0 0 0 rgba(255,68,68,0); }
    50%      { border-color: #FF8888; box-shadow: 0 0 8px 4px rgba(255,68,68,0.3); }
  }
  .alert-title  { font-size: 15px; font-weight: 700; color: #fff; }
  .alert-detail { font-size: 12px; color: #8BA3C7; margin-top: 4px; }

  /* KPI mini cards */
  .kpi-row { display: flex; gap: 12px; margin-bottom: 16px; }
  .kpi-mini {
    flex: 1; background: #0D2348; border: 1px solid #1A3A6B;
    border-radius: 8px; padding: 12px 14px; text-align: center;
  }
  .kpi-mini .label { font-size: 10px; color: #8BA3C7; text-transform: uppercase; letter-spacing: 0.6px; }
  .kpi-mini .value { font-size: 22px; font-weight: 700; color: #fff; margin: 2px 0; }
  .kpi-mini .sub   { font-size: 10px; color: #00C2FF; }

  /* Sidebar event badge */
  .ev-badge {
    background: #0D2348; border: 1px solid #1A3A6B; border-radius: 6px;
    padding: 6px 10px; margin-bottom: 6px; font-size: 11px; color: #8BA3C7;
  }
  .ev-badge .ev-sensor { color: #fff; font-weight: 600; }
  .ev-badge .ev-crit   { color: #FF4444; }
  .ev-badge .ev-warn   { color: #FF8C00; }
  .ev-badge .ev-watch  { color: #FFD700; }

  /* Section labels */
  .section-label {
    font-size: 11px; font-weight: 600; color: #8BA3C7;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;
  }

  /* Play/Pause buttons */
  .stButton > button {
    border-radius: 6px; font-weight: 600; font-size: 13px;
    transition: all 0.2s;
  }
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh: 1 tick per second ──────────────────────────────────────────
st_autorefresh(interval=1000, key="sim_clock")

# ── Load data & init state ────────────────────────────────────────────────────
df = load_full_data()
init_state(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Simulation Controls")

    # Scenario selector
    new_scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index(st.session_state.sim_scenario),
        key="scenario_select",
    )
    if new_scenario != st.session_state.sim_scenario:
        reset(new_scenario)
        st.rerun()

    # Speed
    new_speed = st.selectbox(
        "Speed",
        options=list(SPEED_OPTIONS.keys()),
        index=list(SPEED_OPTIONS.keys()).index(st.session_state.sim_speed),
        key="speed_select",
    )
    st.session_state.sim_speed = new_speed

    # Play / Pause / Reset buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("PLAY" if not st.session_state.sim_playing else "PLAYING",
                     type="primary" if not st.session_state.sim_playing else "secondary",
                     use_container_width=True):
            st.session_state.sim_playing = True
    with c2:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.sim_playing = False
    with c3:
        if st.button("RESET", use_container_width=True):
            reset()
            st.rerun()

    st.divider()

    # Live event log in sidebar
    st.markdown("**Live Event Log**")
    log = st.session_state.sim_event_log
    if not log:
        st.caption("No events detected yet.")
    else:
        for ev in log[:6]:
            sev_class = {"Critical": "ev-crit", "Warning": "ev-warn"}.get(ev["severity"], "ev-watch")
            ts_str = pd.Timestamp(ev["timestamp"]).strftime("%H:%M")
            st.markdown(f"""
            <div class="ev-badge">
              <span class="{sev_class}">● {ev['severity']}</span> &nbsp;
              <span class="ev-sensor">{ev['sensor']}</span><br>
              {ts_str} &nbsp;|&nbsp; z={ev['z_score']} &nbsp;|&nbsp; {ev['value']}
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Asset**")
    st.caption("Chiller Unit A1 — Building Main")
    st.caption("Site: MAIN  |  Install: 2018")
    energy_rate = st.number_input("Energy Rate ($/kWh)", value=0.12, step=0.01, format="%.3f")

# ── Advance simulation on each tick ──────────────────────────────────────────
tick(df)
window = get_window(df, WINDOW_ROWS)
current = get_current(df)

# ── Live anomaly detection ────────────────────────────────────────────────────
alert = detect(window)
if alert and not is_duplicate(alert, st.session_state.sim_event_log):
    push_alert(alert)

active_alert = st.session_state.sim_active_alert

# ── Header ────────────────────────────────────────────────────────────────────
sim_time_str = pd.Timestamp(current["timestamp"]).strftime("%a %d %b %Y  %H:%M:%S")
status_str   = "RUNNING" if current.get("on_off", 0) == 1 else "STANDBY"
status_color = "#00C851" if current.get("on_off", 0) == 1 else "#8BA3C7"
play_icon    = "LIVE" if st.session_state.sim_playing else "PAUSED"
play_color   = "#00C851" if st.session_state.sim_playing else "#FF8C00"

st.markdown(f"""
<div class="syncai-header">
  <div>
    <div class="syncai-logo">Sync<span>.ai</span></div>
    <div class="syncai-tag">Industrial AI Asset Intelligence &nbsp;|&nbsp;
      <span style="color:{status_color};">{status_str}</span>
    </div>
  </div>
  <div class="sim-ticker">
    <span style="color:{play_color};font-weight:700;">{play_icon}</span><br>
    SIM TIME<br>
    <b>{sim_time_str}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Alert banner ──────────────────────────────────────────────────────────────
if active_alert:
    sev = active_alert["severity"]
    cls = {"Critical": "alert-critical", "Warning": "alert-warning"}.get(sev, "alert-watch")
    icon = {"Critical": "CRITICAL ANOMALY", "Warning": "WARNING", "Watch": "WATCH"}.get(sev, "ALERT")
    st.markdown(f"""
    <div class="{cls}">
      <div class="alert-title">
        {icon} &nbsp;|&nbsp; {active_alert['sensor'].replace('_', ' ').upper()}
        &nbsp;|&nbsp; z = {active_alert['z_score']} sigma
        &nbsp;|&nbsp; {active_alert['value']} W
      </div>
      <div class="alert-detail">
        {active_alert['failure_mode']} &nbsp;&mdash;&nbsp; {active_alert['action']}
        &nbsp;|&nbsp; Fault Code: {active_alert['fault_code']}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
power_val = float(current.get("active_power", 0))
hp_val    = float(current.get("high_pressure_1", 0))
lp_val    = float(current.get("low_pressure_1", 0))
pr_val    = round(hp_val / lp_val, 2) if lp_val > 0 else 0
n_events  = len(st.session_state.sim_event_log)
pr_color  = "#FF4444" if pr_val > 1.6 else ("#FFD700" if pr_val > 1.4 else "#00C851")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Active Power", f"{power_val:.1f} W",
              delta=f"{power_val - window['active_power'].mean():.1f} vs avg")
with c2:
    st.metric("High Pressure", f"{hp_val:.2f} bar",
              delta=f"{hp_val - window['high_pressure_1'].mean():.2f} vs avg")
with c3:
    st.metric("Low Pressure", f"{lp_val:.2f} bar",
              delta=f"{lp_val - window['low_pressure_1'].mean():.2f} vs avg")
with c4:
    st.metric("Pressure Ratio", f"{pr_val:.2f}",
              delta="Normal" if pr_val <= 1.4 else "Elevated",
              delta_color="normal" if pr_val <= 1.4 else "inverse")
with c5:
    st.metric("Events Detected", n_events,
              delta=f"{len([e for e in log[:3] if e.get('severity')=='Critical'])} critical")

# ── Gauge row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Live Sensor Gauges</div>', unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)

power_baseline = float(window["active_power"].mean())
hp_baseline    = float(window["high_pressure_1"].mean())
lp_baseline    = float(window["low_pressure_1"].mean())

with g1:
    st.plotly_chart(gauge_chart(
        value=power_val, title="Active Power", unit="W",
        min_val=0, max_val=60,
        green_max=30, yellow_max=45,
        baseline=power_baseline,
    ), use_container_width=True)

with g2:
    st.plotly_chart(gauge_chart(
        value=hp_val, title="High Pressure", unit="bar",
        min_val=8, max_val=36,
        green_max=22, yellow_max=28,
        baseline=hp_baseline,
    ), use_container_width=True)

with g3:
    st.plotly_chart(gauge_chart(
        value=lp_val, title="Low Pressure", unit="bar",
        min_val=6, max_val=26,
        green_max=16, yellow_max=20,
        baseline=lp_baseline,
    ), use_container_width=True)

# ── Scrolling live chart ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">Live Sensor Stream — 12hr Rolling Window</div>',
            unsafe_allow_html=True)
st.plotly_chart(scrolling_chart(window, active_alert), use_container_width=True)

# ── 4 mini panels ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Sensor Panels</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)

with p1:
    st.plotly_chart(temperature_panel(window), use_container_width=True)
with p2:
    st.plotly_chart(pressure_history_panel(window), use_container_width=True)
with p3:
    st.plotly_chart(system_status_panel(window), use_container_width=True)
with p4:
    st.plotly_chart(humidity_co2_panel(window), use_container_width=True)

# ── Event log table ───────────────────────────────────────────────────────────
if log:
    st.divider()
    st.markdown('<div class="section-label">Anomaly Event Log</div>', unsafe_allow_html=True)
    log_df = pd.DataFrame(log)
    display_cols = ["timestamp", "sensor", "value", "z_score", "severity",
                    "failure_mode", "fault_code", "priority"]
    available = [c for c in display_cols if c in log_df.columns]
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        log_df[available].rename(columns={
            "timestamp": "Sim Time", "sensor": "Sensor", "value": "Value",
            "z_score": "Z-Score", "severity": "Severity",
            "failure_mode": "Failure Mode", "fault_code": "Fault Code",
            "priority": "Priority",
        }),
        use_container_width=True,
        height=min(300, 40 + len(log_df) * 35),
    )

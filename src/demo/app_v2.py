"""
Sync.ai — Real-Time HVAC Asset Intelligence Simulation
Run: streamlit run src/demo/app_v2.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent))
from simulator    import (load_full_data, init_state, tick, get_window,
                           get_current, push_alert, reset,
                           SCENARIOS, SPEED_OPTIONS, WINDOW_ROWS)
from anomaly_live import detect, is_duplicate
from charts       import (gauge_chart, scrolling_chart,
                           temperature_panel, pressure_history_panel,
                           system_status_panel, humidity_co2_panel,
                           agent_flow_html)
from agents       import run_sensor_agent, run_diagnosis_agent, run_action_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sync.ai — Live Asset Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Production Design System ──────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
/* ── Design Tokens ── */
:root {
  --bg-base:      #070F1D;
  --bg-card:      #0D1F3C;
  --bg-card-2:    #091627;
  --bg-card-3:    #0A1628;
  --border:       #112240;
  --border-2:     #1A3A6B;
  --text-primary: #FFFFFF;
  --text-sec:     #C8D8F0;
  --text-muted:   #8BA3C7;
  --text-dim:     #4A6A9B;
  --accent-blue:  #00C2FF;
  --accent-green: #00C851;
  --accent-orange:#FF8C00;
  --accent-red:   #FF4444;
  --accent-yellow:#FFD700;
  --accent-purple:#9B59B6;
  --radius-sm:    6px;
  --radius-md:    10px;
  --radius-lg:    14px;
  --shadow-card:  0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
  --font-main:    'Inter', 'Segoe UI', system-ui, sans-serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Base reset ── */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
  font-family: var(--font-main) !important;
  background-color: var(--bg-base) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Streamlit overrides ── */
[data-testid="stSidebar"] {
  background: var(--bg-card-3) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-main) !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
[data-testid="stMetricDelta"] > div {
  font-size: 11px !important;
  font-weight: 500 !important;
}
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 16px 12px !important;
  box-shadow: var(--shadow-card) !important;
  transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-2) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg-card-3);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
  border-bottom: 1px solid var(--border);
  gap: 2px; padding: 4px 6px 0;
}
[data-testid="stTabs"] [role="tab"] {
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  padding: 8px 18px !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.2s, border-color 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--accent-blue) !important;
  border-bottom-color: var(--accent-blue) !important;
  background: rgba(0,194,255,0.06) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text-sec) !important;
  background: rgba(255,255,255,0.03) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.3px !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00A8E0, #0070A8) !important;
  border: 1px solid rgba(0,194,255,0.4) !important;
  box-shadow: 0 0 12px rgba(0,194,255,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 20px rgba(0,194,255,0.4) !important;
  transform: translateY(-1px) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── ════════════════════════════════════════ ──
   Custom components
── ════════════════════════════════════════ ── */

/* HEADER */
.syncai-header {
  background: linear-gradient(90deg, #091627 0%, #0A1E3A 60%, #071628 100%);
  padding: 16px 24px;
  border-radius: var(--radius-lg);
  margin-bottom: 18px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-blue);
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 20px rgba(0,194,255,0.08);
}
.syncai-logo {
  font-size: 26px; font-weight: 800; color: #fff;
  letter-spacing: -0.8px; line-height: 1;
}
.syncai-logo span { color: var(--accent-blue); }
.syncai-tag {
  font-size: 11px; color: var(--text-muted);
  margin-top: 4px; font-weight: 500; letter-spacing: 0.3px;
}
.sim-ticker {
  text-align: right; font-size: 11px; color: var(--text-dim);
  font-family: var(--font-mono);
}
.sim-ticker .ticker-time {
  font-size: 15px; font-weight: 600;
  color: var(--accent-blue); letter-spacing: 0.5px;
}
.sim-status-dot {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; margin-right: 5px;
  animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink {
  0%,100% { opacity:1; } 50% { opacity:0.35; }
}

/* ALERT BANNERS */
.alert-banner {
  border-radius: var(--radius-md);
  padding: 14px 20px;
  margin-bottom: 16px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
}
.alert-banner .alert-icon {
  font-size: 20px; flex-shrink: 0; margin-top: 1px;
}
.alert-critical {
  background: linear-gradient(135deg, #1A0505, #0F0303);
  border: 1px solid rgba(255,68,68,0.6);
  border-left: 4px solid var(--accent-red);
  box-shadow: 0 0 20px rgba(255,68,68,0.12);
  animation: pulse-red 1.8s ease-in-out infinite;
}
.alert-warning {
  background: linear-gradient(135deg, #1A0A00, #100600);
  border: 1px solid rgba(255,140,0,0.5);
  border-left: 4px solid var(--accent-orange);
  box-shadow: 0 0 16px rgba(255,140,0,0.10);
}
.alert-watch {
  background: linear-gradient(135deg, #14110000, #0A0900);
  border: 1px solid rgba(255,215,0,0.4);
  border-left: 4px solid var(--accent-yellow);
}
@keyframes pulse-red {
  0%,100% { box-shadow: 0 0 16px rgba(255,68,68,0.12); }
  50%      { box-shadow: 0 0 28px rgba(255,68,68,0.30); }
}
.alert-title {
  font-size: 14px; font-weight: 700; color: #fff;
  letter-spacing: 0.2px; line-height: 1.3;
}
.alert-detail {
  font-size: 12px; color: var(--text-muted);
  margin-top: 5px; line-height: 1.5;
}
.alert-pill {
  display: inline-block; font-size: 10px; font-weight: 700;
  padding: 2px 8px; border-radius: 20px;
  vertical-align: middle; margin-left: 6px;
}
.pill-crit   { background:rgba(255,68,68,0.2);  color:var(--accent-red);    border:1px solid rgba(255,68,68,0.4); }
.pill-warn   { background:rgba(255,140,0,0.2);  color:var(--accent-orange); border:1px solid rgba(255,140,0,0.4); }
.pill-watch  { background:rgba(255,215,0,0.15); color:var(--accent-yellow); border:1px solid rgba(255,215,0,0.3); }

/* SECTION LABELS */
.section-label {
  font-size: 10px; font-weight: 700; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 1.2px;
  margin: 18px 0 10px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}

/* EVENT BADGES (sidebar) */
.ev-badge {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 10px; margin-bottom: 6px;
  font-size: 11px; color: var(--text-muted);
  transition: border-color 0.2s;
}
.ev-badge:hover { border-color: var(--border-2); }
.ev-sensor { color: var(--text-primary); font-weight: 600; }
.ev-crit   { color: var(--accent-red);    font-weight: 700; }
.ev-warn   { color: var(--accent-orange); font-weight: 700; }
.ev-watch  { color: var(--accent-yellow); font-weight: 700; }

/* AGENT MINI CARDS (sidebar) */
.agent-mini {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px; margin-bottom: 7px;
  font-size: 11px;
}
.am-title { color: var(--accent-blue); font-weight: 700; margin-bottom: 4px; font-size: 11px; }
.am-body  { color: var(--text-muted); line-height: 1.45; }
.am-run   { color: var(--accent-yellow); font-style: italic; }

/* AGENT CARDS (main) */
.agent-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-blue);
  border-radius: var(--radius-md);
  padding: 20px 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-card);
}
.agent-card.diagnosis { border-left-color: var(--accent-orange); }
.agent-card.action    { border-left-color: var(--accent-green); }
.ac-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 14px; padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.ac-icon { font-size: 18px; }
.ac-title {
  font-size: 11px; font-weight: 700; color: var(--text-primary);
  text-transform: uppercase; letter-spacing: 0.8px;
}
.ac-status {
  margin-left: auto; font-size: 10px; font-weight: 600;
  padding: 2px 8px; border-radius: 20px;
  background: rgba(0,200,81,0.15); color: var(--accent-green);
  border: 1px solid rgba(0,200,81,0.3);
}
.ac-body {
  font-size: 13px; color: var(--text-sec);
  line-height: 1.75; white-space: pre-wrap;
  font-family: var(--font-main);
}

/* WORK ORDER CARD */
.wo-card {
  background: linear-gradient(135deg, #071D14, #050F0A);
  border: 1px solid rgba(0,200,81,0.4);
  border-radius: var(--radius-lg);
  padding: 22px 24px;
  margin-bottom: 18px;
  box-shadow: 0 0 24px rgba(0,200,81,0.08);
}
.wo-header {
  display: flex; justify-content: space-between;
  align-items: flex-start; margin-bottom: 14px;
  padding-bottom: 14px; border-bottom: 1px solid rgba(0,200,81,0.15);
}
.wo-num {
  font-size: 22px; font-weight: 800;
  color: var(--accent-blue); font-family: var(--font-mono);
  letter-spacing: 0.5px;
}
.wo-title { font-size: 15px; font-weight: 700; color: #fff; margin: 4px 0; }
.wo-meta  { font-size: 11px; color: var(--text-muted); line-height: 1.6; }
.wo-badge {
  font-size: 10px; font-weight: 700;
  padding: 4px 12px; border-radius: 20px; white-space: nowrap;
}
.wo-badge.High      { background:rgba(255,140,0,0.15); color:var(--accent-orange); border:1px solid rgba(255,140,0,0.4); }
.wo-badge.Emergency { background:rgba(255,68,68,0.15);  color:var(--accent-red);    border:1px solid rgba(255,68,68,0.5); }
.wo-badge.Medium    { background:rgba(255,215,0,0.12);  color:var(--accent-yellow); border:1px solid rgba(255,215,0,0.35); }
.wo-badge.Low       { background:rgba(0,194,255,0.12);  color:var(--accent-blue);   border:1px solid rgba(0,194,255,0.3); }

/* KPI grid row */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st_autorefresh(interval=1000, key="sim_clock")

# ── Load data & init state ────────────────────────────────────────────────────
df = load_full_data()
init_state(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Simulation Controls")

    new_scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index(st.session_state.sim_scenario),
        key="scenario_select",
    )
    if new_scenario != st.session_state.sim_scenario:
        reset(new_scenario)
        st.rerun()

    new_speed = st.selectbox(
        "Speed",
        options=list(SPEED_OPTIONS.keys()),
        index=list(SPEED_OPTIONS.keys()).index(st.session_state.sim_speed),
        key="speed_select",
    )
    st.session_state.sim_speed = new_speed

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

    # Live Event Log
    st.markdown("**Live Event Log**")
    log = st.session_state.sim_event_log
    if not log:
        st.caption("No events detected yet.")
    else:
        for ev in log[:5]:
            sev_class = {"Critical": "ev-crit", "Warning": "ev-warn"}.get(ev["severity"], "ev-watch")
            ts_str = pd.Timestamp(ev["timestamp"]).strftime("%H:%M")
            st.markdown(f"""
            <div class="ev-badge">
              <span class="{sev_class}">&#9679; {ev['severity']}</span>&nbsp;
              <span class="ev-sensor">{ev['sensor']}</span><br>
              {ts_str}&nbsp;|&nbsp;z={ev['z_score']}&nbsp;|&nbsp;{ev['value']}
            </div>""", unsafe_allow_html=True)

    st.divider()

    # AI Agent Insights — show/hide toggle
    col_ai_title, col_ai_toggle = st.columns([3, 1])
    with col_ai_title:
        st.markdown("**AI Agent Insights**")
    with col_ai_toggle:
        toggle_lbl = "Hide" if st.session_state.sim_agents_visible else "Show"
        if st.button(toggle_lbl, key="toggle_agents", use_container_width=True):
            st.session_state.sim_agents_visible = not st.session_state.sim_agents_visible
            st.rerun()

    if st.session_state.sim_agents_visible:
        a_status  = st.session_state.sim_agent_status
        a_results = st.session_state.sim_agent_results

        if a_status == "idle":
            st.caption("Agents on standby. Will activate on next anomaly.")
        elif a_status == "pending":
            st.markdown('<div class="agent-mini"><div class="am-title">Agent 1 — Sensor Intel</div>'
                        '<div class="am-run">Analyzing sensors...</div></div>', unsafe_allow_html=True)
        elif a_status == "agent1_done":
            snip = a_results.get("sensor_text", "")[:120] + "..."
            st.markdown(f'<div class="agent-mini"><div class="am-title">&#10003; Sensor Intel</div>'
                        f'<div class="am-body">{snip}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-mini"><div class="am-title">Agent 2 — Diagnosis</div>'
                        '<div class="am-run">Diagnosing root cause...</div></div>', unsafe_allow_html=True)
        elif a_status == "agent2_done":
            snip1 = a_results.get("sensor_text", "")[:80] + "..."
            snip2 = a_results.get("diagnosis_text", "")[:80] + "..."
            st.markdown(f'<div class="agent-mini"><div class="am-title">&#10003; Sensor Intel</div>'
                        f'<div class="am-body">{snip1}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-mini"><div class="am-title">&#10003; Diagnosis</div>'
                        f'<div class="am-body">{snip2}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-mini"><div class="am-title">Agent 3 — Action</div>'
                        '<div class="am-run">Generating work order...</div></div>', unsafe_allow_html=True)
        elif a_status == "done":
            wo = a_results.get("work_order", {})
            snip1 = a_results.get("sensor_text", "")[:80] + "..."
            snip2 = a_results.get("diagnosis_text", "")[:80] + "..."
            st.markdown(f'<div class="agent-mini"><div class="am-title">&#10003; Sensor Intel</div>'
                        f'<div class="am-body">{snip1}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-mini"><div class="am-title">&#10003; Diagnosis</div>'
                        f'<div class="am-body">{snip2}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-mini" style="border-color:#00C851;">'
                        f'<div class="am-title" style="color:#00C851;">&#10003; Work Order Created</div>'
                        f'<div class="am-body">{wo.get("wo_number","—")}&nbsp;|&nbsp;'
                        f'{wo.get("priority","—")} priority</div></div>', unsafe_allow_html=True)
            st.caption("See AI Analysis tab for full details.")

    st.divider()
    st.markdown("**Asset**")
    st.caption("Chiller Unit A1 — Building Main")
    st.caption("Site: MAIN  |  Install: 2018")
    st.number_input("Energy Rate ($/kWh)", value=0.12, step=0.01, format="%.3f")

# ── Advance simulation ────────────────────────────────────────────────────────
tick(df)
window  = get_window(df, WINDOW_ROWS)
current = get_current(df)

# ── Live anomaly detection ────────────────────────────────────────────────────
log    = st.session_state.sim_event_log
alert  = detect(window)
if alert and not is_duplicate(alert, log):
    push_alert(alert)
    alert_id = str(alert["timestamp"])
    if st.session_state.sim_agent_alert_id != alert_id:
        st.session_state.sim_agent_alert_id = alert_id
        st.session_state.sim_agent_status   = "pending"
        st.session_state.sim_agent_results  = {"alert": alert}
        st.session_state.sim_playing        = False

active_alert = st.session_state.sim_active_alert

# ── Agent step execution (one agent per rerun) ────────────────────────────────
a_status = st.session_state.sim_agent_status

try:
    if a_status == "pending":
        stored_alert = st.session_state.sim_agent_results.get("alert", active_alert)
        with st.spinner("Agent 1 — Sensor Intelligence: analyzing sensor cross-correlations..."):
            text = run_sensor_agent(stored_alert, window, current)
        st.session_state.sim_agent_results["sensor_text"] = text
        st.session_state.sim_agent_status = "agent1_done"
        st.rerun()

    elif a_status == "agent1_done":
        stored_alert = st.session_state.sim_agent_results.get("alert", active_alert)
        with st.spinner("Agent 2 — Diagnosis: ranking root causes..."):
            text = run_diagnosis_agent(
                stored_alert,
                st.session_state.sim_agent_results["sensor_text"],
                current, window,
            )
        st.session_state.sim_agent_results["diagnosis_text"] = text
        st.session_state.sim_agent_status = "agent2_done"
        st.rerun()

    elif a_status == "agent2_done":
        stored_alert = st.session_state.sim_agent_results.get("alert", active_alert)
        with st.spinner("Agent 3 — Action: generating work order..."):
            action_text, work_order = run_action_agent(
                stored_alert,
                st.session_state.sim_agent_results["diagnosis_text"],
                current,
            )
        st.session_state.sim_agent_results["action_text"]         = action_text
        st.session_state.sim_agent_results["work_order"]          = work_order
        st.session_state.sim_agent_results["work_order_approved"] = False
        st.session_state.sim_agent_status = "done"
        st.rerun()

except Exception as e:
    st.session_state.sim_agent_status = "idle"
    st.error(f"Agent pipeline error: {e}")

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
    <div class="syncai-tag">
      Industrial AI Asset Intelligence
      &nbsp;&nbsp;&#9679;&nbsp;&nbsp;
      <span class="sim-status-dot" style="background:{status_color};"></span>
      <span style="color:{status_color};font-weight:600;">{status_str}</span>
      &nbsp;&nbsp;&#9679;&nbsp;&nbsp;
      Chiller Unit A1 &mdash; Site MAIN
    </div>
  </div>
  <div class="sim-ticker">
    <div style="color:{play_color};font-weight:700;font-size:10px;letter-spacing:1px;margin-bottom:3px;">
      {'&#9654; STREAMING' if st.session_state.sim_playing else '&#9646;&#9646; PAUSED'}
    </div>
    <div style="font-size:10px;color:var(--text-dim);letter-spacing:0.5px;">SIM TIME</div>
    <div class="ticker-time">{sim_time_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Alert banner ──────────────────────────────────────────────────────────────
if active_alert:
    sev        = active_alert["severity"]
    cls        = {"Critical": "alert-critical", "Warning": "alert-warning"}.get(sev, "alert-watch")
    icon_emoji = {"Critical": "🔴", "Warning": "🟠", "Watch": "🟡"}.get(sev, "⚡")
    pill_cls   = {"Critical": "pill-crit",   "Warning": "pill-warn"}.get(sev, "pill-watch")
    sev_label  = sev.upper()
    agents_note = (
        ' &nbsp;<span style="color:#FFD700;font-weight:700;">&#9889; Agents activated</span>'
        if st.session_state.sim_agent_status != "idle" else ""
    )
    st.markdown(f"""
    <div class="alert-banner {cls}">
      <div class="alert-icon">{icon_emoji}</div>
      <div>
        <div class="alert-title">
          <span class="alert-pill {pill_cls}">{sev_label}</span>
          &nbsp;{active_alert['sensor'].replace('_',' ').upper()}
          &nbsp;&mdash;&nbsp;z&thinsp;=&thinsp;<strong>{active_alert['z_score']}&thinsp;&sigma;</strong>
          &nbsp;|&nbsp; <span style="font-family:var(--font-mono);">{active_alert['value']} W</span>
        </div>
        <div class="alert-detail">
          {active_alert['failure_mode']} &mdash; {active_alert['action']}
          &nbsp;&nbsp;&#9679;&nbsp;&nbsp;
          Fault Code: <strong>{active_alert['fault_code']}</strong>
          {agents_note}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_ai, tab_flow = st.tabs(["📊 Live Monitor", "🤖 AI Analysis", "🔗 Agent Pipeline"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Live Monitor
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    power_val = float(current.get("active_power", 0))
    hp_val    = float(current.get("high_pressure_1", 0))
    lp_val    = float(current.get("low_pressure_1", 0))
    pr_val    = round(hp_val / lp_val, 2) if lp_val > 0 else 0
    n_events  = len(log)

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

    st.markdown('<div class="section-label">Live Sensor Gauges</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(
            value=power_val, title="Active Power", unit="W",
            min_val=0, max_val=60, green_max=30, yellow_max=45,
            baseline=float(window["active_power"].mean()),
        ), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(
            value=hp_val, title="High Pressure", unit="bar",
            min_val=8, max_val=36, green_max=22, yellow_max=28,
            baseline=float(window["high_pressure_1"].mean()),
        ), use_container_width=True)
    with g3:
        st.plotly_chart(gauge_chart(
            value=lp_val, title="Low Pressure", unit="bar",
            min_val=6, max_val=26, green_max=16, yellow_max=20,
            baseline=float(window["low_pressure_1"].mean()),
        ), use_container_width=True)

    st.markdown('<div class="section-label">Live Sensor Stream — 12hr Rolling Window</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        scrolling_chart(window, active_alert, event_log=log if log else None),
        use_container_width=True,
    )

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

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — AI Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    a_status  = st.session_state.sim_agent_status
    a_results = st.session_state.sim_agent_results

    if a_status == "idle":
        st.info("No active incident. Press PLAY and wait for the anomaly — agents will activate automatically.")

    if "sensor_text" in a_results:
        st.markdown(
            '<div class="agent-card">'
            '<div class="ac-header">'
            '<span class="ac-icon">🔍</span>'
            '<div class="ac-title">Agent 1 — Sensor Intelligence</div>'
            '<span class="ac-status">&#10003;&nbsp;Complete</span>'
            '</div>'
            f'<div class="ac-body">{a_results["sensor_text"]}</div>'
            '</div>', unsafe_allow_html=True)
    elif a_status == "pending":
        st.info("🔍 Agent 1 — Sensor Intelligence: analyzing sensor cross-correlations...")

    if "diagnosis_text" in a_results:
        st.markdown(
            '<div class="agent-card diagnosis">'
            '<div class="ac-header">'
            '<span class="ac-icon">🔬</span>'
            '<div class="ac-title">Agent 2 — Diagnosis</div>'
            '<span class="ac-status">&#10003;&nbsp;Complete</span>'
            '</div>'
            f'<div class="ac-body">{a_results["diagnosis_text"]}</div>'
            '</div>', unsafe_allow_html=True)
    elif a_status == "agent1_done":
        st.info("🔬 Agent 2 — Diagnosis: ranking root causes with confidence scores...")

    if "action_text" in a_results:
        st.markdown(
            '<div class="agent-card action">'
            '<div class="ac-header">'
            '<span class="ac-icon">📋</span>'
            '<div class="ac-title">Agent 3 — Action &amp; Work Order</div>'
            '<span class="ac-status">&#10003;&nbsp;Complete</span>'
            '</div>'
            f'<div class="ac-body">{a_results["action_text"]}</div>'
            '</div>', unsafe_allow_html=True)

        wo = a_results.get("work_order", {})
        if wo:
            priority = wo.get("priority", "High")
            st.markdown(f"""
            <div class="wo-card">
              <div class="wo-header">
                <div>
                  <div class="wo-num">{wo.get('wo_number','—')}</div>
                  <div class="wo-title">{wo.get('title','')}</div>
                  <div class="wo-meta">
                    {wo.get('asset','')} &nbsp;|&nbsp; {wo.get('site','')}
                    &nbsp;|&nbsp; Created: {wo.get('created_at','')}
                    &nbsp;|&nbsp; Est. {wo.get('estimated_hours',0)} hrs
                  </div>
                </div>
                <div class="wo-badge {priority}">{priority}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            woc1, woc2 = st.columns(2)
            with woc1:
                st.markdown("**Maintenance Steps**")
                for i, step in enumerate(wo.get("steps", []), 1):
                    st.markdown(f"{i}. {step}")
                if wo.get("safety_notes"):
                    st.markdown("**Safety Notes**")
                    for note in wo["safety_notes"]:
                        st.markdown(f"⚠️ {note}")
            with woc2:
                st.markdown("**Parts Required**")
                parts = wo.get("parts", [])
                if parts:
                    st.dataframe(pd.DataFrame(parts), use_container_width=True, hide_index=True)

            st.divider()
            if a_results.get("work_order_approved"):
                st.success(f"Work Order {wo.get('wo_number','—')} approved and dispatched.")
            else:
                if st.button("Approve & Dispatch Work Order", type="primary", key="approve_wo"):
                    st.session_state.sim_agent_results["work_order_approved"] = True
                    st.rerun()

    elif a_status == "agent2_done":
        st.info("Agent 3 — Action: generating work order...")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Agent Pipeline Flow
# ══════════════════════════════════════════════════════════════════════════════
with tab_flow:
    a_status  = st.session_state.sim_agent_status
    a_results = st.session_state.sim_agent_results

    st.markdown('<div class="section-label">Live Agent Pipeline — Event → Analysis → Decision → Action</div>',
                unsafe_allow_html=True)

    components.html(agent_flow_html(a_status, a_results), height=520, scrolling=False)

    if a_status != "idle":
        st.divider()
        cs1, cs2, cs3 = st.columns(3)
        done1 = a_status in ("agent1_done", "agent2_done", "done")
        done2 = a_status in ("agent2_done", "done")
        done3 = a_status == "done"
        wo_num = a_results.get("work_order", {}).get("wo_number", "")
        with cs1:
            st.metric("Agent 1 — Sensor Intel",
                      "Complete" if done1 else "Running..." if a_status == "pending" else "Standby")
        with cs2:
            st.metric("Agent 2 — Diagnosis",
                      "Complete" if done2 else "Running..." if a_status == "agent1_done" else "Standby")
        with cs3:
            st.metric("Agent 3 — Work Order",
                      f"Created: {wo_num}" if done3 else "Running..." if a_status == "agent2_done" else "Standby")

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
  .sim-ticker  { text-align: right; font-size: 12px; color: #8BA3C7; font-family: monospace; }
  .sim-ticker b { color: #00C2FF; font-size: 14px; }

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
    0%,100% { border-color:#FF4444; box-shadow:0 0 0 0 rgba(255,68,68,0); }
    50%      { border-color:#FF8888; box-shadow:0 0 8px 4px rgba(255,68,68,0.3); }
  }
  .alert-title  { font-size:15px; font-weight:700; color:#fff; }
  .alert-detail { font-size:12px; color:#8BA3C7; margin-top:4px; }

  .section-label {
    font-size:11px; font-weight:600; color:#8BA3C7;
    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;
  }

  .ev-badge {
    background:#0D2348; border:1px solid #1A3A6B; border-radius:6px;
    padding:6px 10px; margin-bottom:6px; font-size:11px; color:#8BA3C7;
  }
  .ev-badge .ev-sensor { color:#fff; font-weight:600; }
  .ev-badge .ev-crit   { color:#FF4444; }
  .ev-badge .ev-warn   { color:#FF8C00; }
  .ev-badge .ev-watch  { color:#FFD700; }

  .agent-mini {
    background:#0D2348; border:1px solid #1A3A6B; border-radius:8px;
    padding:10px 12px; margin-bottom:8px; font-size:11px;
  }
  .agent-mini .am-title { color:#00C2FF; font-weight:700; margin-bottom:4px; }
  .agent-mini .am-body  { color:#8BA3C7; line-height:1.4; }
  .agent-mini .am-run   { color:#FFD700; font-style:italic; }

  .agent-card {
    background:#0D2348; border:1px solid #1A3A6B;
    border-left:4px solid #00C2FF;
    border-radius:8px; padding:18px 20px; margin-bottom:18px;
  }
  .agent-card.diagnosis { border-left-color:#FF8C00; }
  .agent-card.action    { border-left-color:#00C851; }
  .ac-title { font-size:13px; font-weight:700; color:#fff;
              text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px; }
  .ac-body  { font-size:13px; color:#C8D8F0; line-height:1.7; white-space:pre-wrap; }

  .wo-card {
    background:linear-gradient(135deg,#0D2A1A,#0A1F14);
    border:1px solid #00C851; border-radius:10px; padding:20px; margin-bottom:16px;
  }
  .wo-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }
  .wo-num    { font-size:20px; font-weight:800; color:#00C2FF; }
  .wo-badge  { font-size:11px; font-weight:700; padding:4px 10px; border-radius:6px; }
  .wo-badge.High      { background:#2A1500; color:#FF8C00; border:1px solid #FF8C00; }
  .wo-badge.Emergency { background:#3D0A0A; color:#FF4444; border:1px solid #FF4444; }
  .wo-badge.Medium    { background:#1A1600; color:#FFD700; border:1px solid #FFD700; }
  .wo-title  { font-size:15px; font-weight:700; color:#fff; margin-bottom:4px; }
  .wo-meta   { font-size:11px; color:#8BA3C7; }

  .stButton > button { border-radius:6px; font-weight:600; font-size:13px; }
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
    <div class="syncai-tag">Industrial AI Asset Intelligence &nbsp;|&nbsp;
      <span style="color:{status_color};">{status_str}</span>
    </div>
  </div>
  <div class="sim-ticker">
    <span style="color:{play_color};font-weight:700;">{play_icon}</span><br>
    SIM TIME<br><b>{sim_time_str}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Alert banner ──────────────────────────────────────────────────────────────
if active_alert:
    sev  = active_alert["severity"]
    cls  = {"Critical": "alert-critical", "Warning": "alert-warning"}.get(sev, "alert-watch")
    icon = {"Critical": "CRITICAL ANOMALY", "Warning": "WARNING", "Watch": "WATCH"}.get(sev, "ALERT")
    agents_note = (' &nbsp;|&nbsp; <b style="color:#FFD700;">&#9889; Agents activated</b>'
                   if st.session_state.sim_agent_status != "idle" else "")
    st.markdown(f"""
    <div class="{cls}">
      <div class="alert-title">
        {icon} &nbsp;|&nbsp; {active_alert['sensor'].replace('_',' ').upper()}
        &nbsp;|&nbsp; z = {active_alert['z_score']} sigma
        &nbsp;|&nbsp; {active_alert['value']} W
      </div>
      <div class="alert-detail">
        {active_alert['failure_mode']} &mdash; {active_alert['action']}
        &nbsp;|&nbsp; Fault Code: {active_alert['fault_code']}{agents_note}
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
    st.plotly_chart(scrolling_chart(window, active_alert), use_container_width=True)

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
        st.markdown('<div class="agent-card">'
                    '<div class="ac-title">🔍 Agent 1 — Sensor Intelligence</div>'
                    f'<div class="ac-body">{a_results["sensor_text"]}</div>'
                    '</div>', unsafe_allow_html=True)
    elif a_status == "pending":
        st.info("Agent 1 — Sensor Intelligence: analyzing sensors...")

    if "diagnosis_text" in a_results:
        st.markdown('<div class="agent-card diagnosis">'
                    '<div class="ac-title">🔬 Agent 2 — Diagnosis</div>'
                    f'<div class="ac-body">{a_results["diagnosis_text"]}</div>'
                    '</div>', unsafe_allow_html=True)
    elif a_status == "agent1_done":
        st.info("Agent 2 — Diagnosis: ranking root causes...")

    if "action_text" in a_results:
        st.markdown('<div class="agent-card action">'
                    '<div class="ac-title">📋 Agent 3 — Action</div>'
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

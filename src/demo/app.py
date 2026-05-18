"""
Sync.ai — HVAC Asset Intelligence Demo
Run: streamlit run src/demo/app.py
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load .env from repo root (two levels up from src/demo/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sync.ai — Asset Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand styles ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global font */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Header bar */
  .syncai-header {
    background: linear-gradient(90deg, #0A1628 0%, #0D2348 100%);
    padding: 18px 28px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    border-left: 4px solid #00C2FF;
  }
  .syncai-logo { font-size: 26px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; }
  .syncai-logo span { color: #00C2FF; }
  .syncai-tagline { font-size: 13px; color: #8BA3C7; margin-left: 16px; margin-top: 4px; }

  /* KPI cards */
  .kpi-card {
    background: #0D2348;
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #1A3A6B;
    text-align: center;
    height: 110px;
    display: flex; flex-direction: column; justify-content: center;
  }
  .kpi-label { font-size: 12px; color: #8BA3C7; text-transform: uppercase; letter-spacing: 0.8px; }
  .kpi-value { font-size: 28px; font-weight: 700; color: #FFFFFF; margin: 4px 0; }
  .kpi-sub   { font-size: 12px; color: #00C2FF; }

  /* Status badges */
  .badge-critical { background:#FF4444; color:#fff; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
  .badge-warning  { background:#FF8C00; color:#fff; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
  .badge-watch    { background:#FFD700; color:#000; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
  .badge-normal   { background:#00C851; color:#fff; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }

  /* Work order card */
  .wo-card {
    background: #0D2348; border: 1px solid #1A3A6B; border-left: 4px solid #00C2FF;
    border-radius: 8px; padding: 16px 20px; margin-top: 12px;
  }
  .wo-title { font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px; }
  .wo-row   { font-size: 13px; color: #8BA3C7; margin: 3px 0; }
  .wo-row b { color: #FFFFFF; }

  /* Chat bubbles */
  .chat-user { background:#1A3A6B; border-radius:12px 12px 4px 12px; padding:12px 16px; margin:6px 0; color:#fff; }
  .chat-ai   { background:#0D2348; border:1px solid #1A3A6B; border-radius:12px 12px 12px 4px; padding:12px 16px; margin:6px 0; color:#e0e8f5; }

  /* Section divider */
  hr { border-color: #1A3A6B; }

  /* Streamlit overrides */
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
  .stMetric label { font-size: 12px !important; color: #8BA3C7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).parent
RESULTS_DIR = DEMO_DIR / "results"
ANOMALY_FILE = RESULTS_DIR / "anomalies.csv"
FORECAST_FILE = RESULTS_DIR / "forecast.json"

# ── FMSR static failure mode map ──────────────────────────────────────────────

FAILURE_MODE_MAP = {
    "active_power": {
        "failure_mode": "Compressor Stress / Electrical Fault",
        "description": "Abnormal power draw indicates compressor overload, electrical imbalance, or refrigerant-side restriction forcing the compressor to work harder than design limits.",
        "action": "Inspect compressor windings and capacitors. Check supply voltage balance. Review refrigerant charge.",
        "fault_code": "EL-020",
        "priority": "High",
    },
    "high_pressure_1": {
        "failure_mode": "Refrigerant Overcharge / Condenser Fouling",
        "description": "Elevated high-side pressure suggests excess refrigerant, blocked condenser coils, or non-condensable gases in the system reducing heat rejection capacity.",
        "action": "Inspect and clean condenser coils. Verify refrigerant charge. Purge non-condensables if present.",
        "fault_code": "RF-010",
        "priority": "High",
    },
    "low_pressure_1": {
        "failure_mode": "Refrigerant Leak / Evaporator Issue",
        "description": "Low suction pressure indicates refrigerant loss, restricted liquid line, or evaporator airflow blockage reducing the system's cooling capacity.",
        "action": "Perform leak detection on all joints and valves. Check liquid line filter-drier. Inspect evaporator coil and fan.",
        "fault_code": "RF-005",
        "priority": "Critical",
    },
    "outlet_temp": {
        "failure_mode": "Coil Fouling / Airflow Restriction",
        "description": "Abnormal discharge temperature points to dirty coils, blocked air filters, or fan degradation limiting heat transfer at the air-side of the system.",
        "action": "Clean or replace air filters. Clean coil surfaces. Check fan motor amp draw and blade condition.",
        "fault_code": "MT-030",
        "priority": "Medium",
    },
}

PRIORITY_COLOR = {"Critical": "#FF4444", "High": "#FF8C00", "Medium": "#FFD700", "Low": "#00C851"}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not ANOMALY_FILE.exists():
        st.error("results/anomalies.csv not found. Run `python src/demo/precompute.py` first.")
        st.stop()
    df = pd.read_csv(ANOMALY_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data
def load_forecast():
    if not FORECAST_FILE.exists():
        return None
    with open(FORECAST_FILE) as f:
        return json.load(f)


def make_sensor_chart(df: pd.DataFrame, title: str = "", height: int = 380) -> go.Figure:
    """Multi-axis sensor time-series chart with anomaly overlay."""
    anomalies = df[df["anomaly_label"] == 1]

    fig = go.Figure()

    # Active power (primary y)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["active_power"],
        name="Active Power (W)", line=dict(color="#00C2FF", width=1.5),
        yaxis="y1", hovertemplate="%{x}<br>Power: %{y:.1f} W<extra></extra>",
    ))

    # High pressure
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["high_pressure_1"],
        name="High Pressure (bar)", line=dict(color="#FF8C00", width=1, dash="dot"),
        yaxis="y2", hovertemplate="%{x}<br>HP: %{y:.2f} bar<extra></extra>",
    ))

    # Low pressure
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["low_pressure_1"],
        name="Low Pressure (bar)", line=dict(color="#9B59B6", width=1, dash="dot"),
        yaxis="y2", hovertemplate="%{x}<br>LP: %{y:.2f} bar<extra></extra>",
    ))

    # Anomaly dots
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["active_power"],
            mode="markers", name="Anomaly",
            marker=dict(color="#FF4444", size=7, symbol="circle", line=dict(color="#fff", width=1)),
            yaxis="y1", hovertemplate="%{x}<br>ANOMALY<br>Power: %{y:.1f} W<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=14)) if title else None,
        paper_bgcolor="#0A1628", plot_bgcolor="#0D1F3C",
        font=dict(color="#8BA3C7"),
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        height=height,
        margin=dict(l=10, r=10, t=30 if title else 10, b=40),
        yaxis=dict(title="Power (W)", gridcolor="#1A3A6B", color="#00C2FF"),
        yaxis2=dict(title="Pressure (bar)", overlaying="y", side="right", gridcolor="#1A3A6B", color="#FF8C00"),
        xaxis=dict(gridcolor="#1A3A6B"),
        hovermode="x unified",
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="syncai-header">
  <div>
    <div class="syncai-logo">Sync<span>.ai</span></div>
    <div class="syncai-tagline">Industrial AI Asset Intelligence Platform</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Asset Configuration")
    asset_name = st.selectbox("Asset", ["Chiller Unit A1 — Building Main"])
    st.caption("Site: MAIN  |  Type: HVAC Chiller  |  Install: 2018")
    st.divider()

    df_full = load_data()
    date_min = df_full["timestamp"].min().date()
    date_max = df_full["timestamp"].max().date()

    default_start = date_max - timedelta(days=29)
    date_range = st.date_input(
        "Date Range",
        value=(default_start, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, date_max

    st.divider()
    energy_rate = st.number_input("Energy Rate ($/kWh)", value=0.12, step=0.01, format="%.3f")
    st.divider()
    st.markdown("### Data Quality")
    st.success(f"26,208 rows — 0 nulls")
    st.caption("Jun 01 → Aug 30, 2022  |  5-min intervals")

# ── Filter by date ────────────────────────────────────────────────────────────

mask = (df_full["timestamp"].dt.date >= start_date) & (df_full["timestamp"].dt.date <= end_date)
df = df_full[mask].copy()
forecast = load_forecast()

# ── Pre-compute summary stats ─────────────────────────────────────────────────

latest = df.iloc[-1]
anomaly_count = int(df["anomaly_label"].sum())
anomalies_df = df[df["anomaly_label"] == 1].copy()
pressure_balance = (df["high_pressure_1"].mean() / df["low_pressure_1"].mean()).round(2) if df["low_pressure_1"].mean() > 0 else 0
system_on = bool(latest.get("on_off", 0) == 1)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Live Dashboard",
    "🔴  Anomaly Intelligence",
    "⚡  Energy Forecast",
    "🤖  Sync.ai Assistant",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    status_color = "#00C851" if system_on else "#8BA3C7"
    status_text = "RUNNING" if system_on else "STANDBY"

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">System Status</div>
          <div class="kpi-value" style="color:{status_color};">{status_text}</div>
          <div class="kpi-sub">as of {latest['timestamp'].strftime('%H:%M %d %b')}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        power_w = latest["active_power"]
        power_kw = power_w / 1000 if power_w > 100 else power_w
        unit = "kW" if power_w > 100 else "W"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Active Power</div>
          <div class="kpi-value">{power_kw:.1f}<span style="font-size:16px;color:#8BA3C7;"> {unit}</span></div>
          <div class="kpi-sub">avg {df['active_power'].mean():.1f} {unit} over period</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        severity_counts = anomalies_df["severity"].value_counts()
        critical = severity_counts.get("Critical", 0)
        warning = severity_counts.get("Warning", 0)
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Anomalies Detected</div>
          <div class="kpi-value" style="color:{'#FF4444' if anomaly_count > 0 else '#00C851'};">{anomaly_count}</div>
          <div class="kpi-sub">{critical} critical · {warning} warning</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        pb_color = "#FF4444" if pressure_balance > 1.6 else ("#FFD700" if pressure_balance > 1.4 else "#00C851")
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Pressure Ratio</div>
          <div class="kpi-value" style="color:{pb_color};">{pressure_balance:.2f}</div>
          <div class="kpi-sub">high ÷ low (ideal: 1.2–1.4)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main sensor chart
    st.plotly_chart(
        make_sensor_chart(df, title="Sensor Overview — Active Power & Pressure with Anomaly Events"),
        use_container_width=True,
    )

    # Outlet vs Inlet temperature
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df["timestamp"], y=df["outlet_temp"], name="Outlet Temp", line=dict(color="#FF8C00", width=1.5)))
    fig_temp.add_trace(go.Scatter(x=df["timestamp"], y=df["inlet_temp"], name="Inlet Temp", line=dict(color="#9B59B6", width=1.5)))
    fig_temp.add_trace(go.Scatter(x=df["timestamp"], y=df["ambient_temp"], name="Ambient Temp", line=dict(color="#00C2FF", width=1, dash="dash")))
    fig_temp.add_trace(go.Scatter(x=df["timestamp"], y=df["summer_SP_temp"], name="Summer Setpoint", line=dict(color="#FF4444", width=1, dash="dot")))
    fig_temp.update_layout(
        title=dict(text="Temperature Sensors vs Setpoint", font=dict(color="#FFFFFF", size=14)),
        paper_bgcolor="#0A1628", plot_bgcolor="#0D1F3C",
        font=dict(color="#8BA3C7"), height=280, margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="°C", gridcolor="#1A3A6B"),
        xaxis=dict(gridcolor="#1A3A6B"),
        legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_temp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if anomalies_df.empty:
        st.success("No anomalies detected in the selected date range.")
    else:
        st.markdown(f"**{anomaly_count} anomaly events** detected in selected period — sorted by severity.")

        # Build display table
        display_cols = ["timestamp", "anomaly_sensor", "z_score", "severity", "active_power", "high_pressure_1", "low_pressure_1"]
        available = [c for c in display_cols if c in anomalies_df.columns]
        table_df = anomalies_df[available].copy()
        table_df = table_df.rename(columns={
            "timestamp": "Timestamp", "anomaly_sensor": "Sensor",
            "z_score": "Z-Score", "severity": "Severity",
            "active_power": "Power (W)", "high_pressure_1": "HP (bar)", "low_pressure_1": "LP (bar)",
        })
        severity_order = {"Critical": 0, "Warning": 1, "Watch": 2}
        table_df["_sord"] = table_df["Severity"].map(severity_order).fillna(3)
        table_df = table_df.sort_values(["_sord", "Timestamp"]).drop(columns=["_sord"]).reset_index(drop=True)

        # Deduplicate into events (group consecutive anomaly rows into single events)
        anomalies_df_sorted = anomalies_df.sort_values("timestamp").copy()
        anomalies_df_sorted["event_id"] = (anomalies_df_sorted["timestamp"].diff() > pd.Timedelta("30min")).cumsum()
        events = (
            anomalies_df_sorted.groupby("event_id")
            .agg(
                start=("timestamp", "first"),
                end=("timestamp", "last"),
                sensor=("anomaly_sensor", lambda x: x.mode()[0] if not x.mode().empty else ""),
                severity=("severity", lambda x: x.mode()[0] if not x.mode().empty else "Watch"),
                max_z=("z_score", "max"),
                count=("timestamp", "count"),
            )
            .reset_index(drop=True)
        )

        severity_order_map = {"Critical": 0, "Warning": 1, "Watch": 2}
        events["_s"] = events["severity"].map(severity_order_map).fillna(3)
        events = events.sort_values(["_s", "start"]).drop(columns=["_s"]).reset_index(drop=True)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("#### Anomaly Events")
            selected_idx = st.radio(
                "Select event to inspect:",
                options=events.index.tolist(),
                format_func=lambda i: f"{events.loc[i,'start'].strftime('%b %d %H:%M')} · {events.loc[i,'severity']} · {events.loc[i,'sensor']}",
                label_visibility="collapsed",
            )

        with col_right:
            ev = events.loc[selected_idx]
            ev_start = ev["start"] - timedelta(hours=2)
            ev_end = ev["end"] + timedelta(hours=2)
            zoom_df = df[(df["timestamp"] >= ev_start) & (df["timestamp"] <= ev_end)]

            fig_zoom = make_sensor_chart(zoom_df, title=f"Event Window — {ev['start'].strftime('%b %d %H:%M')} to {ev['end'].strftime('%H:%M')}", height=300)
            # Shade the event window
            fig_zoom.add_vrect(
                x0=ev["start"], x1=ev["end"],
                fillcolor="#FF4444", opacity=0.12, layer="below", line_width=0,
            )
            st.plotly_chart(fig_zoom, use_container_width=True)

        st.divider()

        # Root cause panel
        sensor = ev.get("sensor", "")
        fm = FAILURE_MODE_MAP.get(sensor, {
            "failure_mode": "Sensor Anomaly",
            "description": "Unusual reading detected. Manual inspection recommended.",
            "action": "Review sensor calibration and recent maintenance logs.",
            "fault_code": "GN-001",
            "priority": "Medium",
        })
        pri_color = PRIORITY_COLOR.get(fm["priority"], "#8BA3C7")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### AI Root Cause Analysis")
            st.markdown(f"""
            <div style="background:#0D2348;border:1px solid #1A3A6B;border-left:4px solid {pri_color};border-radius:8px;padding:16px 20px;">
              <div style="font-size:16px;font-weight:700;color:#FFFFFF;margin-bottom:6px;">{fm['failure_mode']}</div>
              <div style="font-size:13px;color:#8BA3C7;margin-bottom:10px;">{fm['description']}</div>
              <div style="font-size:12px;color:#00C2FF;margin-bottom:4px;">RECOMMENDED ACTION</div>
              <div style="font-size:13px;color:#FFFFFF;">{fm['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("#### Generate Work Order")
            if st.button("Create Work Order", type="primary"):
                wo_id = f"WO-{uuid.uuid4().hex[:6].upper()}"
                due_date = (ev["start"] + timedelta(days=3)).strftime("%Y-%m-%d")
                st.markdown(f"""
                <div class="wo-card">
                  <div class="wo-title">Work Order Generated ✓</div>
                  <div class="wo-row"><b>WO ID:</b> {wo_id}</div>
                  <div class="wo-row"><b>Asset:</b> Chiller Unit A1 — Building Main</div>
                  <div class="wo-row"><b>Fault Code:</b> {fm['fault_code']}</div>
                  <div class="wo-row"><b>Failure Mode:</b> {fm['failure_mode']}</div>
                  <div class="wo-row"><b>Priority:</b> <span style="color:{pri_color};">{fm['priority']}</span></div>
                  <div class="wo-row"><b>Detected:</b> {ev['start'].strftime('%Y-%m-%d %H:%M')}</div>
                  <div class="wo-row"><b>Recommended Completion:</b> {due_date}</div>
                  <div class="wo-row"><b>Action:</b> {fm['action']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Click to auto-generate a maintenance work order for this event.")

        st.divider()
        st.markdown("#### All Anomaly Rows")
        st.dataframe(table_df.head(200), use_container_width=True, height=240)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENERGY FORECAST
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    col1, col2 = st.columns([3, 1])

    with col1:
        fig_fc = go.Figure()

        # Historical — last 7 days
        hist_start = df["timestamp"].max() - timedelta(days=7)
        hist = df[df["timestamp"] >= hist_start]
        fig_fc.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["active_power"],
            name="Historical Power", line=dict(color="#8BA3C7", width=1.5),
            hovertemplate="%{x}<br>Actual: %{y:.1f} W<extra></extra>",
        ))

        if forecast:
            fc_ts = pd.to_datetime(forecast["timestamps"])
            fc_pred = np.array(forecast["predicted"])
            fc_upper = np.array(forecast["upper"])
            fc_lower = np.array(forecast["lower"])

            # Scale to watts if values look like kW
            if fc_pred.mean() < 100:
                fc_pred = fc_pred * 1000
                fc_upper = fc_upper * 1000
                fc_lower = fc_lower * 1000

            # Confidence band
            fig_fc.add_trace(go.Scatter(
                x=np.concatenate([fc_ts, fc_ts[::-1]]),
                y=np.concatenate([fc_upper, fc_lower[::-1]]),
                fill="toself", fillcolor="rgba(0,194,255,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Band",
                hoverinfo="skip",
            ))

            # Forecast line
            fig_fc.add_trace(go.Scatter(
                x=fc_ts, y=fc_pred,
                name="AI Forecast", line=dict(color="#00C2FF", width=2, dash="dash"),
                hovertemplate="%{x}<br>Forecast: %{y:.1f} W<extra></extra>",
            ))

            # Vertical divider at forecast start (add_vline broken in Plotly 6 with datetime axis)
            fig_fc.add_shape(
                type="line", xref="x", yref="paper",
                x0=str(fc_ts[0]), x1=str(fc_ts[0]), y0=0, y1=1,
                line=dict(color="#00C2FF", width=1, dash="dot"),
            )
            fig_fc.add_annotation(
                x=str(fc_ts[0]), y=1, yref="paper", xanchor="left",
                text="Forecast Start", font=dict(color="#00C2FF", size=11),
                showarrow=False, bgcolor="rgba(0,0,0,0)",
            )

        fig_fc.update_layout(
            title=dict(text="Active Power — Historical & 24hr AI Forecast", font=dict(color="#FFFFFF", size=14)),
            paper_bgcolor="#0A1628", plot_bgcolor="#0D1F3C",
            font=dict(color="#8BA3C7"), height=380, margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title="Power (W)", gridcolor="#1A3A6B"),
            xaxis=dict(gridcolor="#1A3A6B"),
            legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with col2:
        st.markdown("#### Forecast Summary")
        if forecast:
            fc_pred_arr = np.array(forecast["predicted"])
            if fc_pred_arr.mean() < 100:
                fc_pred_arr = fc_pred_arr * 1000

            peak_w = fc_pred_arr.max()
            peak_kw = peak_w / 1000 if peak_w > 100 else peak_w
            kwh_24 = fc_pred_arr.sum() / (12 * 1000) if fc_pred_arr.mean() > 100 else fc_pred_arr.sum() / 12
            cost_24 = kwh_24 * energy_rate
            setpoint_violations = int((df["ambient_temp"] > df["summer_SP_temp"]).sum())

            st.metric("Peak Forecast", f"{peak_kw:.1f} kW")
            st.metric("Predicted kWh (24h)", f"{kwh_24:.1f} kWh")
            st.metric("Est. Energy Cost", f"${cost_24:.2f}")
            st.metric("Setpoint Violations", setpoint_violations, help="Rows where indoor temp exceeded summer setpoint")
        else:
            st.info("Forecast not available. Run precompute.py.")

    st.divider()
    st.markdown("#### Setpoint Compliance")
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(x=df["timestamp"], y=df["ambient_temp"], name="Ambient Temp", line=dict(color="#00C2FF", width=1.5)))
    fig_sp.add_trace(go.Scatter(x=df["timestamp"], y=df["summer_SP_temp"], name="Summer Setpoint", line=dict(color="#FF4444", width=1, dash="dot")))
    fig_sp.update_layout(
        paper_bgcolor="#0A1628", plot_bgcolor="#0D1F3C", font=dict(color="#8BA3C7"),
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="°C", gridcolor="#1A3A6B"), xaxis=dict(gridcolor="#1A3A6B"),
        legend=dict(orientation="h", y=-0.3, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_sp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SYNC.AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    # Build asset context for system prompt
    top_anomalies = anomalies_df.head(3) if not anomalies_df.empty else pd.DataFrame()
    anomaly_summary = ""
    if not top_anomalies.empty:
        for _, row in top_anomalies.iterrows():
            ts = row["timestamp"].strftime("%b %d %H:%M") if pd.notnull(row["timestamp"]) else "unknown"
            sensor = row.get("anomaly_sensor", "unknown")
            sev = row.get("severity", "")
            anomaly_summary += f"  - {ts}: {sev} anomaly on {sensor}\n"
    else:
        anomaly_summary = "  - No anomalies in selected period\n"

    sensor_summary = (
        f"active_power: mean={df['active_power'].mean():.1f}W, max={df['active_power'].max():.1f}W | "
        f"high_pressure_1: mean={df['high_pressure_1'].mean():.2f}bar | "
        f"low_pressure_1: mean={df['low_pressure_1'].mean():.2f}bar | "
        f"outlet_temp: mean={df['outlet_temp'].mean():.1f}°C | "
        f"ambient_temp: mean={df['ambient_temp'].mean():.1f}°C"
    )

    SYSTEM_PROMPT = f"""You are Sync.ai, an Industrial AI Asset Management assistant specialising in HVAC systems.

Asset: Chiller Unit A1 — Building Main
Dataset period: {start_date} to {end_date} (5-minute sensor readings)
Total records in period: {len(df):,}
Anomalies detected: {anomaly_count} events

Recent anomaly events:
{anomaly_summary}
Sensor averages:
  {sensor_summary}

Failure mode context:
- High pressure anomalies → Refrigerant Overcharge / Condenser Fouling
- Low pressure anomalies → Refrigerant Leak / Evaporator Issue
- Power anomalies → Compressor Stress / Electrical Fault
- Outlet temp anomalies → Coil Fouling / Airflow Restriction

Your role: Help the asset manager understand equipment health, explain anomalies in plain language,
recommend specific maintenance actions, and quantify cost/risk of inaction.
Be concise, direct, and always ground answers in the sensor data above.
Avoid generic advice — be specific to this chiller's readings."""

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_chat, col_hints = st.columns([3, 1])

    with col_hints:
        st.markdown("#### Example Questions")
        hints = [
            "What caused the most critical anomaly in this period?",
            "When should I schedule the next maintenance?",
            "How much money am I wasting due to the pressure imbalance?",
            "Is the system performing efficiently compared to setpoints?",
            "Summarise the health of this chiller for my operations manager.",
        ]
        for h in hints:
            if st.button(h, key=f"hint_{h[:20]}"):
                st.session_state.messages.append({"role": "user", "content": h})
                st.rerun()

    with col_chat:
        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask Sync.ai about your asset..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                with st.chat_message("assistant"):
                    st.warning("ANTHROPIC_API_KEY not set. Set it in your environment and restart the app.")
            else:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""

                        with client.messages.stream(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system=SYSTEM_PROMPT,
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                        ) as stream:
                            for text in stream.text_stream:
                                full_response += text
                                response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    with st.chat_message("assistant"):
                        st.error(f"Claude API error: {e}")

    if st.session_state.messages:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

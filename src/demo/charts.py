"""
Sync.ai Demo — Chart Builders
All Plotly chart factories used by app_v2.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette ─────────────────────────────────────────────────────────────
BG_PAPER   = "#0A1628"
BG_PLOT    = "#0D1F3C"
GRID_COLOR = "#1A3A6B"
TEXT_COLOR = "#8BA3C7"
BLUE       = "#00C2FF"
ORANGE     = "#FF8C00"
PURPLE     = "#9B59B6"
RED        = "#FF4444"
GREEN      = "#00C851"
YELLOW     = "#FFD700"
WHITE      = "#FFFFFF"

_BASE_LAYOUT = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG_PLOT,
    font=dict(color=TEXT_COLOR, size=11),
    hovermode="x unified",
)


# ── 1. Gauge chart ────────────────────────────────────────────────────────────

def gauge_chart(
    value: float,
    title: str,
    unit: str,
    min_val: float,
    max_val: float,
    green_max: float,
    yellow_max: float,
    baseline: float | None = None,
    height: int = 220,
) -> go.Figure:
    """
    Circular gauge with three colour zones and optional delta vs baseline.
    """
    delta_cfg = dict(reference=baseline, valueformat=".1f") if baseline is not None else None

    # Needle colour based on zone
    if value <= green_max:
        bar_color = GREEN
    elif value <= yellow_max:
        bar_color = ORANGE
    else:
        bar_color = RED

    indicator = go.Indicator(
        mode="gauge+number" + ("+delta" if delta_cfg else ""),
        value=value,
        delta=delta_cfg,
        number=dict(suffix=f" {unit}", font=dict(size=20, color=WHITE)),
        title=dict(text=title, font=dict(size=13, color=TEXT_COLOR)),
        gauge=dict(
            axis=dict(
                range=[min_val, max_val],
                tickcolor=TEXT_COLOR,
                tickfont=dict(size=10),
            ),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor=BG_PLOT,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            steps=[
                dict(range=[min_val, green_max],  color="#0D2A1A"),
                dict(range=[green_max, yellow_max], color="#2A1F0A"),
                dict(range=[yellow_max, max_val], color="#2A0A0A"),
            ],
            threshold=dict(
                line=dict(color=WHITE, width=2),
                thickness=0.75,
                value=baseline if baseline else green_max,
            ),
        ),
    )

    fig = go.Figure(indicator)
    fig.update_layout(**{
        **_BASE_LAYOUT,
        "height": height,
        "margin": dict(l=20, r=20, t=40, b=10),
        "paper_bgcolor": BG_PAPER,
    })
    return fig


# ── 2. Scrolling live chart ───────────────────────────────────────────────────

def scrolling_chart(window_df: pd.DataFrame, active_alert: dict | None = None, height: int = 320) -> go.Figure:
    """
    Main live chart: active_power (filled area) + pressure traces on secondary axis.
    Highlights anomaly rows as red markers. Shows alert band if alert is active.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ts = window_df["timestamp"]

    # Filled area under active_power
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["active_power"],
        name="Active Power (W)",
        fill="tozeroy",
        fillcolor="rgba(0,194,255,0.12)",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x|%H:%M}<br>Power: <b>%{y:.1f} W</b><extra></extra>",
    ), secondary_y=False)

    # High pressure
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["high_pressure_1"],
        name="High Pressure (bar)",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
        hovertemplate="HP: %{y:.2f} bar<extra></extra>",
    ), secondary_y=True)

    # Low pressure
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["low_pressure_1"],
        name="Low Pressure (bar)",
        line=dict(color=PURPLE, width=1.5, dash="dot"),
        hovertemplate="LP: %{y:.2f} bar<extra></extra>",
    ), secondary_y=True)

    # Anomaly markers on active_power
    if "anomaly_label" in window_df.columns:
        anom = window_df[window_df["anomaly_label"] == 1]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom["active_power"],
                mode="markers", name="Anomaly",
                marker=dict(color=RED, size=9, symbol="circle",
                            line=dict(color=WHITE, width=1.5)),
                hovertemplate="%{x|%H:%M}<br>ANOMALY: %{y:.1f} W<extra></extra>",
            ), secondary_y=False)

    # Alert highlight band — last 5 rows
    if active_alert is not None and len(window_df) >= 5:
        band_start = str(window_df["timestamp"].iloc[-5])
        band_end   = str(window_df["timestamp"].iloc[-1])
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=band_start, x1=band_end, y0=0, y1=1,
            fillcolor="rgba(255,68,68,0.15)", line_width=0,
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR, showticklabels=True, tickformat="%H:%M\n%d %b"),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    fig.update_yaxes(title_text="Power (W)",    gridcolor=GRID_COLOR,
                     title_font=dict(color=BLUE),   secondary_y=False)
    fig.update_yaxes(title_text="Pressure (bar)", gridcolor=GRID_COLOR,
                     title_font=dict(color=ORANGE), secondary_y=True, showgrid=False)
    return fig


# ── 3. Temperature panel ──────────────────────────────────────────────────────

def temperature_panel(window_df: pd.DataFrame, height: int = 200) -> go.Figure:
    fig = go.Figure()
    ts = window_df["timestamp"]

    # Fill between inlet and outlet
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([window_df["outlet_temp"], window_df["inlet_temp"][::-1]]),
        fill="toself", fillcolor="rgba(155,89,182,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(x=ts, y=window_df["outlet_temp"],
                             name="Outlet", line=dict(color=ORANGE, width=1.5),
                             hovertemplate="Outlet: %{y:.1f}degC<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts, y=window_df["inlet_temp"],
                             name="Inlet", line=dict(color=PURPLE, width=1.5),
                             hovertemplate="Inlet: %{y:.1f}degC<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts, y=window_df["ambient_temp"],
                             name="Ambient", line=dict(color=BLUE, width=1, dash="dash"),
                             hovertemplate="Ambient: %{y:.1f}degC<extra></extra>"))
    if "summer_SP_temp" in window_df.columns:
        fig.add_trace(go.Scatter(x=ts, y=window_df["summer_SP_temp"],
                                 name="Setpoint", line=dict(color=RED, width=1, dash="dot"),
                                 hovertemplate="Setpoint: %{y:.1f}degC<extra></extra>"))

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Temperature", font=dict(color=TEXT_COLOR, size=12)),
        xaxis=dict(gridcolor=GRID_COLOR, tickformat="%H:%M"),
        yaxis=dict(gridcolor=GRID_COLOR, title="degC"),
        legend=dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    return fig


# ── 4. Pressure history panel ─────────────────────────────────────────────────

def pressure_history_panel(window_df: pd.DataFrame, height: int = 200) -> go.Figure:
    fig = go.Figure()
    ts = window_df["timestamp"]

    # Shaded differential band
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([window_df["high_pressure_1"], window_df["low_pressure_1"][::-1]]),
        fill="toself", fillcolor="rgba(0,194,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(x=ts, y=window_df["high_pressure_1"],
                             name="High P", line=dict(color=ORANGE, width=1.5),
                             hovertemplate="HP: %{y:.2f} bar<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts, y=window_df["low_pressure_1"],
                             name="Low P",  line=dict(color=PURPLE, width=1.5),
                             hovertemplate="LP: %{y:.2f} bar<extra></extra>"))

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Pressure Differential", font=dict(color=TEXT_COLOR, size=12)),
        xaxis=dict(gridcolor=GRID_COLOR, tickformat="%H:%M"),
        yaxis=dict(gridcolor=GRID_COLOR, title="bar"),
        legend=dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    return fig


# ── 5. System status panel ────────────────────────────────────────────────────

def system_status_panel(window_df: pd.DataFrame, height: int = 200) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts = window_df["timestamp"]

    # ON/OFF as filled step chart
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["on_off"],
        name="ON/OFF", fill="tozeroy",
        fillcolor="rgba(0,200,81,0.20)",
        line=dict(color=GREEN, width=1.5, shape="hv"),
        hovertemplate="Status: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    # Damper % as bar
    if "damper" in window_df.columns:
        fig.add_trace(go.Bar(
            x=ts, y=window_df["damper"],
            name="Damper %", marker_color="rgba(0,194,255,0.3)",
            hovertemplate="Damper: %{y:.0f}%%<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="System Status & Damper", font=dict(color=TEXT_COLOR, size=12)),
        xaxis=dict(gridcolor=GRID_COLOR, tickformat="%H:%M"),
        legend=dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=10, r=10, t=30, b=50),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="ON/OFF", range=[-0.1, 1.5],
                     gridcolor=GRID_COLOR, secondary_y=False)
    fig.update_yaxes(title_text="Damper %", range=[0, 120],
                     showgrid=False, secondary_y=True)
    return fig


# ── 6. Humidity / CO2 panel ───────────────────────────────────────────────────

def humidity_co2_panel(window_df: pd.DataFrame, height: int = 200) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts = window_df["timestamp"]

    # Comfort band shading
    fig.add_shape(type="rect", xref="paper", yref="y",
                  x0=0, x1=1, y0=40, y1=60,
                  fillcolor="rgba(0,200,81,0.07)", line_width=0)

    fig.add_trace(go.Scatter(
        x=ts, y=window_df["ambient_humidity"],
        name="Humidity %", fill="tozeroy",
        fillcolor="rgba(0,194,255,0.10)",
        line=dict(color=BLUE, width=1.5),
        hovertemplate="Humidity: %{y:.0f}%%<extra></extra>",
    ), secondary_y=False)

    if "co2_1" in window_df.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=window_df["co2_1"],
            name="CO2 (ppm)", line=dict(color=YELLOW, width=1.5, dash="dot"),
            hovertemplate="CO2: %{y:.0f} ppm<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Humidity & CO2", font=dict(color=TEXT_COLOR, size=12)),
        xaxis=dict(gridcolor=GRID_COLOR, tickformat="%H:%M"),
        legend=dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    fig.update_yaxes(title_text="Humidity %", range=[0, 80],
                     gridcolor=GRID_COLOR, secondary_y=False)
    fig.update_yaxes(title_text="CO2 (ppm)", showgrid=False, secondary_y=True)
    return fig


# ── 7. Agent pipeline flow diagram ───────────────────────────────────────────

def agent_flow_html(agent_status: str, agent_results: dict) -> str:
    """
    Returns a self-contained HTML string rendering the n8n-style agent
    pipeline flow diagram. Node colours and badges update based on agent_status.
    """
    # Map each pipeline stage to a visual state
    def _state(stages_done: list[str]) -> str:
        if agent_status == "idle":
            return "idle"
        if agent_status == "pending" and "sensor" in stages_done:
            return "running"
        if agent_status == "agent1_done" and "diagnosis" in stages_done:
            return "running"
        if agent_status == "agent2_done" and "action" in stages_done:
            return "running"
        completed_map = {
            "sensor":    ["agent1_done", "agent2_done", "done"],
            "diagnosis": ["agent2_done", "done"],
            "action":    ["done"],
            "output":    ["done"],
        }
        for stage in stages_done:
            if agent_status in completed_map.get(stage, []):
                return "done"
        return "idle"

    trigger_state  = "done" if agent_status != "idle" else "idle"
    sensor_state   = _state(["sensor"])
    if agent_status == "pending":
        sensor_state = "running"
    diagnosis_state = _state(["diagnosis"])
    if agent_status == "agent1_done":
        diagnosis_state = "running"
    action_state   = _state(["action"])
    if agent_status == "agent2_done":
        action_state = "running"
    output_state   = "done" if agent_status == "done" else "idle"

    wo = agent_results.get("work_order", {})
    wo_num = wo.get("wo_number", "—")

    # Colour scheme per state
    COLORS = {
        "idle":    {"border": "#1A3A6B", "bg": "#0D1F3C", "label": "#8BA3C7", "icon_bg": "#0A1628"},
        "running": {"border": "#00C2FF", "bg": "#0A2540", "label": "#FFFFFF", "icon_bg": "#0D2A4A"},
        "done":    {"border": "#00C851", "bg": "#0A2015", "label": "#FFFFFF", "icon_bg": "#0D2A1A"},
    }

    def _node(icon: str, title: str, subtitle: str, state: str,
              tools: list[str] | None = None, node_id: str = "") -> str:
        c = COLORS[state]
        pulse_anim = "animation: pulse-border 1.4s ease-in-out infinite;" if state == "running" else ""
        badge = ""
        if state == "running":
            badge = '<div class="spinner"></div>'
        elif state == "done":
            badge = '<span style="color:#00C851;font-size:13px;font-weight:900;">&#10003;</span>'

        tool_chips = ""
        if tools:
            chips_html = "".join(
                f'<div class="tool-chip" style="border-color:{c["border"]};color:{c["label"]};">'
                f'<span style="color:#00C2FF;margin-right:4px;">&#9670;</span>{t}</div>'
                for t in tools
            )
            tool_chips = f'<div class="tool-group">{chips_html}</div>'

        return f"""
        <div class="node-wrap" id="{node_id}">
          <div class="node" style="border-color:{c['border']};background:{c['bg']};{pulse_anim}">
            <div class="node-badge">{badge}</div>
            <div class="node-icon" style="background:{c['icon_bg']};">{icon}</div>
            <div class="node-title" style="color:{c['label']};">{title}</div>
            <div class="node-sub">{subtitle}</div>
          </div>
          {tool_chips}
        </div>"""

    def _arrow(state: str) -> str:
        color = "#00C2FF" if state in ("running", "done") else "#1A3A6B"
        animated = "animation: flow-arrow 1s linear infinite;" if state == "running" else ""
        return f"""
        <div class="arrow-wrap">
          <div class="arrow-line" style="background:{color};{animated}"></div>
          <div class="arrow-head" style="border-left-color:{color};"></div>
        </div>"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0A1628;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 24px 16px;
    min-height: 520px;
  }}

  /* ── Header ── */
  .flow-header {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 28px;
  }}
  .flow-title {{
    font-size: 13px; font-weight: 700; color: #8BA3C7;
    text-transform: uppercase; letter-spacing: 0.8px;
  }}
  .status-pill {{
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    background: #0D2348; border: 1px solid #1A3A6B; color: #8BA3C7;
  }}
  .status-pill.running {{ background:#0A2540; border-color:#00C2FF; color:#00C2FF; }}
  .status-pill.done    {{ background:#0A2015; border-color:#00C851; color:#00C851; }}

  /* ── Pipeline row ── */
  .pipeline {{
    display: flex;
    align-items: flex-start;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 12px;
  }}

  /* ── Nodes ── */
  .node-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 140px;
  }}
  .node {{
    position: relative;
    border: 2px solid #1A3A6B;
    border-radius: 12px;
    padding: 14px 12px 10px;
    width: 130px;
    text-align: center;
    cursor: default;
    transition: border-color 0.3s, background 0.3s;
  }}
  .node-badge {{
    position: absolute; top: 7px; right: 9px;
    width: 18px; height: 18px;
    display: flex; align-items: center; justify-content: center;
  }}
  .node-icon {{
    width: 44px; height: 44px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 8px;
    background: #0A1628;
  }}
  .node-title {{
    font-size: 11px; font-weight: 700;
    color: #8BA3C7; letter-spacing: 0.4px;
    margin-bottom: 3px;
  }}
  .node-sub {{
    font-size: 9px; color: #4A6A9B;
    letter-spacing: 0.3px;
  }}

  /* ── Tool chips (below node) ── */
  .tool-group {{
    display: flex; flex-direction: column;
    align-items: center; gap: 4px;
    margin-top: 10px; width: 100%;
  }}
  .tool-chip {{
    font-size: 9px; font-weight: 600;
    padding: 3px 8px; border-radius: 5px;
    border: 1px solid #1A3A6B;
    color: #8BA3C7; background: #0D1F3C;
    letter-spacing: 0.3px;
    white-space: nowrap;
    transition: border-color 0.3s, color 0.3s;
  }}

  /* ── Arrows ── */
  .arrow-wrap {{
    display: flex; align-items: center;
    margin-top: 30px; flex-shrink: 0;
  }}
  .arrow-line {{
    width: 36px; height: 2px;
    background: #1A3A6B;
    transition: background 0.3s;
  }}
  .arrow-head {{
    width: 0; height: 0;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 8px solid #1A3A6B;
    transition: border-left-color 0.3s;
  }}

  /* ── Work order badge ── */
  .wo-badge {{
    margin-top: 14px;
    background: #0A2015;
    border: 1px solid #00C851;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 10px; color: #00C851;
    text-align: center; width: 130px;
    font-weight: 600;
  }}
  .wo-num {{ font-size: 11px; font-weight: 800; color: #00C2FF; }}

  /* ── Animations ── */
  @keyframes pulse-border {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(0,194,255,0); border-color: #00C2FF; }}
    50%      {{ box-shadow: 0 0 12px 4px rgba(0,194,255,0.25); border-color: #66DFFF; }}
  }}
  @keyframes spin {{
    to {{ transform: rotate(360deg); }}
  }}
  .spinner {{
    width: 14px; height: 14px;
    border: 2px solid rgba(0,194,255,0.3);
    border-top-color: #00C2FF;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }}
  @keyframes flow-arrow {{
    0%   {{ opacity: 0.4; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.4; }}
  }}

  /* ── Legend ── */
  .legend {{
    display: flex; gap: 20px; margin-top: 28px;
    padding-top: 16px; border-top: 1px solid #1A3A6B;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 10px; color: #4A6A9B;
  }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    border: 2px solid;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="flow-header">
  <div class="flow-title">Agent Pipeline — Live Status</div>
  {"<div class='status-pill running'>&#9654; RUNNING</div>" if agent_status in ("pending","agent1_done","agent2_done") else
   "<div class='status-pill done'>&#10003; COMPLETE</div>" if agent_status == "done" else
   "<div class='status-pill'>STANDBY</div>"}
</div>

<!-- Pipeline -->
<div class="pipeline">

  {_node("⚡", "ANOMALY TRIGGER", "z-score threshold breach",
         trigger_state, node_id="n-trigger")}

  {_arrow(sensor_state)}

  {_node("🔍", "SENSOR INTEL", "Agent 1",
         sensor_state,
         tools=["get_sensor_snapshot", "get_anomaly_context", "get_asset_profile"],
         node_id="n-sensor")}

  {_arrow(diagnosis_state)}

  {_node("🔬", "DIAGNOSIS", "Agent 2",
         diagnosis_state,
         tools=["get_fault_library", "get_system_health_metrics"],
         node_id="n-diag")}

  {_arrow(action_state)}

  {_node("📋", "ACTION", "Agent 3",
         action_state,
         tools=["get_maintenance_history", "create_work_order"],
         node_id="n-action")}

  {_arrow(output_state)}

  <!-- Output node -->
  <div class="node-wrap">
    <div class="node" style="border-color:{'#00C851' if output_state=='done' else '#1A3A6B'};
                              background:{'#0A2015' if output_state=='done' else '#0D1F3C'};">
      <div class="node-icon" style="background:{'#0D2A1A' if output_state=='done' else '#0A1628'};">
        {'📄' if output_state!='done' else '✅'}
      </div>
      <div class="node-title" style="color:{'#FFFFFF' if output_state=='done' else '#8BA3C7'};">
        WORK ORDER
      </div>
      <div class="node-sub">{'CMMS / ERP' if output_state!='done' else 'Pending Approval'}</div>
    </div>
    {"<div class='wo-badge'><div class='wo-num'>" + wo_num + "</div><div>Pending Approval</div></div>"
     if output_state == "done" else ""}
  </div>

</div>

<!-- Legend -->
<div class="legend">
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#1A3A6B;background:#0D1F3C;"></div>Standby
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#00C2FF;background:#0A2540;"></div>Running
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#00C851;background:#0A2015;"></div>Complete
  </div>
  <div class="legend-item" style="margin-left:auto;color:#4A6A9B;">
    &#9670; = tool call &nbsp;|&nbsp; &#8594; = data flow
  </div>
</div>

</body>
</html>
"""
    return html

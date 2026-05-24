"""
Sync.ai Demo — Chart Builders  (production-grade)
All Plotly chart factories used by app_v2.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette ─────────────────────────────────────────────────────────────
BG_PAPER    = "#070F1D"
BG_PLOT     = "#0D1F3C"
BG_PLOT2    = "#091627"        # slightly darker for depth
GRID_COLOR  = "#112240"
TICK_COLOR  = "#4A6A9B"
TEXT_COLOR  = "#8BA3C7"
TEXT_BRIGHT = "#C8D8F0"
BLUE        = "#00C2FF"
BLUE_DIM    = "rgba(0,194,255,0.12)"
ORANGE      = "#FF8C00"
ORANGE_DIM  = "rgba(255,140,0,0.10)"
PURPLE      = "#9B59B6"
PURPLE_DIM  = "rgba(155,89,182,0.10)"
RED         = "#FF4444"
RED_DIM     = "rgba(255,68,68,0.15)"
GREEN       = "#00C851"
GREEN_DIM   = "rgba(0,200,81,0.08)"
YELLOW      = "#FFD700"
WHITE       = "#FFFFFF"

_BASE_LAYOUT = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG_PLOT,
    font=dict(color=TEXT_COLOR, size=11, family="Inter, Segoe UI, sans-serif"),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0D2348",
        bordercolor="#1A3A6B",
        font=dict(color=WHITE, size=12, family="Inter, Segoe UI, sans-serif"),
    ),
)

_AXIS_STYLE = dict(
    gridcolor=GRID_COLOR,
    tickcolor=TICK_COLOR,
    linecolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    tickfont=dict(color=TEXT_COLOR, size=10),
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
    """Circular gauge with three colour zones and optional delta vs baseline."""
    delta_cfg = dict(
        reference=baseline,
        valueformat=".1f",
        increasing=dict(color=RED),
        decreasing=dict(color=GREEN),
    ) if baseline is not None else None

    if value <= green_max:
        bar_color = GREEN
        zone_label = "NORMAL"
    elif value <= yellow_max:
        bar_color = ORANGE
        zone_label = "WARNING"
    else:
        bar_color = RED
        zone_label = "CRITICAL"

    indicator = go.Indicator(
        mode="gauge+number" + ("+delta" if delta_cfg else ""),
        value=value,
        delta=delta_cfg,
        number=dict(
            suffix=f" {unit}",
            font=dict(size=22, color=WHITE, family="Inter, Segoe UI, sans-serif"),
        ),
        title=dict(
            text=f"{title}<br><span style='font-size:10px;color:{TICK_COLOR};'>{zone_label}</span>",
            font=dict(size=12, color=TEXT_COLOR),
        ),
        gauge=dict(
            axis=dict(
                range=[min_val, max_val],
                tickcolor=TICK_COLOR,
                tickfont=dict(size=9, color=TICK_COLOR),
                nticks=5,
            ),
            bar=dict(color=bar_color, thickness=0.22),
            bgcolor=BG_PLOT2,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            steps=[
                dict(range=[min_val, green_max],    color="#0A2015"),
                dict(range=[green_max, yellow_max], color="#1A1200"),
                dict(range=[yellow_max, max_val],   color="#1F0808"),
            ],
            threshold=dict(
                line=dict(color=TICK_COLOR, width=1.5),
                thickness=0.75,
                value=baseline if baseline else green_max,
            ),
        ),
    )

    fig = go.Figure(indicator)
    fig.update_layout(**{
        **_BASE_LAYOUT,
        "height": height,
        "margin": dict(l=16, r=16, t=36, b=8),
        "paper_bgcolor": BG_PAPER,
    })
    return fig


# ── 2. Scrolling live chart ───────────────────────────────────────────────────

def scrolling_chart(
    window_df: pd.DataFrame,
    active_alert: dict | None = None,
    event_log: list[dict] | None = None,
    height: int = 340,
) -> go.Figure:
    """
    Main live chart: active_power (filled area) + pressure traces on secondary axis.
    Reference bands show normal operating range. Anomaly events annotated.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts = window_df["timestamp"]

    # ── Normal operating band for active_power (0–30 W) ──────────────────────
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor=GREEN_DIM, line_width=0,
        annotation_text="Normal", annotation_position="right",
        annotation=dict(font_size=9, font_color=GREEN),
        secondary_y=False,
    )
    # Warning band 30–45 W
    fig.add_hrect(
        y0=30, y1=45,
        fillcolor=ORANGE_DIM, line_width=0,
        annotation_text="Warning", annotation_position="right",
        annotation=dict(font_size=9, font_color=ORANGE),
        secondary_y=False,
    )

    # ── Active power — gradient filled area ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["active_power"],
        name="Active Power",
        fill="tozeroy",
        fillcolor=BLUE_DIM,
        line=dict(color=BLUE, width=2.5, shape="spline", smoothing=0.8),
        hovertemplate="<b>%{x|%H:%M %d %b}</b><br>Power: <b>%{y:.1f} W</b><extra></extra>",
    ), secondary_y=False)

    # ── High pressure ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["high_pressure_1"],
        name="High Pressure",
        line=dict(color=ORANGE, width=1.8, dash="dot", shape="spline", smoothing=0.8),
        hovertemplate="HP: <b>%{y:.2f} bar</b><extra></extra>",
        opacity=0.85,
    ), secondary_y=True)

    # ── Low pressure ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["low_pressure_1"],
        name="Low Pressure",
        line=dict(color=PURPLE, width=1.8, dash="dot", shape="spline", smoothing=0.8),
        hovertemplate="LP: <b>%{y:.2f} bar</b><extra></extra>",
        opacity=0.85,
    ), secondary_y=True)

    # ── Anomaly markers from event log ────────────────────────────────────────
    if event_log:
        ev_ts   = [pd.Timestamp(e["timestamp"]) for e in event_log]
        ev_vals = []
        ev_labs = []
        for e in event_log:
            ts_match = window_df["timestamp"].searchsorted(pd.Timestamp(e["timestamp"]))
            if 0 <= ts_match < len(window_df):
                ev_vals.append(float(window_df["active_power"].iloc[ts_match]))
            else:
                ev_vals.append(None)
            ev_labs.append(f"{e['sensor']} z={e['z_score']}")
        valid = [(t, v, l) for t, v, l in zip(ev_ts, ev_vals, ev_labs) if v is not None]
        if valid:
            v_ts, v_vals, v_labs = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(v_ts), y=list(v_vals),
                mode="markers+text",
                name="Anomaly Event",
                text=list(v_labs),
                textposition="top center",
                textfont=dict(color=RED, size=9),
                marker=dict(
                    color=RED, size=10, symbol="circle",
                    line=dict(color=WHITE, width=1.5),
                ),
                hovertemplate="<b>ANOMALY</b><br>%{text}<extra></extra>",
            ), secondary_y=False)

    # ── Alert highlight band — last 5 rows ────────────────────────────────────
    if active_alert is not None and len(window_df) >= 5:
        band_start = str(window_df["timestamp"].iloc[-5])
        band_end   = str(window_df["timestamp"].iloc[-1])
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=band_start, x1=band_end, y0=0, y1=1,
            fillcolor=RED_DIM, line_width=0,
        )
        fig.add_annotation(
            x=window_df["timestamp"].iloc[-3], y=1,
            xref="x", yref="paper",
            text="⚠ ANOMALY", showarrow=False,
            font=dict(color=RED, size=10, family="Inter, Segoe UI, sans-serif"),
            yanchor="top", yshift=-4,
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.16, bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=TEXT_COLOR),
            itemsizing="constant",
        ),
        xaxis=dict(
            **_AXIS_STYLE,
            showticklabels=True,
            tickformat="%H:%M\n%d %b",
            rangeslider=dict(visible=False),
        ),
        margin=dict(l=12, r=60, t=16, b=48),
    )
    fig.update_yaxes(
        title_text="Power (W)",
        **_AXIS_STYLE,
        title_font=dict(color=BLUE, size=11),
        secondary_y=False,
        range=[0, None],
    )
    fig.update_yaxes(
        title_text="Pressure (bar)",
        **_AXIS_STYLE,
        title_font=dict(color=ORANGE, size=11),
        secondary_y=True,
        showgrid=False,
    )
    return fig


# ── 3. Temperature panel ──────────────────────────────────────────────────────

def temperature_panel(window_df: pd.DataFrame, height: int = 210) -> go.Figure:
    fig = go.Figure()
    ts = window_df["timestamp"]

    # Comfort/design zone between setpoint and +5°C
    if "summer_SP_temp" in window_df.columns:
        sp = window_df["summer_SP_temp"]
        fig.add_traces([
            go.Scatter(x=ts, y=sp + 5, fill=None, line=dict(color="rgba(0,0,0,0)"),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=ts, y=sp, fill="tonexty",
                       fillcolor="rgba(0,200,81,0.06)", line=dict(color="rgba(0,0,0,0)"),
                       showlegend=False, hoverinfo="skip"),
        ])

    # Fill between inlet and outlet
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([window_df["outlet_temp"], window_df["inlet_temp"][::-1]]),
        fill="toself", fillcolor=PURPLE_DIM,
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=ts, y=window_df["outlet_temp"],
        name="Outlet", line=dict(color=ORANGE, width=2, shape="spline", smoothing=0.8),
        hovertemplate="Outlet: <b>%{y:.1f}°C</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["inlet_temp"],
        name="Inlet", line=dict(color=PURPLE, width=1.5, shape="spline", smoothing=0.8),
        hovertemplate="Inlet: <b>%{y:.1f}°C</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["ambient_temp"],
        name="Ambient", line=dict(color=BLUE, width=1.2, dash="dash"),
        hovertemplate="Ambient: <b>%{y:.1f}°C</b><extra></extra>",
        opacity=0.75,
    ))
    if "summer_SP_temp" in window_df.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=window_df["summer_SP_temp"],
            name="Setpoint", line=dict(color=RED, width=1, dash="dot"),
            hovertemplate="Setpoint: <b>%{y:.1f}°C</b><extra></extra>",
            opacity=0.7,
        ))

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Temperature", font=dict(color=TEXT_BRIGHT, size=12), x=0.01),
        xaxis=dict(**_AXIS_STYLE, tickformat="%H:%M"),
        yaxis=dict(**_AXIS_STYLE, title="°C", title_font=dict(size=10)),
        legend=dict(orientation="h", y=-0.4, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=32, b=52),
    )
    return fig


# ── 4. Pressure history panel ─────────────────────────────────────────────────

def pressure_history_panel(window_df: pd.DataFrame, height: int = 210) -> go.Figure:
    fig = go.Figure()
    ts = window_df["timestamp"]

    # Normal operating range band: high pressure 15–22 bar
    fig.add_hrect(y0=15, y1=22, fillcolor="rgba(0,200,81,0.05)", line_width=0,
                  annotation_text="HP Normal", annotation_position="right",
                  annotation=dict(font_size=8, font_color=GREEN))

    # Shaded differential band between HP and LP
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([window_df["high_pressure_1"], window_df["low_pressure_1"][::-1]]),
        fill="toself", fillcolor="rgba(0,194,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=ts, y=window_df["high_pressure_1"],
        name="High P", line=dict(color=ORANGE, width=2, shape="spline", smoothing=0.8),
        hovertemplate="HP: <b>%{y:.2f} bar</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["low_pressure_1"],
        name="Low P", line=dict(color=PURPLE, width=2, shape="spline", smoothing=0.8),
        hovertemplate="LP: <b>%{y:.2f} bar</b><extra></extra>",
    ))

    # Pressure ratio as annotation on the chart (current value)
    if len(window_df) > 0:
        hp_now = float(window_df["high_pressure_1"].iloc[-1])
        lp_now = max(float(window_df["low_pressure_1"].iloc[-1]), 0.01)
        pr_now = round(hp_now / lp_now, 2)
        pr_color = GREEN if pr_now <= 1.4 else ORANGE if pr_now <= 1.8 else RED
        fig.add_annotation(
            x=window_df["timestamp"].iloc[-1], y=hp_now,
            text=f"P.Ratio: {pr_now}",
            showarrow=False, xanchor="right",
            font=dict(color=pr_color, size=9),
            bgcolor=BG_PAPER, bordercolor=pr_color, borderpad=2,
        )

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Pressure Differential", font=dict(color=TEXT_BRIGHT, size=12), x=0.01),
        xaxis=dict(**_AXIS_STYLE, tickformat="%H:%M"),
        yaxis=dict(**_AXIS_STYLE, title="bar", title_font=dict(size=10)),
        legend=dict(orientation="h", y=-0.4, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=32, b=52),
    )
    return fig


# ── 5. System status panel ────────────────────────────────────────────────────

def system_status_panel(window_df: pd.DataFrame, height: int = 210) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts = window_df["timestamp"]

    # ON/OFF as filled step chart
    fig.add_trace(go.Scatter(
        x=ts, y=window_df["on_off"],
        name="ON/OFF", fill="tozeroy",
        fillcolor="rgba(0,200,81,0.18)",
        line=dict(color=GREEN, width=1.5, shape="hv"),
        hovertemplate="Status: <b>%{customdata}</b><extra></extra>",
        customdata=["ON" if v == 1 else "OFF" for v in window_df["on_off"]],
    ), secondary_y=False)

    if "damper" in window_df.columns:
        fig.add_trace(go.Bar(
            x=ts, y=window_df["damper"],
            name="Damper %", marker_color="rgba(0,194,255,0.25)",
            marker_line_color="rgba(0,194,255,0.4)", marker_line_width=0.5,
            hovertemplate="Damper: <b>%{y:.0f}%</b><extra></extra>",
        ), secondary_y=True)

    # Runtime % annotation
    on_pct = round(float(window_df["on_off"].mean() * 100), 0)
    fig.add_annotation(
        x=0.02, y=0.92, xref="paper", yref="paper",
        text=f"Runtime: {on_pct}%",
        showarrow=False, font=dict(color=GREEN, size=10),
        bgcolor=BG_PAPER, borderpad=3,
    )

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="System Status & Damper", font=dict(color=TEXT_BRIGHT, size=12), x=0.01),
        xaxis=dict(**_AXIS_STYLE, tickformat="%H:%M"),
        legend=dict(orientation="h", y=-0.4, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=32, b=52),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="ON / OFF", range=[-0.1, 1.5],
                     **_AXIS_STYLE, secondary_y=False)
    fig.update_yaxes(title_text="Damper %", range=[0, 130],
                     showgrid=False, tickfont=dict(color=TEXT_COLOR, size=10),
                     secondary_y=True)
    return fig


# ── 6. Humidity / CO2 panel ───────────────────────────────────────────────────

def humidity_co2_panel(window_df: pd.DataFrame, height: int = 210) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts = window_df["timestamp"]

    # Comfort band 40–60 % RH
    fig.add_hrect(y0=40, y1=60, fillcolor=GREEN_DIM, line_width=0,
                  annotation_text="Comfort band", annotation_position="right",
                  annotation=dict(font_size=8, font_color=GREEN),
                  secondary_y=False)

    fig.add_trace(go.Scatter(
        x=ts, y=window_df["ambient_humidity"],
        name="Humidity %", fill="tozeroy",
        fillcolor="rgba(0,194,255,0.08)",
        line=dict(color=BLUE, width=1.8, shape="spline", smoothing=0.8),
        hovertemplate="Humidity: <b>%{y:.0f}%</b><extra></extra>",
    ), secondary_y=False)

    if "co2_1" in window_df.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=window_df["co2_1"],
            name="CO₂ (ppm)",
            line=dict(color=YELLOW, width=1.5, dash="dot", shape="spline", smoothing=0.8),
            hovertemplate="CO₂: <b>%{y:.0f} ppm</b><extra></extra>",
            opacity=0.85,
        ), secondary_y=True)

    fig.update_layout(
        **_BASE_LAYOUT, height=height,
        title=dict(text="Humidity & CO₂", font=dict(color=TEXT_BRIGHT, size=12), x=0.01),
        xaxis=dict(**_AXIS_STYLE, tickformat="%H:%M"),
        legend=dict(orientation="h", y=-0.4, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=32, b=52),
    )
    fig.update_yaxes(title_text="Humidity %", range=[0, 100],
                     **_AXIS_STYLE, secondary_y=False)
    fig.update_yaxes(title_text="CO₂ (ppm)", showgrid=False,
                     tickfont=dict(color=TEXT_COLOR, size=10), secondary_y=True)
    return fig


# ── 7. Agent pipeline flow diagram ───────────────────────────────────────────

def agent_flow_html(agent_status: str, agent_results: dict) -> str:
    """
    Returns a self-contained HTML string rendering the n8n-style agent
    pipeline flow diagram. Node colours and badges update based on agent_status.
    """
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

    trigger_state   = "done" if agent_status != "idle" else "idle"
    sensor_state    = _state(["sensor"])
    if agent_status == "pending":
        sensor_state = "running"
    diagnosis_state = _state(["diagnosis"])
    if agent_status == "agent1_done":
        diagnosis_state = "running"
    action_state    = _state(["action"])
    if agent_status == "agent2_done":
        action_state = "running"
    output_state    = "done" if agent_status == "done" else "idle"

    wo     = agent_results.get("work_order", {})
    wo_num = wo.get("wo_number", "—")

    COLORS = {
        "idle":    {"border": "#1A3A6B", "bg": "#0D1F3C", "label": "#8BA3C7", "icon_bg": "#091627"},
        "running": {"border": "#00C2FF", "bg": "#071D35", "label": "#FFFFFF", "icon_bg": "#0A2540"},
        "done":    {"border": "#00C851", "bg": "#071D14", "label": "#FFFFFF", "icon_bg": "#0A2015"},
    }

    def _node(icon: str, title: str, subtitle: str, state: str,
              tools: list[str] | None = None, node_id: str = "") -> str:
        c = COLORS[state]
        pulse_anim = "animation: pulse-border 1.4s ease-in-out infinite;" if state == "running" else ""
        if state == "running":
            badge = '<div class="spinner"></div>'
        elif state == "done":
            badge = '<span style="color:#00C851;font-size:14px;font-weight:900;">✓</span>'
        else:
            badge = ""

        tool_chips = ""
        if tools:
            chips_html = "".join(
                f'<div class="tool-chip" style="border-color:{c["border"]};color:{c["label"]};">'
                f'<span style="color:#00C2FF;margin-right:3px;">◆</span>{t}</div>'
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
        color     = "#00C2FF" if state in ("running", "done") else "#1A3A6B"
        animated  = "animation: flow-arrow 1s linear infinite;" if state == "running" else ""
        return f"""
        <div class="arrow-wrap">
          <div class="arrow-line" style="background:{color};{animated}"></div>
          <div class="arrow-head" style="border-left-color:{color};"></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #070F1D;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 20px 16px;
    min-height: 520px;
  }}

  .flow-header {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 24px;
  }}
  .flow-title {{
    font-size: 12px; font-weight: 700; color: #8BA3C7;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  .status-pill {{
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    background: #0D2348; border: 1px solid #1A3A6B; color: #8BA3C7;
  }}
  .status-pill.running {{ background:#071D35; border-color:#00C2FF; color:#00C2FF; }}
  .status-pill.done    {{ background:#071D14; border-color:#00C851; color:#00C851; }}

  .pipeline {{
    display: flex;
    align-items: flex-start;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 12px;
  }}

  .node-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 142px;
  }}
  .node {{
    position: relative;
    border: 2px solid #1A3A6B;
    border-radius: 14px;
    padding: 14px 12px 12px;
    width: 132px;
    text-align: center;
    cursor: default;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s;
  }}
  .node-badge {{
    position: absolute; top: 8px; right: 9px;
    width: 18px; height: 18px;
    display: flex; align-items: center; justify-content: center;
  }}
  .node-icon {{
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 8px;
  }}
  .node-title {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.6px;
    margin-bottom: 3px; text-transform: uppercase;
  }}
  .node-sub {{
    font-size: 9px; color: #4A6A9B; letter-spacing: 0.3px;
  }}

  .tool-group {{
    display: flex; flex-direction: column;
    align-items: center; gap: 4px;
    margin-top: 10px; width: 100%;
  }}
  .tool-chip {{
    font-size: 8.5px; font-weight: 600;
    padding: 3px 7px; border-radius: 5px;
    border: 1px solid #1A3A6B;
    color: #8BA3C7; background: #0D1F3C;
    letter-spacing: 0.2px; white-space: nowrap;
    transition: border-color 0.3s, color 0.3s;
  }}

  .arrow-wrap {{
    display: flex; align-items: center;
    margin-top: 31px; flex-shrink: 0;
  }}
  .arrow-line {{
    width: 32px; height: 2px;
    background: #1A3A6B;
    transition: background 0.3s;
  }}
  .arrow-head {{
    width: 0; height: 0;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 7px solid #1A3A6B;
    transition: border-left-color 0.3s;
  }}

  .wo-badge {{
    margin-top: 12px;
    background: #071D14;
    border: 1px solid #00C851;
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 9.5px; color: #00C851;
    text-align: center; width: 132px;
    font-weight: 600;
  }}
  .wo-num {{ font-size: 10px; font-weight: 800; color: #00C2FF; margin-bottom: 2px; }}

  @keyframes pulse-border {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(0,194,255,0); border-color:#00C2FF; }}
    50%      {{ box-shadow: 0 0 14px 5px rgba(0,194,255,0.22); border-color:#66DFFF; }}
  }}
  @keyframes spin {{
    to {{ transform: rotate(360deg); }}
  }}
  .spinner {{
    width: 14px; height: 14px;
    border: 2px solid rgba(0,194,255,0.25);
    border-top-color: #00C2FF;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }}
  @keyframes flow-arrow {{
    0%   {{ opacity:0.35; }} 50% {{ opacity:1; }} 100% {{ opacity:0.35; }}
  }}

  .legend {{
    display: flex; gap: 20px; margin-top: 24px;
    padding-top: 14px; border-top: 1px solid #112240;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 10px; color: #4A6A9B;
  }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 50%; border: 2px solid;
  }}
</style>
</head>
<body>

<div class="flow-header">
  <div class="flow-title">AI Agent Pipeline &mdash; Live Status</div>
  {"<div class='status-pill running'>&#9654;&nbsp;RUNNING</div>" if agent_status in ("pending","agent1_done","agent2_done") else
   "<div class='status-pill done'>&#10003;&nbsp;COMPLETE</div>" if agent_status == "done" else
   "<div class='status-pill'>STANDBY</div>"}
</div>

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

  <div class="node-wrap">
    <div class="node" style="border-color:{'#00C851' if output_state=='done' else '#1A3A6B'};
                              background:{'#071D14' if output_state=='done' else '#0D1F3C'};">
      <div class="node-icon" style="background:{'#0A2015' if output_state=='done' else '#091627'};">
        {'✅' if output_state=='done' else '📄'}
      </div>
      <div class="node-title" style="color:{'#FFFFFF' if output_state=='done' else '#8BA3C7'};">
        WORK ORDER
      </div>
      <div class="node-sub">{'Pending Approval' if output_state=='done' else 'CMMS / ERP'}</div>
    </div>
    {"<div class='wo-badge'><div class='wo-num'>" + wo_num + "</div><div>Pending Approval</div></div>"
     if output_state == "done" else ""}
  </div>

</div>

<div class="legend">
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#1A3A6B;background:#0D1F3C;"></div>Standby
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#00C2FF;background:#071D35;"></div>Running
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="border-color:#00C851;background:#071D14;"></div>Complete
  </div>
  <div class="legend-item" style="margin-left:auto;color:#4A6A9B;font-size:10px;">
    ◆ = tool call &nbsp;&nbsp;→ = data flow
  </div>
</div>

</body>
</html>"""
    return html

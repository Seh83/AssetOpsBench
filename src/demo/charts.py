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
    margin=dict(l=10, r=10, t=30, b=10),
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
                gridcolor=GRID_COLOR,
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
    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor=BG_PAPER,
    )
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

# Sync.ai — HVAC Asset Intelligence Demo

A client-facing demo powered by real HVAC sensor data (3 months, 5-min intervals) showing
AI-driven anomaly detection, energy forecasting, and a natural language assistant.

## Quick Start

### 1. Install dependencies
```bash
pip install -r src/demo/requirements.txt
```

### 2. Set your Anthropic API key (for the AI chat tab)
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Pre-compute ML results (run once — takes ~2–5 min)
```bash
python src/demo/precompute.py
```
This generates `src/demo/results/anomalies.csv` and `src/demo/results/forecast.json`.

### 4. Launch the demo
```bash
streamlit run src/demo/app.py
streamlit run src/demo/app_v2.py
```
Opens at http://localhost:8501

---

## What the demo shows

| Tab | Content |
|---|---|
| **Live Dashboard** | KPI cards (status, power, anomalies, pressure ratio) + multi-sensor chart |
| **Anomaly Intelligence** | Detected events table, zoom-in chart, AI root cause, work order generator |
| **Energy Forecast** | 24hr power forecast with confidence bands + setpoint compliance chart |
| **Sync.ai Assistant** | Claude-powered chat grounded in your actual sensor data |

## Dataset
`src/dataset/hvac_demo_ready.csv` — 26,208 rows × 18 columns, Jun–Aug 2022.
Pre-cleaned: nulls interpolated, outliers capped, full 5-min grid.

## ML Backend
`precompute.py` tries IBM TinyTimeMixer (tsfm_public) first.
If not installed, falls back to rolling z-score (anomaly) + Holt-Winters (forecast) automatically.

## Notes
- The AI chat tab requires `ANTHROPIC_API_KEY`. All other tabs work offline.
- Re-run `precompute.py` if you change the dataset.
- Adjust the energy rate ($/kWh) in the sidebar to match your client's tariff.

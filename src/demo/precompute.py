"""
Sync.ai Demo — Pre-computation Pipeline
Run this once before launching app.py.
Produces: results/anomalies.csv and results/forecast.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "src" / "dataset" / "hvac_demo_ready.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANOMALY_OUT = RESULTS_DIR / "anomalies.csv"
FORECAST_OUT = RESULTS_DIR / "forecast.json"

TARGET_COLS = ["active_power", "high_pressure_1", "low_pressure_1", "outlet_temp"]
TIMESTAMP_COL = "timestamp"
FREQUENCY = "5_minutes"
FORECAST_HORIZON = 288  # 24 hours at 5-min intervals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity(z: float) -> str:
    if abs(z) >= 3.5:
        return "Critical"
    if abs(z) >= 2.8:
        return "Warning"
    return "Watch"


def _zscore_anomaly(df: pd.DataFrame, cols: list[str], window: int = 96) -> pd.DataFrame:
    """Rolling z-score anomaly detection — statistical fallback."""
    print("  Using statistical fallback (rolling z-score, window=96 steps / 8 hrs)")
    result = df.copy()
    result["anomaly_label"] = 0
    result["z_score"] = 0.0
    result["anomaly_sensor"] = ""
    result["severity"] = ""

    for col in cols:
        rolling_mean = result[col].rolling(window, center=True, min_periods=1).mean()
        rolling_std = result[col].rolling(window, center=True, min_periods=1).std().fillna(1)
        z = ((result[col] - rolling_mean) / rolling_std).abs()
        mask = z > 2.5
        # Only mark as anomaly if z is the largest seen so far for this row
        update_mask = mask & (z > result["z_score"])
        result.loc[update_mask, "anomaly_label"] = 1
        result.loc[update_mask, "z_score"] = z[update_mask].round(3)
        result.loc[update_mask, "anomaly_sensor"] = col
        result.loc[update_mask, "severity"] = z[update_mask].apply(_severity)

    return result


def _holt_winters_forecast(series: pd.Series, horizon: int) -> dict:
    """Simple exponential smoothing forecast — fallback when tsfm unavailable."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    print("  Using Holt-Winters fallback for forecasting")
    # Use last 2016 steps (7 days) for fitting
    fit_data = series.dropna().tail(2016)
    model = ExponentialSmoothing(
        fit_data, trend="add", seasonal="add", seasonal_periods=288
    ).fit(optimized=True)
    pred = model.forecast(horizon)
    std = fit_data.std()
    return {
        "predicted": pred.tolist(),
        "upper": (pred + 1.96 * std).tolist(),
        "lower": np.maximum(pred - 1.96 * std, 0).tolist(),
    }


# ── TSFM path ─────────────────────────────────────────────────────────────────

def _tsfm_anomaly(df: pd.DataFrame, csv_path: str) -> pd.DataFrame | None:
    try:
        import tsfm_public  # noqa: F401
    except ImportError:
        return None

    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from servers.tsfm.main import run_integrated_tsad
    except ImportError as e:
        print(f"  Could not import tsfm server: {e}")
        return None

    os.environ.setdefault("PATH_TO_MODELS_DIR", str(REPO_ROOT / "models"))
    os.environ.setdefault("PATH_TO_DATASETS_DIR", str(REPO_ROOT / "src" / "dataset"))
    os.environ.setdefault("PATH_TO_OUTPUTS_DIR", str(RESULTS_DIR))

    print("  Running TSFM integrated anomaly detection...")
    result = run_integrated_tsad(
        dataset_path=csv_path,
        timestamp_column=TIMESTAMP_COL,
        target_columns=TARGET_COLS,
        frequency_sampling=FREQUENCY,
        conditional_columns=["on_off", "damper"],
        n_calibration=0.2,
        false_alarm=0.05,
    )

    if hasattr(result, "error"):
        print(f"  TSFM error: {result.error}")
        return None

    tsad_df = pd.read_csv(result.results_file)
    return tsad_df


def _tsfm_forecast(csv_path: str) -> dict | None:
    try:
        import tsfm_public  # noqa: F401
    except ImportError:
        return None

    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from servers.tsfm.main import run_tsfm_forecasting
    except ImportError:
        return None

    os.environ.setdefault("PATH_TO_MODELS_DIR", str(REPO_ROOT / "models"))
    os.environ.setdefault("PATH_TO_DATASETS_DIR", str(REPO_ROOT / "src" / "dataset"))
    os.environ.setdefault("PATH_TO_OUTPUTS_DIR", str(RESULTS_DIR))

    print("  Running TSFM forecasting for active_power...")
    result = run_tsfm_forecasting(
        dataset_path=csv_path,
        timestamp_column=TIMESTAMP_COL,
        target_columns=["active_power"],
        forecast_horizon=FORECAST_HORIZON,
        frequency_sampling=FREQUENCY,
        conditional_columns=["on_off", "damper"],
    )

    if hasattr(result, "error"):
        print(f"  TSFM forecast error: {result.error}")
        return None

    with open(result.results_file) as f:
        raw = json.load(f)

    predictions = np.array(raw["target_prediction"])
    timestamps = raw["timestamp"]
    if predictions.ndim == 3:
        predictions = predictions[0, :, 0]
    elif predictions.ndim == 2:
        predictions = predictions[:, 0]

    std = float(np.std(predictions))
    return {
        "timestamps": timestamps if isinstance(timestamps[0], str) else [str(t) for t in timestamps],
        "predicted": predictions.tolist(),
        "upper": (predictions + 1.96 * std).tolist(),
        "lower": np.maximum(predictions - 1.96 * std, 0).tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print("  Sync.ai — Pre-computation Pipeline")
    print(f"{'='*55}")
    print(f"  Dataset : {DATASET_PATH}")
    print(f"  Output  : {RESULTS_DIR}\n")

    print("[1/3] Loading dataset...")
    df = pd.read_csv(DATASET_PATH, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    print(f"  {len(df):,} rows, {df.shape[1]} columns, {df[TIMESTAMP_COL].min().date()} to {df[TIMESTAMP_COL].max().date()}")

    # ── Anomaly detection ─────────────────────────────────────────────────────
    print("\n[2/3] Anomaly detection...")
    csv_path = str(DATASET_PATH)

    tsfm_result = _tsfm_anomaly(df, csv_path)

    if tsfm_result is not None:
        print("  TSFM anomaly detection succeeded.")
        # Merge TSFM anomaly labels back onto the full dataframe
        if "timestamp" in tsfm_result.columns:
            tsfm_result["timestamp"] = pd.to_datetime(tsfm_result["timestamp"])
            result_df = df.merge(
                tsfm_result[["timestamp", "anomaly_label"]].rename(columns={"timestamp": TIMESTAMP_COL}),
                on=TIMESTAMP_COL,
                how="left",
            )
        else:
            result_df = df.copy()
            result_df["anomaly_label"] = tsfm_result.get("anomaly_label", 0).values if len(tsfm_result) == len(df) else 0

        result_df["anomaly_label"] = result_df["anomaly_label"].fillna(0).astype(int)

        # Compute z-scores for severity enrichment regardless
        result_df["z_score"] = 0.0
        result_df["anomaly_sensor"] = ""
        result_df["severity"] = ""
        for col in TARGET_COLS:
            rolling_mean = result_df[col].rolling(96, center=True, min_periods=1).mean()
            rolling_std = result_df[col].rolling(96, center=True, min_periods=1).std().fillna(1)
            z = ((result_df[col] - rolling_mean) / rolling_std).abs()
            mask = (result_df["anomaly_label"] == 1) & (z > result_df["z_score"])
            result_df.loc[mask, "z_score"] = z[mask].round(3)
            result_df.loc[mask, "anomaly_sensor"] = col
            result_df.loc[mask, "severity"] = z[mask].apply(_severity)
    else:
        print("  tsfm_public not available — falling back to statistical detection.")
        result_df = _zscore_anomaly(df, TARGET_COLS)

    anomaly_count = result_df["anomaly_label"].sum()
    print(f"  Anomalies detected: {anomaly_count:,} rows ({anomaly_count/len(result_df)*100:.1f}%)")

    result_df.to_csv(ANOMALY_OUT, index=False)
    print(f"  Saved -> {ANOMALY_OUT}")

    # ── Forecasting ───────────────────────────────────────────────────────────
    print("\n[3/3] Forecasting active_power (next 24 hrs)...")
    forecast = _tsfm_forecast(csv_path)

    if forecast is None:
        print("  tsfm_public not available — falling back to Holt-Winters.")
        series = df.set_index(TIMESTAMP_COL)["active_power"]
        hw = _holt_winters_forecast(series, FORECAST_HORIZON)
        # Generate timestamp index continuing from last known point
        last_ts = df[TIMESTAMP_COL].max()
        future_idx = pd.date_range(last_ts + pd.Timedelta("5min"), periods=FORECAST_HORIZON, freq="5min")
        forecast = {
            "timestamps": [str(t) for t in future_idx],
            **hw,
        }

    with open(FORECAST_OUT, "w") as f:
        json.dump(forecast, f)
    print(f"  Forecast horizon: {len(forecast['predicted'])} steps ({len(forecast['predicted']) * 5 // 60} hrs)")
    print(f"  Saved -> {FORECAST_OUT}")

    print(f"\n{'='*55}")
    print("  Pre-computation complete. Run the demo:")
    print("  streamlit run src/demo/app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

# app.py
import os
import glob
import json
import math
from datetime import datetime
from flask import Flask, request, jsonify

import xarray as xr
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

app = Flask(__name__)

# -----------------------
# Helper functions
# -----------------------
def parse_train_years(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [str(int(y)) for y in x]
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    raise ValueError("train_years must be list or comma-separated string")

def find_nc_files(folder_path):
    p = os.path.expanduser(folder_path)
    files = sorted(glob.glob(os.path.join(p, "*.nc4")))
    return files

def load_series_from_nc(nc_files, variable, lat, lon):
    if not nc_files:
        raise FileNotFoundError("No .nc4 files found in folder.")
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    da = ds[variable].sel(lat=lat, lon=lon, method='nearest')
    df = da.to_dataframe().sort_index()
    df.index = pd.to_datetime(df.index)
    return df

def compute_probability_above_threshold(model_fit, df_train, variable, future_date, threshold):
    future_date_dt = pd.to_datetime(future_date)
    last_train_date = df_train.index.max()
    forecast_steps = (future_date_dt - last_train_date).days
    if forecast_steps <= 0:
        raise ValueError("Future date must be after the last training date.")
    pred = model_fit.get_forecast(steps=forecast_steps)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=0.05)
    mean_val = float(pred_mean.iloc[-1])
    ci_lower = float(conf_int.iloc[-1, 0])
    ci_upper = float(conf_int.iloc[-1, 1])
    se_est = (ci_upper - ci_lower) / (2.0 * 1.96)
    if se_est <= 0 or not math.isfinite(se_est):
        se_est = 1e-8
    z = (threshold - mean_val) / se_est
    cdf_at_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    prob_gt = 1.0 - cdf_at_z
    prob_percent = float(max(0.0, min(1.0, prob_gt)) * 100.0)
    hist_percent = float((df_train[variable] > threshold).sum() / len(df_train) * 100.0) if len(df_train) > 0 else 0.0

    return {
        "forecast_mean": mean_val,
        "forecast_95ci": {"lower": ci_lower, "upper": ci_upper},
        "probability_above_threshold_percent": prob_percent,
        "historical_percentage_above_threshold": hist_percent,
        "days_ahead": int(forecast_steps)
    }

def aggregate_global_data(nc_files, variable):
    if not nc_files:
        raise FileNotFoundError("No .nc4 files found in folder.")
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    da = ds[variable].isel(time=-1)
    lats = da["lat"].values
    lons = da["lon"].values
    data = da.values

    globe_data = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = float(data[i, j])
            globe_data.append({"lat": float(lat), "lon": float(lon), variable: val})
    return globe_data

# -----------------------
# Forecast endpoint
# -----------------------
@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    folder_path = payload.get("folder_path", "MERRA2_daily")
    lat = float(payload.get("lat", -90.0))
    lon = float(payload.get("lon", -178.75))
    variable = payload.get("variable", "T2MMAX")
    train_years_raw = payload.get("train_years", None)
    train_years = parse_train_years(train_years_raw) if train_years_raw is not None else None
    future_date = payload.get("future_date", None)
    threshold = float(payload.get("threshold", 300.0))

    if future_date is None:
        return jsonify({"error": "future_date is required (YYYY-MM-DD)"}), 400
    try:
        _ = pd.to_datetime(future_date)
    except Exception:
        return jsonify({"error": "future_date not valid. Use YYYY-MM-DD"}), 400

    try:
        nc_files = find_nc_files(folder_path)
        if not nc_files:
            return jsonify({"error": f"No .nc4 files found in folder '{folder_path}'"}), 400

        df = load_series_from_nc(nc_files, variable, lat, lon)
        if df.empty:
            return jsonify({"error": "No data found for the selected location/variable."}), 400

        if train_years:
            df_train = df[df.index.year.isin([int(y) for y in train_years])]
        else:
            df_train = df.copy()

        if df_train.empty:
            return jsonify({"error": "Training set is empty for provided train_years"}), 400

        stepwise_model = auto_arima(df_train[variable], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
        p, d, q = stepwise_model.order

        model = ARIMA(df_train[variable], order=(p, d, q))
        model_fit = model.fit()

        prob_result = compute_probability_above_threshold(model_fit, df_train, variable, future_date, threshold)

        response = {
            "location": {"lat": float(lat), "lon": float(lon)},
            "variable": variable,
            "future_date": future_date,
            "threshold": threshold,
            "model_order": {"p": int(p), "d": int(d), "q": int(q)},
            "last_training_date": str(df_train.index.max().date()),
            **prob_result
        }

        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal error", "message": str(e)}), 500

# -----------------------
# Globe endpoint
# -----------------------
@app.route("/globe-data", methods=["GET"])
def globe_data():
    folder_path = request.args.get("folder_path", "MERRA2_daily")
    variable = request.args.get("variable", "T2MMAX")
    try:
        nc_files = find_nc_files(folder_path)
        globe_json = aggregate_global_data(nc_files, variable)
        return jsonify(globe_json), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal error", "message": str(e)}), 500

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

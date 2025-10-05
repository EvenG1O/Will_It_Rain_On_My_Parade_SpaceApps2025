import xarray as xr
import pandas as pd
import glob
import json
import math
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# ---------------------------
# PARAMETERS
# ---------------------------
folder_path = "MERRA2_daily"    # Folder with your .nc4 files
lat, lon = -90.0, -178.75       # Target location
variable = "T2MMAX"             # Variable to forecast
train_years = ['2020', '2021', '2022']  # Years for training
future_date = "2023-10-04"      # Specific future date to predict (YYYY-MM-DD)
threshold = 300.0               # Example threshold in K for "high temperature"

# ---------------------------
# LOAD DATA
# ---------------------------
nc_files = glob.glob(f"{folder_path}/*.nc4")
if not nc_files:
    raise FileNotFoundError("No .nc4 files found in the folder.")

ds = xr.open_mfdataset(nc_files, combine='by_coords')
if variable not in ds:
    raise ValueError(f"{variable} not found in dataset. Available variables: {list(ds.data_vars)}")

# Extract variable at the nearest lat/lon
data = ds[variable].sel(lat=lat, lon=lon, method='nearest')
df = data.to_dataframe().sort_index()
df.index = pd.to_datetime(df.index)

# Keep only training years
df_train = df[df.index.year.isin([int(y) for y in train_years])]
if df_train.empty:
    raise ValueError("Training set is empty for the selected years. Check your data and train_years.")

# ---------------------------
# TRAIN ARIMA
# ---------------------------
# Auto select order (quiet)
stepwise_model = auto_arima(df_train[variable], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
p, d, q = stepwise_model.order

model = ARIMA(df_train[variable], order=(p, d, q))
model_fit = model.fit()

# ---------------------------
# FORECAST to future_date
# ---------------------------
future_date_dt = pd.to_datetime(future_date)

# Determine number of days ahead to forecast from the last date in training set
last_train_date = df_train.index.max()
forecast_steps = (future_date_dt - last_train_date).days

if forecast_steps <= 0:
    raise ValueError("Future date must be after the last training date.")

# Use statsmodels' get_forecast to obtain mean and confidence intervals
pred = model_fit.get_forecast(steps=forecast_steps)
pred_mean = pred.predicted_mean  # pandas Series
conf_int = pred.conf_int(alpha=0.05)  # DataFrame with lower/upper columns

# We're interested in the final step (the value on the future_date)
mean_val = float(pred_mean.iloc[-1])
ci_lower = float(conf_int.iloc[-1, 0])
ci_upper = float(conf_int.iloc[-1, 1])

# Estimate standard error from 95% CI: CI = mean +/- 1.96*se  => se = (upper - lower) / (2*1.96)
se_est = (ci_upper - ci_lower) / (2.0 * 1.96)
# If se_est is zero or extremely small, set a tiny value to avoid division-by-zero
if se_est <= 0 or not math.isfinite(se_est):
    se_est = 1e-8

# Compute probability P(X > threshold) assuming normal(mean=mean_val, sd=se_est)
# Normal CDF using math.erf: CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
z = (threshold - mean_val) / se_est
cdf_at_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
prob_gt = 1.0 - cdf_at_z  # probability that X > threshold
prob_percent = float(max(0.0, min(1.0, prob_gt)) * 100.0)

# Also compute historical frequency above threshold (for context)
hist_percent = float((df_train[variable] > threshold).sum() / len(df_train) * 100.0)

# ---------------------------
# OUTPUT JSON
# ---------------------------
result = {
    "location": {"lat": float(lat), "lon": float(lon)},
    "variable": variable,
    "future_date": future_date,
    "forecast_mean": mean_val,
    "forecast_95ci": {"lower": ci_lower, "upper": ci_upper},
    "threshold": threshold,
    "probability_above_threshold_percent": prob_percent,
    "historical_percentage_above_threshold": hist_percent,
    "model_order": {"p": int(p), "d": int(d), "q": int(q)},
    "last_training_date": str(last_train_date.date()),
    "days_ahead": int(forecast_steps)
}

print(json.dumps(result, indent=4))

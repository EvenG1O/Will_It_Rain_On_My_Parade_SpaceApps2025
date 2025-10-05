import os
import glob
import math
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from datetime import timedelta

# ---------------------------
# PARAMETERS
# ---------------------------
folder_path = "MERRA2_daily"  # folder containing .nc4 files
variable = "T2MMAX"
lat, lon = -90.0, -178.75
train_years = ["2020", "2021", "2022"]  # training data
forecast_days = 180                      # forecast ~6 months
threshold = 300.0                        # example threshold

# ---------------------------
# HELPERS
# ---------------------------
def find_nc_files(folder_path):
    p = os.path.expanduser(folder_path)
    files = sorted(glob.glob(os.path.join(p, "*.nc4")))
    if not files:
        raise FileNotFoundError(f"No .nc4 files found in '{folder_path}'")
    return files

def load_series_from_nc(nc_files, variable, lat, lon):
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    da = ds[variable].sel(lat=lat, lon=lon, method='nearest')
    df = da.to_dataframe().sort_index()
    df.index = pd.to_datetime(df.index)
    return df

# ---------------------------
# LOAD DATA
# ---------------------------
nc_files = find_nc_files(folder_path)
df = load_series_from_nc(nc_files, variable, lat, lon)

# Split train/test: training and next 6 months as test
df_train = df[df.index.year.isin([int(y) for y in train_years])]
last_train_date = df_train.index.max()
forecast_start = last_train_date + pd.Timedelta(days=1)
forecast_end = forecast_start + pd.Timedelta(days=forecast_days-1)

df_test = df[(df.index >= forecast_start) & (df.index <= forecast_end)]

print("Train rows:", len(df_train))
print("Test rows (actuals for comparison):", len(df_test))

# ---------------------------
# AUTOMATIC ARIMA
# ---------------------------
print("Finding optimal ARIMA order...")
stepwise_model = auto_arima(df_train[variable], seasonal=False, trace=True)
p,d,q = stepwise_model.order
print(f"Selected ARIMA order: p={p}, d={d}, q={q}")

# ---------------------------
# FIT ARIMA MODEL
# ---------------------------
model = ARIMA(df_train[variable], order=(p,d,q))
model_fit = model.fit()
print(model_fit.summary())

# ---------------------------
# FORECAST NEXT 6 MONTHS
# ---------------------------
forecast_steps = len(df_test) if not df_test.empty else forecast_days
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

# Create forecast dataframe aligned with test data (or next 6 months)
forecast_index = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_df = pd.DataFrame({
    variable: forecast_mean.values,
    'ci_lower': forecast_ci.iloc[:,0].values,
    'ci_upper': forecast_ci.iloc[:,1].values
}, index=forecast_index)

# ---------------------------
# PROBABILITY ABOVE THRESHOLD
# ---------------------------
def probability_above_threshold(pred_mean, pred_ci, threshold):
    mean_val = float(pred_mean)
    ci_lower, ci_upper = float(pred_ci[0]), float(pred_ci[1])
    se_est = max(1e-8, (ci_upper - ci_lower) / (2*1.96))
    z = (threshold - mean_val) / se_est
    cdf_at_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob_percent = float(max(0, min(1, 1 - cdf_at_z)) * 100)
    return prob_percent

forecast_df['prob_above_threshold'] = forecast_df.apply(
    lambda row: probability_above_threshold(row[variable], (row['ci_lower'], row['ci_upper']), threshold),
    axis=1
)

# Historical percentage above threshold
hist_percent = float((df_train[variable] > threshold).sum() / len(df_train) * 100)
print(f"Historical percentage above {threshold}: {hist_percent:.2f}%")

# ---------------------------
# PLOT RESULTS
# ---------------------------
plt.figure(figsize=(15,6))
plt.plot(df_train.index, df_train[variable], label='Historical', color='blue')
plt.plot(forecast_df.index, forecast_df[variable], label='Forecast', color='red', linestyle=':')
if not df_test.empty:
    plt.plot(df_test.index, df_test[variable], label='Actual (Test)', color='green')
plt.fill_between(forecast_df.index, forecast_df['ci_lower'], forecast_df['ci_upper'], color='pink', alpha=0.3, label='95% CI')
plt.xlabel('Date')
plt.ylabel(variable)
plt.title(f'ARIMA 6-Month Forecast vs Actual for {variable} at lat={lat}, lon={lon}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# SAVE FORECAST TO CSV
# ---------------------------
forecast_df.to_csv(f"forecast_vs_actual_6months_{variable}.csv")
print("Forecast saved to CSV.")

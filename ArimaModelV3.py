import os
import glob
import math
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# ---------------------------
# PARAMETERS
# ---------------------------
folder_path = "MERRA2_daily"  # folder containing .nc4 files
variable = "T2MMAX"
lat, lon = -90.0, -178.75
train_years = ["2020", "2021"]
test_year = "2022"  # forward test
threshold = 300.0   # example threshold

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

# Split train/test
df_train = df[df.index.year.isin([int(y) for y in train_years])]
df_test = df[df.index.year == int(test_year)]

print("Train rows:", len(df_train))
print("Test rows:", len(df_test))
print(df_train.head())
print(df_test.head())

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
# FORECAST
# ---------------------------
forecast_steps = len(df_test)
forecast = model_fit.forecast(steps=forecast_steps)

# ---------------------------
# PROBABILITY ABOVE THRESHOLD
# ---------------------------
last_train_date = df_train.index.max()
pred = model_fit.get_forecast(steps=forecast_steps)
pred_mean = pred.predicted_mean
conf_int = pred.conf_int(alpha=0.05)
mean_val = float(pred_mean.iloc[-1])
ci_lower = float(conf_int.iloc[-1, 0])
ci_upper = float(conf_int.iloc[-1, 1])
se_est = max(1e-8, (ci_upper - ci_lower) / (2*1.96))
z = (threshold - mean_val) / se_est
cdf_at_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
prob_percent = float(max(0, min(1, 1 - cdf_at_z)) * 100)
hist_percent = float((df_train[variable] > threshold).sum() / len(df_train) * 100)

print(f"\nProbability of exceeding {threshold}: {prob_percent:.2f}%")
print(f"Historical percentage above {threshold}: {hist_percent:.2f}%")

# ---------------------------
# PLOT RESULTS
# ---------------------------
plt.figure(figsize=(15,6))
plt.plot(df_train.index, df_train[variable], label='Train', color='blue')
plt.plot(df_test.index, df_test[variable], label='Test', color='green')
plt.plot(df_test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel(variable)
plt.title(f'ARIMA Forecast for {variable} at lat={lat}, lon={lon}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import xarray as xr
import pandas as pd
import glob
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# ---------------------------
# PARAMETERS
# ---------------------------
folder_path = "MERRA2_daily"  # folder with .nc4 files
lat, lon = -90.0, -178.75     # target grid point
train_years = ['2020', '2021'] # years for training
test_year = '2022'             # year for testing

# Variables that exist in your files
variable_list = ['T2MMAX', 'T2MMEAN', 'T2MMIN']

# ---------------------------
# OPEN ALL DAILY FILES
# ---------------------------
nc_files = glob.glob(f"{folder_path}/*.nc4")
print(f"Found {len(nc_files)} .nc4 files")

# Open multiple files as one xarray dataset
ds = xr.open_mfdataset(nc_files, combine='by_coords')  # sequential read

# ---------------------------
# EXTRACT VARIABLES AT LAT/LON
# ---------------------------
data = {}
for var in variable_list:
    if var in ds:
        data[var] = ds[var].sel(lat=lat, lon=lon, method='nearest')

# Combine variables into pandas DataFrame
df = xr.Dataset(data).to_dataframe()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Keep only relevant columns
columns_to_keep = ['T2MMAX', 'T2MMEAN', 'T2MMIN']
df = df[columns_to_keep]

print("Sample data:")
print(df.head())

# ---------------------------
# SPLIT INTO TRAIN / TEST
# ---------------------------
df_train = df[df.index.year.isin([int(y) for y in train_years])]
df_test = df[df.index.year == int(test_year)]

print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

# ---------------------------
# ARIMA MODELING ON T2MMAX
# ---------------------------
print("Finding optimal ARIMA order for T2MMAX...")
stepwise_model = auto_arima(df_train['T2MMAX'], seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(f"Selected ARIMA order: p={p}, d={d}, q={q}")

# Fit ARIMA
model = ARIMA(df_train['T2MMAX'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# Forecast
forecast_steps = len(df_test)
forecast = model_fit.forecast(steps=forecast_steps)

# ---------------------------
# PLOT RESULTS
# ---------------------------
plt.figure(figsize=(15,6))
plt.plot(df_train.index, df_train['T2MMAX'], label='Train', color='blue')
plt.plot(df_test.index, df_test['T2MMAX'], label='Test', color='green')
plt.plot(df_test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('T2MMAX')
plt.title(f'ARIMA Forecast for T2MMAX at lat={lat}, lon={lon}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

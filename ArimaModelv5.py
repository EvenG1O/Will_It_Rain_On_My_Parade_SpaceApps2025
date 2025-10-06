import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

# ---------------------------
# PARAMETERS
# ---------------------------
train_years = ["2020", "2021"]
test_year = "2022"
n_forecast = 7  # 6 months ahead

# ---------------------------
# GENERATE EXAMPLE DATA (monthly temps)
# ---------------------------
dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="M")
# Synthetic seasonal temperature: winter cold, summer hot
temperature = 10 + 15*np.sin(2*np.pi*dates.month/12) + np.random.normal(0, 2, len(dates))
df = pd.DataFrame({"temperature": temperature}, index=dates)

# ---------------------------
# SPLIT TRAIN/TEST
# ---------------------------
df_train = df[df.index.year.isin([int(y) for y in train_years])]
df_test = df[df.index.year == int(test_year)]

# ---------------------------
# FIT SARIMA MODEL (captures seasonality)
# ---------------------------
# Seasonal period = 12 months
stepwise_model = auto_arima(
    df_train['temperature'],
    seasonal=True,
    m=12,
    D=1,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

print("Selected SARIMA order:", stepwise_model.order, "seasonal_order:", stepwise_model.seasonal_order)

# Fit SARIMAX (from statsmodels for easier forecasting)
model = SARIMAX(df_train['temperature'], 
                order=stepwise_model.order, 
                seasonal_order=stepwise_model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit()
print(model_fit.summary())

# ---------------------------
# FORECAST NEXT 6 MONTHS
# ---------------------------
forecast_index = pd.date_range(start=df_train.index[-1]+pd.DateOffset(months=1), periods=n_forecast, freq='M')
forecast = model_fit.get_forecast(steps=n_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# ---------------------------
# PLOT RESULTS
# ---------------------------
plt.figure(figsize=(12,5))
plt.plot(df_train.index, df_train['temperature'], label='Train (2020-2021)', color='blue')
plt.plot(df_test.index, df_test['temperature'], label='Test (2022)', color='green')
plt.plot(forecast_index, forecast_mean, label='Forecast (6 months ahead)', color='red', linestyle='--')
plt.fill_between(forecast_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Forecast Example')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# SAVE FORECAST TO JSON
# ---------------------------
forecast_list = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(f)} for d, f in zip(forecast_index, forecast_mean)]
with open("forecast_6months.json", "w") as f:
    json.dump(forecast_list, f, indent=2)

print("Forecast saved to 'forecast_6months.json'")


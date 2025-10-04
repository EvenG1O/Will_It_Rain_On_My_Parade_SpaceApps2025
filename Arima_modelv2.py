import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the database
db_path = "merra2_daily.db"
conn = sqlite3.connect(db_path)

# Table name
table_name = 'weather'

# Parameters
target_lat = -89.5
target_lon = 81.875
year = 2020

# SQL query: filter by lat, lon, and year
query = f"""
SELECT date, T2MMAX, T2MMEAN, T2MMIN
FROM {table_name}
WHERE lat = {target_lat} 
  AND lon = {target_lon} 
  AND strftime('%Y', date) = '{year}'
ORDER BY date
"""

# Load data into DataFrame
df = pd.read_sql_query(query, conn)
conn.close()

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Ensure data is sorted by date
df = df.sort_values('date')

# Preview first 10 rows
print(df.head(10))

# Columns
print(df.columns)

# Total rows
print("Total rows:", len(df))

# --- Plotting ---
plt.figure(figsize=(15,6))

# Plot temperature variables
plt.plot(df['date'], df['T2MMAX'], label='T2MMAX', color='red')
plt.plot(df['date'], df['T2MMEAN'], label='T2MMEAN', color='blue')
plt.plot(df['date'], df['T2MMIN'], label='T2MMIN', color='green')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title(f'Weather Temperatures at lat={target_lat}, lon={target_lon} for {year}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

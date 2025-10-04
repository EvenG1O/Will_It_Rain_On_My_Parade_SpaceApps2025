import sqlite3
import xarray as xr
from pathlib import Path

# Folder with downloaded .nc4 files
data_dir = Path("MERRA2_daily")

# SQLite database
conn = sqlite3.connect("merra2_daily.db")
cursor = conn.cursor()

# Create a table for your data
cursor.execute("""
CREATE TABLE IF NOT EXISTS weather (
    date TEXT,
    lat REAL,
    lon REAL,
    T2MMEAN REAL,
    T2MMAX REAL,
    T2MMIN REAL,
    HOURNORAIN REAL
)
""")

# Loop through files
for nc_file in data_dir.glob("*.nc4"):
    ds = xr.open_dataset(nc_file)
    date = str(ds.time.values[0])[:10]  # YYYY-MM-DD
    
    # Flatten the arrays
    lats = ds.lat.values
    lons = ds.lon.values
    T2MMEAN = ds.T2MMEAN.values[0]  # first (and only) timestep
    T2MMAX = ds.T2MMAX.values[0]
    T2MMIN = ds.T2MMIN.values[0]
    HOURNORAIN = ds.HOURNORAIN.values[0]
    
    # Insert all grid points
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cursor.execute("""
            INSERT INTO weather (date, lat, lon, T2MMEAN, T2MMAX, T2MMIN, HOURNORAIN)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date, lat, lon, float(T2MMEAN[i,j]), float(T2MMAX[i,j]), float(T2MMIN[i,j]), float(HOURNORAIN[i,j])))

conn.commit()
conn.close()

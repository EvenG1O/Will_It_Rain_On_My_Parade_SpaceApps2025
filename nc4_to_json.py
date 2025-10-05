import xarray as xr
import json

# Load the .nc4 file
ds = xr.open_dataset("MERRA2_daily/MERRA2_400.statD_2d_slv_Nx.20191228.nc4")

# Select variable, e.g., T2MMAX
variable = "T2MMAX"
da = ds[variable].isel(time=0)  # pick first time step for simplicity

# Convert to 2D array and coordinates
heatmap_data = {
    "lat": da.coords["lat"].values.tolist(),   # <--- fixed
    "lon": da.coords["lon"].values.tolist(),   # <--- fixed
    "values": da.values.tolist()               # DataArray to list
}

# Save as JSON
with open("heatmap.json", "w") as f:
    json.dump(heatmap_data, f)

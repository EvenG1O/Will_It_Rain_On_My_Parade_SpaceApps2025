# heatmap_app.py
import os
import glob
from flask import Flask, request, jsonify
import xarray as xr
import pandas as pd

app = Flask(__name__)

def find_nc_files(folder_path):
    p = os.path.expanduser(folder_path)
    files = sorted(glob.glob(os.path.join(p, "*.nc4")))
    return files

def get_heatmap_data(nc_files, variable):
    if not nc_files:
        raise FileNotFoundError("No .nc4 files found in folder.")
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    
    # Take last time step
    da = ds[variable].isel(time=-1)
    lats = da["lat"].values
    lons = da["lon"].values
    data = da.values

    heatmap = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            heatmap.append({"lat": float(lat), "lon": float(lon), variable: float(data[i,j])})
    return heatmap

@app.route("/heatmap", methods=["GET"])
def heatmap():
    folder_path = request.args.get("folder_path", "MERRA2_daily")
    variable = request.args.get("variable", "T2MMAX")
    try:
        nc_files = find_nc_files(folder_path)
        data = get_heatmap_data(nc_files, variable)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

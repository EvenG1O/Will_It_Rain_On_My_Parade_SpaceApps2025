import os
import glob
import xarray as xr
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from React frontend

# ---------------------------
# Configuration
# ---------------------------
FOLDER_PATH = "MERRA2_daily"  # folder containing .nc4 files

# ---------------------------
# Helper to find .nc4 files
# ---------------------------
def find_nc_files(folder_path):
    p = os.path.expanduser(folder_path)
    files = sorted(glob.glob(os.path.join(p, "*.nc4")))
    if not files:
        raise FileNotFoundError(f"No .nc4 files found in '{folder_path}'")
    return files

# ---------------------------
# Convert nc4 to flat GridPoint array
# ---------------------------
def nc_to_gridpoints(nc_files, variable):
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    
    da = ds[variable]

    # Flatten all lat/lon points (latest timestep)
    latest = da.isel(time=-1)  # take last time index
    lats = latest.lat.values
    lons = latest.lon.values
    values = latest.values  # 2D array lat x lon

    data = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            data.append({
                "lat": float(lat),
                "lon": float(lon),
                "value": float(values[i, j])
            })
    return data

# ---------------------------
# Flask endpoint
# ---------------------------
@app.route("/heatmap")
def heatmap():
    variable = request.args.get("variable", "T2MMAX")  # default variable
    date = request.args.get("date")  # optional, not used in this example

    try:
        nc_files = find_nc_files(FOLDER_PATH)
        gridpoints = nc_to_gridpoints(nc_files, variable)
        return jsonify(gridpoints)
    except Exception as e:
        return jsonify({"error": str(e), "data": []})

# ---------------------------
# Run server
# ---------------------------


if __name__ == "__main__":
    # Listen on all interfaces and turn off debug for production
    app.run(host="0.0.0.0", port=5000)

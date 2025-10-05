# File to explore how our .nc4  look like  and data they contain
import xarray as xr
import matplotlib.pyplot as plt

# ---- Path to your .nc4 file ----
FILE = "MERRA2_daily/MERRA2_400.statD_2d_slv_Nx.20191228.nc4"  

# ---- Load the dataset ----
ds = xr.open_dataset(FILE)
print("\n📂 VARIABLES & STRUCTURE\n")
print(ds)

# ---- List all variable names ----
print("\nAvailable variables:\n", list(ds.data_vars))

# ---- Example: look at one variable ----
var = "T2M"  # change to the variable you want (like 'PRECTOT', 'T2M', 'U10M', etc.)
if var in ds:
    data = ds[var]
    print(f"\nVariable {var}:\n", data)
else:
    print(f"\n❌ Variable {var} not found. Choose one from {list(ds.data_vars)[:10]}")

# ---- Plot first time slice as a map ----
if var in ds:
    # select first time index (e.g. daily/hourly)
    slice_ = data.isel(time=0)

    plt.figure(figsize=(10, 5))
    slice_.plot(cmap="coolwarm")
    plt.title(f"{var} on {str(data.time.values[0])[:10]}")
    plt.show()

import pandas as pd

h5_path = "IEX_data/resampled.h5"

# Open the HDF5 store and list all keys
with pd.HDFStore(h5_path, mode="r") as store:
    print("Keys in file:", store.keys())
    df = store.select(store.keys()[0], stop=5)
    print("Columns:", df.columns.tolist())
    print(df.head())
import pandas as pd

h5_path = "IEX_data/resampled.h5"

# Open the HDF5 store and list all keys
with pd.HDFStore(h5_path, mode="r") as store:
    print("Keys in file:", store.keys())

    # Read a small sample from the first dataset
    key = store.keys()[0] if store.keys() else None
    if key:
        print(f"\nPreview of {key}:")
        df = store.select(key, start=0, stop=5)
        print(df)
        print("\nColumn names:", list(df.columns))
        print("\nDataFrame info:")
        print(df.info())
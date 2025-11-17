import pandas as pd
if __name__ == "__main__":

    with pd.HDFStore("Data/raw_orderbook.h5", "r") as s:
        print(s.get_storer("deep").nrows)
    
    with pd.HDFStore("Data/raw_orderbook.h5", "r") as s:
        df0 = s.select("deep", start=0, stop=5)
        print(df0.columns)
        print(df0.head())
        
    print("Inspecting resampled data:") 
    df = pd.read_hdf("Data/resampled.h5", "deep")
    print(df.shape)
    print(df.head())
    print(df.tail())
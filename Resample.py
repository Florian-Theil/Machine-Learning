    if __name__ == "__main__":
    import sys
    import pandas as pd

    if len(sys.argv) < 2:
        print("Usage: python3 Resample.py input.h5 output.h5")
        sys.exit(1)
    
    inputfile = sys.argv[1];
    outputfile = sys.argv[2];
    
    print(f"Resampling {inputfile} to {outputfile}")
    df = pd.read_hdf(inputfile, "deep")
    df_rs = resample(df, freq="100ms")
    df_rs.to_hdf(outputfile, key="deep", format="table")
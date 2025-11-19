if __name__ == "__main__":
    import sys
    import pandas as pd

    if len(sys.argv) < 3:
        print("Usage: python3 Resample.py input.h5 output.h5")
        sys.exit(1)

    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    print(f"Resampling {inputfile} to {outputfile}")

    store = pd.HDFStore(inputfile, mode="r")
    chunksize = 5_000_000
    iter = store.select("deep", chunksize=chunksize)
    num_rows = store.get_storer("deep").nrows
    print(f"Total number of chunks to process: {num_rows/chunksize}")
    first = True
    count = 0
    
    for chunk in iter:
        count += 1
        print
        chunk["ts"] = pd.to_datetime(chunk["ts"])
        chunk = chunk.set_index("ts")

        # Remove duplicate timestamps (IEX often sends many with identical ns)
        chunk = chunk[~chunk.index.duplicated(keep="last")]

        # Now safe to resample
        chunk_rs = (
            chunk
            .resample("100ms")
            .ffill()
            .dropna(how="all")    # <–– add this line here
        )

        # chunk_rs = chunk.resample("100ms").ffill()

        chunk_rs.to_hdf(
            outputfile,
            key="deep",
            format="table",
            append=not first,
        )
        first = False

    store.close()
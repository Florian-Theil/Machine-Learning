import pandas as pd
from sklearn.decomposition import PCA

# === PARAMETERS ===
filename = "Dataresampled.h5"
key = "deep"
min_events = 5000      # ignore symbols with fewer samples
n_components = 20      # number of principal components to consider
top_k = 20             # how many symbols to output

print("Loading data...")
df = pd.read_hdf(filename, key).dropna(subset=["symbol", "mid"])

# === Filter by liquidity ===
print("Filtering symbols...")
counts = df["symbol"].value_counts()
symbols = counts[counts > min_events].index
df = df[df["symbol"].isin(symbols)]

print(f"Remaining symbols: {len(symbols)}")

# === Pivot to wide format ===
print("Pivoting to wide matrix...")
wide = df.pivot_table(index=df.index, columns="symbol", values="mid")

# === Compute returns ===
print("Computing returns...")
returns = wide.pct_change().dropna().fillna(0)

# === Run PCA using randomized SVD ===
print("Running PCA...")
pca = PCA(n_components=n_components, svd_solver="randomized")
pca.fit(returns)

# === Compute importance scores ===
# Each component gives a loading per symbol; sum absolute contributions
loadings = abs(pca.components_).sum(axis=0)
scores = pd.Series(loadings, index=returns.columns)

# === Output top symbols ===
ranking = scores.sort_values(ascending=False).head(top_k)
print("\nTop symbols by PCA importance:")
print(ranking)

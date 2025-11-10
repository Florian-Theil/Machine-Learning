def momentum_strategy(df, window=20, threshold=0.002, trade_size=100):
    trades = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        g["ma"] = g["mid"].rolling(window, min_periods=1).mean()
        for _, row in g.iterrows():
            if row["mid"] > (1 + threshold) * row["ma"]:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "buy", "size": trade_size})
            elif row["mid"] < (1 - threshold) * row["ma"]:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "sell", "size": trade_size})
    return trades


def random_strategy(df, prob=0.0005, trade_size=100):
    np.random.seed(42)
    trades = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        for _, row in g.iterrows():
            r = np.random.rand()
            if r < prob:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "buy", "size": trade_size})
            elif r < 2 * prob:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "sell", "size": trade_size})
    return trades
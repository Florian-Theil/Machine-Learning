import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def evaluate_strategies(
    h5_filename,
    dataset_name,
    strategies,
    start_time=None,
    end_time=None,
    initial_cash=0.0,
    transaction_cost=0.0001,
    plot=False,
):
    """Fast multi-strategy evaluator with P&L decomposition."""

    df = pd.read_hdf(h5_filename, dataset_name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    if start_time:
        df = df[df["timestamp"] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df["timestamp"] <= pd.to_datetime(end_time)]

    # --- Pre-group data by symbol ---
    data_by_symbol = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        data_by_symbol[sym] = {
            "ts_array": g["timestamp"].values.astype("datetime64[ns]"),
            "best_bid": g["best_bid"].to_numpy(),
            "best_ask": g["best_ask"].to_numpy(),
            "mid": g["mid"].to_numpy(),
        }

    def _evaluate_one(strategy_func):
        trades = strategy_func(df)
        if not trades:
            cols = ["strategy_outcome", "spread_loss", "transaction_cost", "PnL_total"]
            return pd.DataFrame(columns=cols).set_index(pd.DatetimeIndex([], name="timestamp"))

        trades_df = pd.DataFrame(trades)
        trades_df["timestamp"] = pd.to_datetime(
            trades_df.get("timestamp", df["timestamp"].iloc[0])
        )
        trades_df = trades_df.sort_values("timestamp")

        cash_actual = initial_cash
        cash_mid = initial_cash
        pos_actual, pos_mid = {}, {}
        spread_loss_cum = 0.0
        tx_cost_cum = 0.0
        rec = []
        last_mid = {}

        # Group trades by symbol
        for sym, tgroup in trades_df.groupby("symbol"):
            if sym not in data_by_symbol:
                continue
            gdata = data_by_symbol[sym]
            ts_array = gdata["ts_array"]
            bids = gdata["best_bid"]
            asks = gdata["best_ask"]
            mids = gdata["mid"]

            for _, tr in tgroup.iterrows():
                side = tr["side"]
                size = float(tr["size"])
                ts_val = np.datetime64(tr["timestamp"], "ns")

                idx = np.searchsorted(ts_array, ts_val)
                if idx >= len(ts_array):
                    idx = len(ts_array) - 1

                bid = bids[idx]
                ask = asks[idx]
                mid = mids[idx]
                spread = ask - bid

                # Actual execution
                exec_price = ask if side == "buy" else bid
                if side == "buy":
                    cash_actual -= exec_price * size * (1 + transaction_cost)
                    pos_actual[sym] = pos_actual.get(sym, 0.0) + size
                else:
                    cash_actual += exec_price * size * (1 - transaction_cost)
                    pos_actual[sym] = pos_actual.get(sym, 0.0) - size

                # Mid execution (strategy outcome)
                if side == "buy":
                    cash_mid -= mid * size
                    pos_mid[sym] = pos_mid.get(sym, 0.0) + size
                else:
                    cash_mid += mid * size
                    pos_mid[sym] = pos_mid.get(sym, 0.0) - size

                spread_loss_cum += 0.5 * spread * size
                tx_cost_cum += transaction_cost * exec_price * size
                last_mid[sym] = mid

                mtm_actual = sum(q * last_mid[s] for s, q in pos_actual.items() if s in last_mid)
                mtm_mid = sum(q * last_mid[s] for s, q in pos_mid.items() if s in last_mid)
                strategy_outcome = (cash_mid + mtm_mid) - initial_cash
                pnl_total = (cash_actual + mtm_actual) - initial_cash

                rec.append(
                    (
                        tr["timestamp"],
                        strategy_outcome,
                        -abs(spread_loss_cum),
                        -abs(tx_cost_cum),
                        pnl_total,
                    )
                )

        # Final liquidation
        final_t = df["timestamp"].iloc[-1]
        mtm_actual = sum(
            q * df.loc[df["symbol"] == s, "mid"].iloc[-1]
            for s, q in pos_actual.items()
            if s in df["symbol"].values
        )
        mtm_mid = sum(
            q * df.loc[df["symbol"] == s, "mid"].iloc[-1]
            for s, q in pos_mid.items()
            if s in df["symbol"].values
        )
        strategy_outcome = (cash_mid + mtm_mid) - initial_cash
        pnl_total = (cash_actual + mtm_actual) - initial_cash

        rec.append(
            (
                final_t,
                strategy_outcome,
                -abs(spread_loss_cum),
                -abs(tx_cost_cum),
                pnl_total,
            )
        )

        out = pd.DataFrame(
            rec,
            columns=[
                "timestamp",
                "strategy_outcome",
                "spread_loss",
                "transaction_cost",
                "PnL_total",
            ],
        ).set_index("timestamp")

        out = out[~out.index.duplicated(keep="last")]
        return out

    components = ["strategy_outcome", "spread_loss", "transaction_cost", "PnL_total"]
    results = {c: pd.DataFrame() for c in components}

    for name, func in strategies.items():
        print(f"Evaluating strategy: {name}")
        comp_df = _evaluate_one(func)
        for c in components:
            results[c][name] = comp_df[c]

    for c in components:
        if not results[c].empty:
            results[c].index = pd.to_datetime(results[c].index)
            results[c] = results[c].sort_index()
            results[c] = results[c][~results[c].index.duplicated(keep="last")]
            results[c] = results[c].interpolate(method="time")

    if plot and HAVE_MPL:
        plt.figure(figsize=(9, 4))
        results["PnL_total"].plot(title="Total P&L (BBO exec + costs, marked to mid)")
        plt.xlabel("Time")
        plt.ylabel("P&L")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(9, 4))
        results["strategy_outcome"].plot(title="Strategy Outcome at Mid, Zero Cost (sign-symmetric)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
    elif plot and not HAVE_MPL:
        print("[Info] matplotlib not available — skipping plot.")

    return results


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


def contrarian_strategy(df, window=20, threshold=0.002, trade_size=100):
    trades = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        g["ma"] = g["mid"].rolling(window, min_periods=1).mean()
        for _, row in g.iterrows():
            if row["mid"] > (1 + threshold) * row["ma"]:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "sell", "size": trade_size})
            elif row["mid"] < (1 - threshold) * row["ma"]:
                trades.append({"timestamp": row["timestamp"], "symbol": sym, "side": "buy", "size": trade_size})
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


if __name__ == "__main__":
    strategies = {
        "Momentum": momentum_strategy,
        "Contrarian": contrarian_strategy,
        "Random": random_strategy,
    }

    results = evaluate_strategies(
        h5_filename="IEX_data/resampled.h5",
        dataset_name="/deep",
        strategies=strategies,
        start_time="2025-10-17 12:00:00",
        end_time="2025-10-17 12:00:10",
        initial_cash=1_000_000.0,
        transaction_cost=0.0001,
        plot=False,
    )

    for k, df_k in results.items():
        print(f"\n--- {k} ---")
        print(df_k.tail())

    if (
        "strategy_outcome" in results
        and "Momentum" in results["strategy_outcome"].columns
        and "Contrarian" in results["strategy_outcome"].columns
    ):
        s_m = results["strategy_outcome"]["Momentum"].iloc[-1]
        s_c = results["strategy_outcome"]["Contrarian"].iloc[-1]
        print(
            f"\nSymmetry check:\n"
            f"  StrategyOutcome(Momentum)    = {s_m:.2f}\n"
            f"  StrategyOutcome(Contrarian)  = {s_c:.2f}\n"
            f"  Sum (should be ≈ 0)          = {s_m + s_c:.2e}"
        )

    for comp, df_comp in results.items():
        out_name = f"pnl_{comp}.csv"
        df_comp.to_csv(out_name)
        print(f"[Saved] {out_name} with shape {df_comp.shape}")
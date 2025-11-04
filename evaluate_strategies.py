import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate_strategies(h5_filename, dataset_name, strategies,
                        start_time=None, end_time=None,
                        initial_cash=0.0, transaction_cost=0.0001,
                        plot=True):
    """
    Evaluate several trading strategies on IEX HDF5 data.

    Parameters
    ----------
    h5_filename : str
        Path to the HDF5 file.
    dataset_name : str
        Dataset key (e.g. '/deep').
    strategies : dict[str, callable]
        Mapping of strategy names to functions df -> list of trade dicts.
    start_time, end_time : str or datetime, optional
        Restrict the evaluation period.
    initial_cash : float
        Starting balance.
    transaction_cost : float
        Proportional transaction cost (e.g. 0.0001 = 0.01%).
    plot : bool
        Whether to plot all P&L curves.

    Returns
    -------
    pnl_df : pd.DataFrame
        Columns are strategy names; index is timestamp.
    """

    # --- Load data ---
    df = pd.read_hdf(h5_filename, dataset_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    if start_time:
        df = df[df['timestamp'] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df['timestamp'] <= pd.to_datetime(end_time)]

    # --- Helper: single strategy evaluator returning PnL series ---
    def _evaluate_one(strategy_func):
        trades = strategy_func(df)
        if not trades:
            return pd.Series(dtype=float)

        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df.get('timestamp', df['timestamp'].iloc[0]))
        trades_df = trades_df.sort_values('timestamp')

        cash = initial_cash
        positions = {}
        pnl_records = []

        for _, trade in trades_df.iterrows():
            sym = trade['symbol']
            side = trade['side']
            size = trade['size']

            # nearest market snapshot
            snapshot = df[df['symbol'] == sym]
            if snapshot.empty:
                continue
            idx = snapshot['timestamp'].searchsorted(trade['timestamp'])
            idx = min(idx, len(snapshot) - 1)
            row = snapshot.iloc[idx]

            if side == 'buy':
                exec_price = row['best_ask']
                cash -= exec_price * size * (1 + transaction_cost)
                positions[sym] = positions.get(sym, 0) + size
            elif side == 'sell':
                exec_price = row['best_bid']
                cash += exec_price * size * (1 - transaction_cost)
                positions[sym] = positions.get(sym, 0) - size

            # mark to market
            mtm = 0.0
            for s, qty in positions.items():
                if qty == 0:
                    continue
                local = df[df['symbol'] == s]
                mid = local.iloc[min(idx, len(local)-1)]['mid']
                mtm += qty * mid
            total_value = cash + mtm
            pnl_records.append((trade['timestamp'], total_value - initial_cash))

        # final liquidation
        final_time = df['timestamp'].iloc[-1]
        mtm = 0.0
        for s, qty in positions.items():
            if qty != 0:
                current_mid = df[df['symbol'] == s]['mid'].iloc[-1]
                mtm += qty * current_mid
        pnl_records.append((final_time, cash + mtm - initial_cash))

        return pd.Series(
            [p for _, p in pnl_records],
            index=[t for t, _ in pnl_records],
            name='PnL'
        )

    # --- Run all strategies ---
    pnl_df = pd.DataFrame()
    for name, func in strategies.items():
        pnl_df[name] = _evaluate_one(func)

    # align on common time index
    pnl_df = pnl_df.sort_index().interpolate(method='time')

    # --- Plot ---
    if plot and not pnl_df.empty:
        plt.figure(figsize=(8, 4))
        pnl_df.plot(title="Cumulative P&L Comparison")
        plt.xlabel("Time")
        plt.ylabel("P&L")
        plt.grid(True)
        plt.show()

    return pnl_df


# --- Example strategies ---

def moving_average_strategy(df, window=20, threshold=0.005, trade_size=100):
    """Mean reversion: buy if mid < (1-threshold)*MA, sell if > (1+threshold)*MA."""
    trades = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp')
        g['ma'] = g['mid'].rolling(window, min_periods=1).mean()
        for _, row in g.iterrows():
            if row['mid'] < (1 - threshold)*row['ma']:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'buy', 'size': trade_size})
            elif row['mid'] > (1 + threshold)*row['ma']:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'sell', 'size': trade_size})
    return trades


def momentum_strategy(df, window=20, threshold=0.002, trade_size=100):
    """Momentum: buy if mid > MA*(1+threshold), sell if mid < MA*(1-threshold)."""
    trades = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp')
        g['ma'] = g['mid'].rolling(window, min_periods=1).mean()
        for _, row in g.iterrows():
            if row['mid'] > (1 + threshold)*row['ma']:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'buy', 'size': trade_size})
            elif row['mid'] < (1 - threshold)*row['ma']:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'sell', 'size': trade_size})
    return trades


def random_strategy(df, prob=0.0005, trade_size=100):
    """Random buy/sell decisions for testing."""
    np.random.seed(42)
    trades = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp')
        for _, row in g.iterrows():
            r = np.random.rand()
            if r < prob:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'buy', 'size': trade_size})
            elif r < 2*prob:
                trades.append({'timestamp': row['timestamp'], 'symbol': sym, 'side': 'sell', 'size': trade_size})
    return trades


# --- Example run ---
if __name__ == "__main__":
    strategies = {
        "MeanReversion": moving_average_strategy,
        "Momentum": momentum_strategy,
        "Random": random_strategy,
    }

    pnl_df = evaluate_strategies(
        h5_filename="data.h5",
        dataset_name="/deep",
        strategies=strategies,
        start_time="2025-10-17 09:30:00",
        end_time="2025-10-17 16:00:00",
        initial_cash=1_000_000.0,
        transaction_cost=0.0001,
        plot=True
    )

    print(pnl_df.tail())
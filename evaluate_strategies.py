from compute_pnl import compute_pnl
from strategies import momentum_strategy, random_strategy

strategies = {
    "Momentum": momentum_strategy,
    "Random": random_strategy,
}

results = compute_pnl(
    h5_filename="data/resampled.h5",
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

    # Combine all P&L components into a single DataFrame with MultiIndex columns
    combined = pd.concat(results, axis=1)  # outer keys = components, inner = strategies

    # Ensure sorted timestamps and clean index
    combined = combined.sort_index()
    combined.index.name = "timestamp"

    # Save to a single Excel file
    output_file = "pnl_results_combined.xlsx"
    combined.to_excel(output_file, engine="xlsxwriter")

    print(f"[Saved combined results to] {output_file}")
    print(f"Shape: {combined.shape}")
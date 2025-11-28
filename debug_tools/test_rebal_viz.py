import pandas as pd
import numpy as np

# Mock Data
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
trades_df = pd.DataFrame({
    "Date": dates,
    "Realized P&L": np.random.randn(len(dates)) * 1000,
    "Realized ST P&L": np.random.randn(len(dates)) * 500,
    "Realized LT P&L": np.random.randn(len(dates)) * 500
})

unrealized_pl_df = pd.DataFrame({
    "Unrealized P&L": np.random.randn(len(dates)) * 10000
}, index=dates)

def test_aggregation(view_freq):
    print(f"Testing {view_freq}...")
    df_chart = trades_df.copy()
    
    if view_freq == "Quarterly":
        df_chart["Quarter"] = df_chart["Date"].dt.to_period("Q")
        agg_df = df_chart.groupby("Quarter")[["Realized P&L", "Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        
        # Merge Unrealized
        unrealized_q = unrealized_pl_df.resample("Q").last()
        unrealized_q.index = unrealized_q.index.to_period("Q")
        
        # Check indices
        print("Agg Index Type:", type(agg_df.index))
        print("Unrealized Index Type:", type(unrealized_q.index))
        
        agg_df = agg_df.join(unrealized_q[["Unrealized P&L"]], how="outer").fillna(0.0)
        print("Joined Index Type:", type(agg_df.index))
        print(agg_df.head())
        
        x_axis = agg_df.index.astype(str)
        print("X-Axis:", x_axis)

    elif view_freq == "Monthly":
        df_chart["Month"] = df_chart["Date"].dt.to_period("M")
        agg_df = df_chart.groupby("Month")[["Realized P&L", "Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        
        unrealized_m = unrealized_pl_df.resample("M").last()
        unrealized_m.index = unrealized_m.index.to_period("M")
        
        agg_df = agg_df.join(unrealized_m[["Unrealized P&L"]], how="outer").fillna(0.0)
        
        x_axis = agg_df.index.astype(str)
        print("X-Axis:", x_axis)

test_aggregation("Quarterly")
test_aggregation("Monthly")

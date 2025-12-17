import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def generate_html_report(results):
    """
    Generates a standalone HTML report from the backtest results.
    """
    # Extract Data
    port_series = results.get("port_series", pd.DataFrame())
    stats = results.get("stats", {})
    trades_df = results.get("trades_df", pd.DataFrame())
    pl_by_year = results.get("pl_by_year", pd.DataFrame())
    
    if port_series.empty:
        return "<html><body><h1>No Data Available</h1></body></html>"
        
    # --- Generate Charts for Report ---
    
    # 1. Equity Curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=port_series.index, 
        y=port_series, 
        mode='lines', 
        name='Portfolio Value',
        line=dict(color='#00CC96', width=2)
    ))
    fig_equity.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        height=400
    )
    html_equity = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 2. Drawdown Chart
    # Calculate Drawdown
    rolling_max = port_series.cummax()
    drawdown = (port_series - rolling_max) / rolling_max
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=port_series.index, 
        y=drawdown, 
        mode='lines', 
        name='Drawdown',
        line=dict(color='#EF553B', width=1),
        fill='tozeroy'
    ))
    fig_dd.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=300
    )
    html_dd = fig_dd.to_html(full_html=False, include_plotlyjs=False) # JS already included
    
    # 3. Annual Returns Bar Chart
    # Calculate Annual Returns
    yearly_resampled = port_series.resample("YE").last()
    yearly_returns = yearly_resampled.pct_change().dropna()
    yearly_returns.index = yearly_returns.index.year
    
    fig_annual = go.Figure()
    fig_annual.add_trace(go.Bar(
        x=yearly_returns.index,
        y=yearly_returns,
        marker_color=["#00CC96" if x > 0 else "#EF553B" for x in yearly_returns],
        name="Annual Return"
    ))
    fig_annual.update_layout(
        title="Annual Returns",
        xaxis_title="Year",
        yaxis_title="Return",
        template="plotly_white",
        height=300
    )
    html_annual = fig_annual.to_html(full_html=False, include_plotlyjs=False)
    
    # 4. Margin Usage Chart (if available)
    html_margin = ""
    margin_stats_html = ""
    
    usage_series = results.get("usage_series")
    if usage_series is not None and not usage_series.empty:
        # Generate Chart
        fig_margin = go.Figure()
        fig_margin.add_trace(go.Scatter(
            x=usage_series.index,
            y=usage_series,
            mode='lines',
            name='Margin Usage',
            line=dict(color='#FFA15A', width=1),
            fill='tozeroy'
        ))
        # Add 100% line
        fig_margin.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Margin Call (100%)")
        
        fig_margin.update_layout(
            title="Margin Usage",
            xaxis_title="Date",
            yaxis_title="Usage Ratio",
            template="plotly_white",
            height=300
        )
        html_margin = f"""
        <h2>Margin Analysis</h2>
        <div class="chart-container">
            {fig_margin.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        """
        
        # Calculate Margin Stats
        max_usage = usage_series.max()
        avg_usage = usage_series.mean()
        margin_calls = len(usage_series[usage_series >= 1.0])
        time_in_margin = len(usage_series[usage_series > 0]) / len(usage_series)
        
        margin_stats_html = f"""
        <div class='stats-grid'>
            <div class='stat-card'>
                <div class='stat-label'>Max Usage</div>
                <div class='stat-value'>{max_usage:.2%}</div>
            </div>
            <div class='stat-card'>
                <div class='stat-label'>Avg Usage</div>
                <div class='stat-value'>{avg_usage:.2%}</div>
            </div>
            <div class='stat-card'>
                <div class='stat-label'>Margin Calls</div>
                <div class='stat-value'>{margin_calls}</div>
            </div>
            <div class='stat-card'>
                <div class='stat-label'>Time in Margin</div>
                <div class='stat-value'>{time_in_margin:.1%}</div>
            </div>
        </div>
        """

    # --- Generate HTML Content ---
    
    # Format Stats
    stats_html = "<div class='stats-grid'>"
    for k, v in stats.items():
        # Clean up keys
        label = k.replace("_", " ").title()
        # Format values
        if isinstance(v, float):
            val_str = f"{v:.2f}"
            if "pct" in k.lower() or "cagr" in k.lower() or "drawdown" in k.lower():
                val_str += "%"
        else:
            val_str = str(v)
            
        stats_html += f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{val_str}</div>
        </div>
        """
    stats_html += "</div>"
    
    # Format Tax Table (if available)
    tax_html = ""
    if not pl_by_year.empty:
        # Convert to HTML table
        tax_table = pl_by_year.copy()
        # Format columns
        for col in tax_table.columns:
            if tax_table[col].dtype == float:
                tax_table[col] = tax_table[col].apply(lambda x: f"${x:,.2f}")
        
        tax_html = f"""
        <h2>Tax Analysis</h2>
        <div class='table-container'>
            {tax_table.to_html(classes='styled-table')}
        </div>
        """

    # Full HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 20px; background: #f4f4f9; }}
            .container {{ max_width: 1000px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #eee; }}
            .stat-label {{ font-size: 0.85em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
            .stat-value {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
            .chart-container {{ margin-bottom: 40px; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            .table-container {{ overflow-x: auto; }}
            .styled-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 0.9em; }}
            .styled-table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #dddddd; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #009879; }}
            
            @media print {{
                body {{ background: white; padding: 0; }}
                .container {{ box-shadow: none; padding: 0; max-width: 100%; }}
                .chart-container {{ break-inside: avoid; }}
                h1, h2 {{ page-break-after: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            {stats_html}
            
            <h2>Equity Curve</h2>
            <div class="chart-container">
                {html_equity}
            </div>
            
            <h2>Drawdown</h2>
            <div class="chart-container">
                {html_dd}
            </div>
            
            <h2>Annual Returns</h2>
            <div class="chart-container">
                {html_annual}
            </div>
            
            {html_margin}
            {margin_stats_html}
            
            {tax_html}
            
            <div style="margin-top: 50px; font-size: 0.8em; color: #777; text-align: center;">
                <p>Generated by Testfol Margin Stresser</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

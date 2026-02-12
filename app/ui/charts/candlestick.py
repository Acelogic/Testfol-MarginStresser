import streamlit as st
import pandas as pd
from lightweight_charts.widgets import StreamlitChart


@st.cache_data(show_spinner=False)
def _prepare_candlestick_data(ohlc_df, equity_series, loan_series,  # noqa: ARG001 (kept for cache key)
                               usage_series, equity_pct_series,
                               bench_series, comparison_series,
                               effective_rate_series=None):
    """Compute SMAs and format DataFrames for lightweight-charts. Cached."""
    _ = equity_series, loan_series  # unused but part of cache key from caller
    # OHLC in lightweight-charts format (lowercase + 'time' column)
    candle_df = pd.DataFrame({
        'time': ohlc_df.index,
        'open': ohlc_df['Open'].values,
        'high': ohlc_df['High'].values,
        'low': ohlc_df['Low'].values,
        'close': ohlc_df['Close'].values,
    })

    # SMAs
    close = ohlc_df['Close']
    sma_20 = close.rolling(window=min(20, len(ohlc_df))).mean().dropna()
    sma_50 = close.rolling(window=min(50, len(ohlc_df))).mean().dropna()
    sma_200 = close.rolling(window=min(200, len(ohlc_df))).mean().dropna()

    # Align all overlay series to candle timestamps so lines connect properly
    candle_idx = ohlc_df.index

    def _to_line_df(series, col_name):
        """Convert a pandas Series to a line DataFrame for lightweight-charts."""
        if series is None or (hasattr(series, 'empty') and series.empty):
            return None
        aligned = series.reindex(candle_idx, method='nearest',
                                  tolerance=pd.Timedelta(days=5))
        s = aligned.dropna()
        if s.empty:
            return None
        return pd.DataFrame({'time': s.index, col_name: s.values})

    sma_20_df = _to_line_df(sma_20, 'SMA 20')
    sma_50_df = _to_line_df(sma_50, 'SMA 50')
    sma_200_df = _to_line_df(sma_200, 'SMA 200')

    bench_name = None
    bench_df = None
    if bench_series is not None:
        bench_name = (bench_series.name
                      if hasattr(bench_series, 'name') and bench_series.name
                      else "Benchmark")
        bench_df = _to_line_df(bench_series, bench_name)

    comp_name = None
    comp_df = None
    if comparison_series is not None:
        comp_name = (comparison_series.name
                     if hasattr(comparison_series, 'name') and comparison_series.name
                     else "Standard Rebalance")
        comp_df = _to_line_df(comparison_series, comp_name)

    # Margin metrics (convert ratio -> %, align to candle timestamps)
    usage_df = None
    usage_danger_df = None
    danger_timestamps = None  # DatetimeIndex of bars where usage >= 100%
    if usage_series is not None:
        usage_pct = (usage_series * 100).dropna()
        if not usage_pct.empty:
            aligned = usage_pct.reindex(candle_idx, method='nearest',
                                         tolerance=pd.Timedelta(days=5))
            aligned = aligned.dropna()
            if not aligned.empty:
                usage_df = pd.DataFrame({
                    'time': aligned.index,
                    'Margin Usage %': aligned.values,
                })
                # Red overlay for segments at or above 100%
                danger_mask = aligned >= 100
                if danger_mask.any():
                    # Expand mask to include one adjacent point at each
                    # boundary so the red line connects to the yellow
                    expanded = danger_mask.copy()
                    for i in range(1, len(danger_mask)):
                        if danger_mask.iloc[i] and not danger_mask.iloc[i - 1]:
                            expanded.iloc[i - 1] = True
                        elif not danger_mask.iloc[i] and danger_mask.iloc[i - 1]:
                            expanded.iloc[i] = True
                    danger_vals = aligned[expanded]
                    usage_danger_df = pd.DataFrame({
                        'time': danger_vals.index,
                        'Margin Danger %': danger_vals.values,
                    })

                    # Timestamps for vertical_span shading (list form avoids
                    # calculateTrendLine which needs candlestick data on the
                    # subchart — we only have line series there).
                    danger_timestamps = aligned.index[danger_mask]

    equity_pct_df = None
    if equity_pct_series is not None:
        eq_pct = (equity_pct_series * 100).dropna()
        if not eq_pct.empty:
            aligned = eq_pct.reindex(candle_idx, method='nearest',
                                      tolerance=pd.Timedelta(days=5))
            aligned = aligned.dropna()
            if not aligned.empty:
                equity_pct_df = pd.DataFrame({
                    'time': aligned.index,
                    'Equity %': aligned.values,
                })

    # Effective margin rate (already in %, e.g. 5.0 = 5%)
    eff_rate_df = None
    if effective_rate_series is not None:
        eff_rate_df = _to_line_df(effective_rate_series, 'Margin Rate %')

    # OHLC table data (preformatted for display)
    table_data = ohlc_df.copy()
    table_data['Date'] = table_data.index.strftime('%Y-%m-%d')
    table_data['Change %'] = (
        (table_data['Close'] - table_data['Open']) / table_data['Open'] * 100
    ).round(2)
    table_data['Range'] = table_data['High'] - table_data['Low']
    display_df = table_data[
        ['Date', 'Open', 'High', 'Low', 'Close', 'Change %', 'Range']
    ].copy()
    display_df = display_df.sort_index(ascending=False)
    for col in ['Open', 'High', 'Low', 'Close', 'Range']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")

    return {
        'candle_df': candle_df,
        'sma_20_df': sma_20_df,
        'sma_50_df': sma_50_df,
        'sma_200_df': sma_200_df,
        'bench_df': bench_df,
        'bench_name': bench_name,
        'comp_df': comp_df,
        'comp_name': comp_name,
        'usage_df': usage_df,
        'usage_danger_df': usage_danger_df,
        'equity_pct_df': equity_pct_df,
        'eff_rate_df': eff_rate_df,
        'danger_timestamps': danger_timestamps,
        'display_df': display_df,
    }


def render_candlestick_chart(ohlc_df, equity_series, loan_series,
                              usage_series, equity_pct_series, timeframe,
                              log_scale, show_range_slider=True,
                              show_volume=True, bench_series=None,
                              comparison_series=None,
                              effective_rate_series=None):
    title_map = {
        "1D": "Daily", "1W": "Weekly", "1M": "Monthly",
        "3M": "Quarterly", "1Y": "Yearly",
    }
    timeframe_label = title_map.get(timeframe, timeframe) or "Daily"

    # show_range_slider and show_volume are no-ops:
    # LWC has built-in scroll/zoom; no real volume data exists
    _ = show_range_slider, show_volume

    # ── Timeframe selector (TradingView-style topbar) ──────────────────
    tf_options = ["1D", "1W", "1M", "3M", "1Y"]
    st.pills(
        "Timeframe", tf_options,
        default=timeframe, key="candlestick_tf",
        label_visibility="collapsed",
    )

    # ── Margin series toggle pills ─────────────────────────────────────
    margin_series_options = ["Usage %", "Equity %", "Rate %", "Danger"]
    selected_margin = st.pills(
        "Margin Series", margin_series_options,
        default=["Usage %", "Equity %", "Danger"],
        selection_mode='multi',
        key="candlestick_margin_series",
        label_visibility="collapsed",
    )
    selected_margin = selected_margin or []

    data = _prepare_candlestick_data(
        ohlc_df, equity_series, loan_series,
        usage_series, equity_pct_series,
        bench_series, comparison_series,
        effective_rate_series,
    )

    # ── Chart with synced margin indicator pane ────────────────────────
    chart = StreamlitChart(
        height=900,
        inner_width=1,
        inner_height=0.60,  # type: ignore[arg-type]  # 60% main, 40% margin
        toolbox=True,
    )

    # Styling (TradingView dark theme)
    chart.layout(
        background_color='#131722',
        text_color='#d1d4dc',
        font_family='Trebuchet MS',
    )
    chart.candle_style(
        up_color='#26a69a',
        down_color='#ef5350',
        border_up_color='#26a69a',
        border_down_color='#ef5350',
        wick_up_color='#26a69a',
        wick_down_color='#ef5350',
    )
    chart.legend(visible=True, font_size=12)
    chart.crosshair(mode='normal')
    chart.watermark(timeframe_label, color='rgba(180, 180, 240, 0.3)')
    chart.price_scale(mode='logarithmic' if log_scale else 'normal')

    # Main OHLC data
    chart.set(data['candle_df'])

    # SMA overlays
    if data['sma_20_df'] is not None:
        sma20 = chart.create_line(
            'SMA 20', color='#2962FF', width=1, style='solid',
            price_line=False, price_label=False,
        )
        sma20.set(data['sma_20_df'])

    if data['sma_50_df'] is not None:
        sma50 = chart.create_line(
            'SMA 50', color='#FF6D00', width=1, style='solid',
            price_line=False, price_label=False,
        )
        sma50.set(data['sma_50_df'])

    if data['sma_200_df'] is not None:
        sma200 = chart.create_line(
            'SMA 200', color='#9C27B0', width=2, style='solid',
            price_line=False, price_label=False,
        )
        sma200.set(data['sma_200_df'])

    # Benchmark overlay (optional)
    if data['bench_df'] is not None:
        bench_line = chart.create_line(
            data['bench_name'], color='#FFD700', width=2, style='dashed',
            price_line=False, price_label=False,
        )
        bench_line.set(data['bench_df'])

    # Comparison overlay (optional)
    if data['comp_df'] is not None:
        comp_line = chart.create_line(
            data['comp_name'], color='#00FFFF', width=2, style='dashed',
            price_line=False, price_label=False,
        )
        comp_line.set(data['comp_df'])

    # ── Margin Risk Metrics (synced indicator pane, like RSI/MACD) ─────
    margin_pane = chart.create_subchart(
        width=1, height=0.40, sync=True,
    )
    margin_pane.layout(background_color='#131722', text_color='#d1d4dc')
    margin_pane.legend(visible=True, font_size=11)

    if "Usage %" in selected_margin and data['usage_df'] is not None:
        usage_line = margin_pane.create_line(
            'Margin Usage %', color='#FFD700', width=2, style='solid',
            price_line=False, price_label=False,
        )
        usage_line.set(data['usage_df'])

    if "Danger" in selected_margin and data['usage_danger_df'] is not None:
        danger_line = margin_pane.create_line(
            'Margin Danger %', color='#FF5252', width=2, style='solid',
            price_line=False, price_label=False,
        )
        danger_line.set(data['usage_danger_df'])

    if "Danger" in selected_margin and data['danger_timestamps'] is not None:
        margin_pane.vertical_span(
            data['danger_timestamps'],
            color='rgba(255, 82, 82, 0.15)',
        )

    if "Equity %" in selected_margin and data['equity_pct_df'] is not None:
        eq_pct_line = margin_pane.create_line(
            'Equity %', color='#1DB954', width=2, style='dashed',
            price_line=False, price_label=False,
        )
        eq_pct_line.set(data['equity_pct_df'])

    if "Rate %" in selected_margin and data['eff_rate_df'] is not None:
        rate_line = margin_pane.create_line(
            'Margin Rate %', color='#FF00FF', width=1, style='dotted',
            price_line=False, price_label=False,
            price_scale_id='right',
        )
        rate_line.set(data['eff_rate_df'])

    # 100% threshold line — show when Usage or Danger series are visible
    if "Usage %" in selected_margin or "Danger" in selected_margin:
        margin_pane.horizontal_line(
            100, color='#FF5252', width=1, style='dashed',
            axis_label_visible=False,
        )

    # Default visible range: zoom into recent data so candles are readable.
    # User can always scroll/zoom out to see the full history.
    candle_df = data['candle_df']
    n = len(candle_df)
    visible_bars = {'1D': 120, '1W': 52, '1M': 36, '3M': 20, '1Y': 10}
    bars = min(visible_bars.get(timeframe, 60), n)
    if bars < n:
        chart.set_visible_range(
            candle_df['time'].iloc[-bars],
            candle_df['time'].iloc[-1],
        )
    else:
        chart.fit()
    chart.load()

    # ── OHLC Table ─────────────────────────────────────────────────────
    with st.expander("\U0001f4ca OHLC Table View", expanded=False):
        st.dataframe(
            data['display_df'],
            use_container_width=True,
            hide_index=True,
            height=400,
        )

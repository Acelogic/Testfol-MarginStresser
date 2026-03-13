"""Withdrawals tab — summary metrics, cumulative chart, and event table."""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Regex patterns for log lines produced by shadow_backtest.py
_DRAW_RE = re.compile(
    r"Draw (\d{4}-\d{2}-\d{2}): \+\$([\d,]+(?:\.\d+)?)\s*→\s*loan \$([\d,]+(?:\.\d+)?)"
)
_REPAY_RE = re.compile(
    r"Repay (\d{4}-\d{2}-\d{2}): -\$([\d,]+(?:\.\d+)?)\s*→\s*loan \$([\d,]+(?:\.\d+)?)"
)


def _parse_amount(s: str) -> float:
    return float(s.replace(",", ""))


def _parse_events(logs: list) -> pd.DataFrame:
    """Parse draw/repay log lines into a structured DataFrame."""
    rows: list[dict] = []
    for line in logs or []:
        m = _DRAW_RE.search(line)
        if m:
            rows.append({
                "Date": pd.Timestamp(m.group(1)),
                "Type": "Draw",
                "Amount": _parse_amount(m.group(2)),
                "Loan After": _parse_amount(m.group(3)),
            })
            continue
        m = _REPAY_RE.search(line)
        if m:
            rows.append({
                "Date": pd.Timestamp(m.group(1)),
                "Type": "Repayment",
                "Amount": _parse_amount(m.group(2)),
                "Loan After": _parse_amount(m.group(3)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def render_withdrawals_tab(
    tab,
    logs: list,
    draw_monthly: float,
    draw_start_date=None,
    loan_series: pd.Series | None = None,
) -> None:
    with tab:
        # ── Config banner ────────────────────────────────────────────────
        if draw_monthly <= 0:
            st.info("No monthly draw configured.")
            return

        _ds_label = str(draw_start_date) if draw_start_date is not None else "Backtest start"
        st.markdown(
            f"**Monthly Draw:** ${draw_monthly:,.0f} &nbsp;|&nbsp; **Start Date:** {_ds_label}"
        )

        # ── Parse events ─────────────────────────────────────────────────
        events = _parse_events(logs)

        if events.empty:
            st.info("No withdrawal events recorded. Make sure the backtest covers the draw start date.")
            return

        draws = events[events["Type"] == "Draw"]
        repays = events[events["Type"] == "Repayment"]

        total_drawn = draws["Amount"].sum()
        total_repaid = repays["Amount"].sum()
        has_repayments = len(repays) > 0

        # Shadow backtest loan (draws + interest only)
        shadow_final_loan = events["Loan After"].iloc[-1]

        # True loan from render-time margin sim (includes taxes, full interest)
        has_true_loan = loan_series is not None and not loan_series.empty
        true_final_loan = loan_series.iloc[-1] if has_true_loan else shadow_final_loan
        loan_diverges = has_true_loan and abs(true_final_loan - shadow_final_loan) > 1

        # ── Summary metrics ──────────────────────────────────────────────
        if has_repayments:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Drawn", f"${total_drawn:,.0f}", f"{len(draws)} draws")
            c2.metric("Total Repaid", f"${total_repaid:,.0f}", f"{len(repays)} repayments")
            net_impact = total_drawn - total_repaid
            c3.metric("Net Impact", f"${net_impact:,.0f}")
        else:
            c1, _, c4 = st.columns([1, 1, 1])
            c1.metric("Total Drawn", f"${total_drawn:,.0f}", f"{len(draws)} monthly draws")

        if loan_diverges:
            extra = true_final_loan - shadow_final_loan
            c4.metric(
                "Loan Balance",
                f"${true_final_loan:,.0f}",
                f"+${extra:,.0f} taxes & interest",
                delta_color="inverse",
            )
        else:
            c4.metric("Loan Balance", f"${true_final_loan:,.0f}")

        # ── Loan balance chart ───────────────────────────────────────────
        chart_df = events.copy()
        chart_df["Signed"] = chart_df.apply(
            lambda r: r["Amount"] if r["Type"] == "Draw" else -r["Amount"], axis=1
        )
        chart_df["Cumulative Draws"] = chart_df["Signed"].cumsum()
        chart_df = chart_df.set_index("Date")

        fig = go.Figure()
        _hover = "%{x|%b %Y}<br>$%{y:,.0f}<extra>%{fullData.name}</extra>"

        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["Cumulative Draws"],
            name="Cumulative Draws", mode="lines",
            line=dict(dash="dash"), hovertemplate=_hover,
        ))
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["Loan After"],
            name="Loan (Draws + Interest)", mode="lines",
            hovertemplate=_hover,
        ))

        if has_true_loan:
            monthly_loan = loan_series.resample("MS").last()
            start = chart_df.index.min()
            end = chart_df.index.max()
            monthly_loan = monthly_loan[(monthly_loan.index >= start) & (monthly_loan.index <= end)]
            if not monthly_loan.empty:
                fig.add_trace(go.Scatter(
                    x=monthly_loan.index, y=monthly_loan.values,
                    name="Loan (Full Sim)", mode="lines",
                    hovertemplate=_hover,
                ))

        fig.update_layout(
            title="Loan Balance",
            yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        if loan_diverges:
            st.caption(
                "**Draws + Interest** = loan from backtest engine. "
                "**Full Sim** = includes tax payments charged to the loan."
            )

        # ── Event table ──────────────────────────────────────────────────
        with st.expander(f"Event Log ({len(events)} events)", expanded=False):
            display = events.copy()
            display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
            display["Amount"] = display["Amount"].map(lambda v: f"${v:,.0f}")
            display["Loan After"] = display["Loan After"].map(lambda v: f"${v:,.0f}")
            st.dataframe(display, hide_index=True, use_container_width=True)

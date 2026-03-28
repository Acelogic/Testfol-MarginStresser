"""
NDXMEGASPLIT (w/ ERs) Corrections Analysis
Generates a Charlie Bilello-style corrections table (>5% drawdowns) for the
NDXMEGASPLIT portfolio vs SPY, with recovery times and market event labels.

Outputs:
  - PNG chart:  data/results/ad_hoc/ndxmegasplit_corrections.png
  - HTML viewer: data/results/ad_hoc/ndxmegasplit_corrections.html
  - CSV data:   data/results/ad_hoc/ndxmegasplit_corrections.csv

Usage:
  python data/ndx_simulation/scripts/ad_hoc/corrections_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Path setup (same pattern as other scripts) ──────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
import config

# Project root (Testfol-MarginStresser/)
PROJECT_ROOT = os.path.abspath(os.path.join(config.BASE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from app.services.data_service import fetch_component_data
from app.core.shadow_backtest import run_shadow_backtest

# ── Configuration ────────────────────────────────────────────────────────
THRESHOLD = -0.05  # 5% drawdown minimum
START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")
START_VAL = 100_000

# NDXMEGASPLIT (w/ ERs) allocation — matches data/presets.json
ALLOCATION = {
    "NDXMEGASIM?L=2&E=0.95": 60.0,
    "GLDSIM?E=0.40":         20.0,
    "VXUSSIM?E=0.05":        15.0,
    "QQQSIM?L=3&E=0.88":     5.0,
}

# Output paths
AD_HOC_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "ad_hoc")
CSV_OUT = os.path.join(AD_HOC_RESULTS_DIR, "ndxmegasplit_corrections.csv")
PNG_OUT = os.path.join(AD_HOC_RESULTS_DIR, "ndxmegasplit_corrections.png")
HTML_OUT = os.path.join(AD_HOC_RESULTS_DIR, "ndxmegasplit_corrections.html")

# ── Market event descriptions ────────────────────────────────────────────
EVENT_MAP = {
    (2000, 1): "Y2K Fears, Dot-com Excess",
    (2000, 3): "Dot-com Bubble Burst, Tech Wreck",
    (2000, 7): "Dot-com Continued, Tech Wreck",
    (2001, 1): "Dot-com Unwind, Recession Begins",
    (2001, 9): "9/11 Attacks, Market Shutdown",
    (2002, 3): "WorldCom/Enron Scandals, Accounting Crisis",
    (2002, 7): "Corporate Fraud, Market Capitulation",
    (2003, 1): "Iraq War Buildup, Terror Fears",
    (2003, 3): "Iraq Invasion, Oil Spike, SARS",
    (2004, 3): "Madrid Bombings, Rate Hike Fears",
    (2004, 7): "Oil >$50, Rate Hikes Begin",
    (2005, 3): "GM/Ford Downgrades, Credit Fears",
    (2005, 10): "Katrina Aftermath, Oil Spike, Inflation",
    (2006, 5): "EM Selloff, Rate Hike Fears",
    (2006, 7): "Lebanon War, Oil $78, Housing Cools",
    (2007, 2): "Subprime Early Signs, Shanghai Crash",
    (2007, 7): "Subprime Contagion, Quant Blowups",
    (2007, 11): "GFC: Lehman/AIG/Bear Stearns Collapse",
    (2010, 11): "Ireland Bailout, EU Debt Contagion",
    (2011, 2): "Arab Spring, Japan Earthquake/Fukushima",
    (2011, 4): "EU Debt (Portugal), S&P Warning",
    (2011, 7): "US Downgrade (AAA->AA+), EU Crisis",
    (2012, 3): "EU Debt (Spain/Italy), Austerity",
    (2012, 9): "Fiscal Cliff Fears, Election",
    (2013, 5): "Taper Tantrum, Fed Signals QE Wind-Down",
    (2014, 1): "EM Currency Crisis (Turkey/Argentina)",
    (2014, 3): "Russia Annexes Crimea, Sanctions",
    (2014, 7): "Ukraine/MH17, Gaza, ISIS, Ebola",
    (2014, 9): "Ebola Fears, Oil Price Collapse, ISIS",
    (2014, 11): "Oil Crash, OPEC Refuses Cut, Ruble",
    (2015, 3): "Dollar Surge, Rate Hike Fears",
    (2015, 4): "China Slowdown, Dollar Drag",
    (2015, 7): "China Devaluation, Yuan Shock, EM Crash",
    (2015, 11): "Paris Attacks, Rate Hike, Oil <$40",
    (2015, 12): "China, Oil <$30, Recession Fears",
    (2016, 6): "Brexit Vote, EU Uncertainty",
    (2016, 10): "Election Uncertainty, Trump Shock",
    (2017, 6): "Tech/FANG Rotation, Valuation Fears",
    (2018, 1): "Volmageddon (XIV Blowup), Rate Fears",
    (2018, 2): "Inflation/Rate Fears, VIX Aftershock",
    (2018, 3): "Trade War (Tariffs), Facebook Scandal",
    (2018, 6): "Tariff Escalation, Turkey Lira Crisis",
    (2018, 7): "Trade War Intensifies, EM Stress",
    (2018, 8): "Trade War + Fed Hawkish + Housing",
    (2019, 4): "Trade War, Tariff Tweets, China",
    (2019, 7): "Yield Curve Inversion, Recession Signal",
    (2020, 2): "COVID Pandemic, Lockdowns, Depression Fears",
    (2020, 6): "COVID Second Wave, Reopening Doubts",
    (2020, 7): "COVID Resurgence, Tech Bubble Talk",
    (2020, 8): "Softbank Whale, Value Rotation",
    (2020, 9): "No Stimulus, COVID, Election Uncertainty",
    (2021, 1): "GameStop/Meme Frenzy, Rate Scare",
    (2021, 2): "Rate Spike (10Y>1.5%), Value Rotation",
    (2021, 4): "Inflation Spike (CPI 5%), Tax Plan",
    (2021, 7): "Delta Variant, China Tech Crackdown",
    (2021, 9): "Evergrande, Fed Taper, Supply Chain",
    (2021, 11): "Omicron, Fed Hawkish, Inflation 6.8%",
    (2023, 7): "Fitch Downgrade, 10Y>4.5%, Higher Longer",
    (2023, 12): "Rate Cuts Repriced, Strong Economy",
    (2024, 1): "Rate Cut Delay, Strong Jobs",
    (2024, 4): "Hot CPI, Rate Cuts Repriced, Iran/Israel",
    (2024, 7): "Yen Carry Unwind, Japan Hike, Nikkei Crash",
    (2024, 11): "Post-Election Tariff Fears, Strong $",
    (2024, 12): "DeepSeek AI Shock, Mag7 Selloff, Tariffs",
    (2025, 8): "Recession Fears, AI Capex Pullback",
    (2025, 10): "Tariff Impact, Earnings Downgrades",
    (2026, 1): "Iran War, Tariff Recession, AI Bubble Fears",
}


def get_event(peak_date):
    key = (peak_date.year, peak_date.month)
    if key in EVENT_MAP:
        return EVENT_MAP[key]
    for offset in [1, -1, 2, -2]:
        m = peak_date.month + offset
        y = peak_date.year
        if m > 12: m -= 12; y += 1
        if m < 1: m += 12; y -= 1
        if (y, m) in EVENT_MAP:
            return EVENT_MAP[(y, m)]
    return ""


def fmt_recov(days):
    if days >= 365:
        return f"{days / 365.25:.1f}yr"
    if days >= 60:
        return f"{days // 30}mo"
    return f"{days}d"


# ── Portfolio simulation (via shadow backtest engine) ────────────────────
def build_portfolio():
    """Run the real shadow backtest engine for NDXMEGASPLIT (w/ ERs)."""
    print("Fetching component prices...")
    tickers = list(ALLOCATION.keys())
    prices_df = fetch_component_data(tickers, START_DATE, END_DATE)

    print("Running shadow backtest engine...")
    _trades, _pl, _comp, _unreal, _logs, portfolio, _twr, _pm = run_shadow_backtest(
        allocation=ALLOCATION,
        start_val=float(START_VAL),
        start_date=START_DATE,
        end_date=END_DATE,
        prices_df=prices_df,
        rebalance_freq="Custom",
        custom_freq="Yearly",
        rebalance_month=1,
        rebalance_day=1,
        invest_dividends=True,
    )

    if portfolio.empty:
        raise RuntimeError("Shadow backtest returned empty portfolio series. Check logs.")

    # SPY benchmark
    spy_prices = fetch_component_data(["SPY"], START_DATE, END_DATE)
    spy = spy_prices["SPY"].reindex(portfolio.index).ffill()
    spy_norm = spy / spy.iloc[0] * START_VAL

    return portfolio, spy_norm


# ── Drawdown episode detection ───────────────────────────────────────────
def find_episodes(series, threshold=-0.05):
    dd = series / series.cummax() - 1.0
    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough_dd = 0.0
    for date, d in dd.items():
        if d == 0.0:
            if in_dd and trough_dd < threshold:
                episodes.append({
                    "peak_date": peak_date, "peak_val": series[peak_date],
                    "trough_date": trough_date, "trough_val": series[trough_date],
                    "dd": trough_dd, "recovery": date,
                })
            in_dd = False
            peak_date = date
            trough_dd = 0.0
        else:
            in_dd = True
            if d < trough_dd:
                trough_dd = d
                trough_date = date
    if in_dd and trough_dd < threshold:
        episodes.append({
            "peak_date": peak_date, "peak_val": series[peak_date],
            "trough_date": trough_date, "trough_val": series[trough_date],
            "dd": trough_dd, "recovery": None,
        })
    return episodes


# ── Build table rows ─────────────────────────────────────────────────────
def build_rows(portfolio, spy_norm):
    dd_spy = spy_norm / spy_norm.cummax() - 1.0
    episodes = find_episodes(portfolio, THRESHOLD)
    rows = []

    for ep in episodes:
        peak, trough, recovery = ep["peak_date"], ep["trough_date"], ep["recovery"]
        pdd = ep["dd"] * 100
        n_days = (trough - peak).days

        end = recovery if recovery else dd_spy.index[-1]
        sw = spy_norm.loc[peak:end]
        sddw = sw / sw.cummax() - 1.0
        sdd = sddw.min() * 100
        spy_trough = sddw.idxmin()
        ratio = abs(pdd / sdd) if sdd != 0 else 0

        # SPLIT recovery
        last_date = portfolio.index[-1]
        if recovery:
            split_recov = fmt_recov((recovery - trough).days)
            split_total = fmt_recov((recovery - peak).days)
        else:
            split_recov = f"ongoing ({fmt_recov((last_date - trough).days)})"
            split_total = f"ongoing ({fmt_recov((last_date - peak).days)})"

        # SPY recovery
        spy_peak_val = spy_norm.loc[peak]
        spy_after = spy_norm.loc[spy_trough:]
        spy_recovered = spy_after[spy_after >= spy_peak_val]
        if len(spy_recovered) > 0:
            spy_rd = spy_recovered.index[0]
            spy_recov = fmt_recov((spy_rd - spy_trough).days)
            spy_total = fmt_recov((spy_rd - peak).days)
        else:
            spy_recov = f"ongoing ({fmt_recov((last_date - spy_trough).days)})"
            spy_total = f"ongoing ({fmt_recov((last_date - peak).days)})"

        period = f"{peak.strftime('%b %d, %Y')} - {trough.strftime('%b %d, %Y')}"
        if not recovery:
            period += "*"

        rows.append({
            "period": period,
            "peak_date": peak.strftime("%Y-%m-%d"),
            "trough_date": trough.strftime("%Y-%m-%d"),
            "days": n_days,
            "high": ep["peak_val"],
            "low": ep["trough_val"],
            "decline": pdd,
            "split_recov": split_recov,
            "split_total": split_total,
            "spy_dd": sdd,
            "spy_recov": spy_recov,
            "spy_total": spy_total,
            "ratio": ratio,
            "event": get_event(peak),
            "ongoing": recovery is None,
            "severe": abs(pdd) >= 25,
        })

    return rows, episodes


# ── CSV export ───────────────────────────────────────────────────────────
def export_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"  CSV: {CSV_OUT}")


# ── PNG chart (matplotlib) ───────────────────────────────────────────────
def export_png(rows, portfolio, episodes):
    n = len(rows)
    row_h = 0.62
    header_h = 0.75
    title_h = 2.2
    median_h = 0.7
    footer_h = 0.8
    fig_h = title_h + header_h + n * row_h + median_h + footer_h + 0.5
    fig_w = 36

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Theme
    bg = "#111827"; hdr_bg = "#1e293b"; row_a = "#111827"; row_b = "#162032"
    red_bg = "#2d0a0a"; gold_bg = "#1a1800"; text_w = "#e2e8f0"; text_g = "#94a3b8"
    gold = "#fbbf24"; red = "#ef4444"; orange = "#f97316"; yellow = "#eab308"; green = "#34d399"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    col_x = [0.003, 0.145, 0.185, 0.245, 0.305, 0.355, 0.405, 0.45, 0.495, 0.54, 0.585, 0.62]
    col_align = ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left"]
    headers = ["Correction Period", "Days", "SPLIT High", "SPLIT Low", "% Decline", "Recovery\nfrom Bottom", "Decline +\nRecovery Time", "SPY DD", "Recovery\nfrom Bottom", "Decline +\nRecovery Time", "Ratio", '"Markets Fall On..."']

    # Title
    ty = fig_h - 0.6
    ax.text(0.5, ty, "NDXMEGASPLIT (w/ ERs) Corrections >5% since Jul 2000", fontsize=28, fontweight="bold", color=text_w, ha="center", va="center", fontfamily="sans-serif")
    last_date = portfolio.index[-1].strftime("%-m/%-d/%Y")
    ax.text(0.5, ty - 0.7, f"*as of {last_date}  |  60% NDXMEGA 2x + 20% GLD + 15% VXUS + 5% QQQ 3x  |  Yearly Rebal  |  Data: Testfol SIM", fontsize=14, color=text_g, ha="center", va="center", fontfamily="sans-serif")

    # Group headers
    hy_group = fig_h - title_h + 0.35
    ax.text(0.355, hy_group, "--- SPLIT ---", fontsize=11, color="#64748b", ha="center", va="center", fontfamily="sans-serif")
    ax.text(0.495, hy_group, "--- SPY ---", fontsize=11, color="#64748b", ha="center", va="center", fontfamily="sans-serif")

    # Header row
    hy = fig_h - title_h
    ax.add_patch(plt.Rectangle((0, hy - header_h / 2), 1, header_h, fc=hdr_bg, ec="#334155", lw=0.5))
    for j, h in enumerate(headers):
        ax.text(col_x[j], hy, h, fontsize=13, fontweight="bold", color=gold, ha=col_align[j], va="center", fontfamily="sans-serif")

    # Data rows
    for i, r in enumerate(rows):
        ry = hy - header_h / 2 - row_h / 2 - i * row_h
        if r["ongoing"]: rbg = gold_bg
        elif r["severe"]: rbg = red_bg
        elif i % 2 == 0: rbg = row_a
        else: rbg = row_b
        ax.add_patch(plt.Rectangle((0, ry - row_h / 2), 1, row_h, fc=rbg, ec="#1e293b", lw=0.3))

        dd = r["decline"]
        if abs(dd) >= 25: dc, dw = red, "bold"
        elif abs(dd) >= 15: dc, dw = orange, "bold"
        elif abs(dd) >= 10: dc, dw = yellow, "normal"
        else: dc, dw = text_g, "normal"

        pw = "bold" if r["severe"] or r["ongoing"] else "normal"
        pc = text_w if r["severe"] or r["ongoing"] else text_g
        rc = red if "ongoing" in r["split_recov"] else text_w
        rw = "bold" if "ongoing" in r["split_recov"] else "normal"
        src = red if r["spy_recov"] == "ongoing" else text_g
        srw = "bold" if r["spy_recov"] == "ongoing" else "normal"
        ratio_c = green if r["ratio"] < 1.5 else (text_w if r["ratio"] < 3 else orange)
        ec_ = text_g if not (r["severe"] or r["ongoing"]) else "#cbd5e1"

        fs = 12
        vals = [
            (r["period"], pc, pw, fs), (str(r["days"]), text_w, "normal", fs),
            (f"${r['high']:,.0f}", text_w, "normal", fs), (f"${r['low']:,.0f}", text_w, "normal", fs),
            (f"{r['decline']:.1f}%", dc, dw, fs + 1), (r["split_recov"], rc, rw, fs),
            (r["split_total"], rc, rw, fs), (f"{r['spy_dd']:.1f}%", text_g, "normal", fs),
            (r["spy_recov"], src, srw, fs), (r["spy_total"], src, srw, fs),
            (f"{r['ratio']:.1f}x", ratio_c, "normal", fs), (r["event"], ec_, "normal", 11),
        ]
        for j, (v, c, w, s) in enumerate(vals):
            ax.text(col_x[j], ry, v, fontsize=s, color=c, fontweight=w, ha=col_align[j], va="center", fontfamily="sans-serif")

    # Median row
    my = hy - header_h / 2 - row_h / 2 - n * row_h - 0.15
    ax.add_patch(plt.Rectangle((0, my - median_h / 2), 1, median_h, fc="#0c2744", ec="#3b82f6", lw=2))
    med_days = int(np.median([r["days"] for r in rows]))
    med_decline = np.median([r["decline"] for r in rows])
    recov_days_list = [(ep["recovery"] - ep["trough_date"]).days for ep in episodes if ep["recovery"]]
    med_recov = fmt_recov(int(np.median(recov_days_list)))
    med_ratio = np.median([r["ratio"] for r in rows])
    med_spy_dd = np.median([r["spy_dd"] for r in rows])

    mfs = 14
    ax.text(col_x[0], my, "Median", fontsize=mfs, fontweight="bold", color=gold, ha="left", va="center", fontfamily="sans-serif")
    ax.text(col_x[1], my, str(med_days), fontsize=mfs, fontweight="bold", color=gold, ha="right", va="center", fontfamily="sans-serif")
    ax.text(col_x[4], my, f"{med_decline:.1f}%", fontsize=mfs, fontweight="bold", color=gold, ha="right", va="center", fontfamily="sans-serif")
    ax.text(col_x[5], my, med_recov, fontsize=mfs, fontweight="bold", color=gold, ha="right", va="center", fontfamily="sans-serif")
    ax.text(col_x[7], my, f"{med_spy_dd:.1f}%", fontsize=mfs, fontweight="bold", color=gold, ha="right", va="center", fontfamily="sans-serif")
    ax.text(col_x[10], my, f"{med_ratio:.1f}x", fontsize=mfs, fontweight="bold", color=gold, ha="right", va="center", fontfamily="sans-serif")

    # Footer
    fy = my - median_h / 2 - 0.4
    severe_n = sum(1 for r in rows if abs(r["decline"]) >= 25)
    mod_n = sum(1 for r in rows if 15 <= abs(r["decline"]) < 25)
    mild_n = sum(1 for r in rows if 10 <= abs(r["decline"]) < 15)
    minor_n = sum(1 for r in rows if 5 <= abs(r["decline"]) < 10)
    med_recov_mo = np.median(recov_days_list) / 30.44 if recov_days_list else 0
    footer = f"{len(rows)} corrections:  {severe_n} severe (>25%)  |  {mod_n} moderate (15-25%)  |  {mild_n} mild (10-15%)  |  {minor_n} minor (5-10%)      Median SPLIT recovery: {med_recov_mo:.1f}mo"
    ax.text(0.5, fy, footer, fontsize=13, color=text_g, ha="center", va="center", fontfamily="sans-serif")

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    os.makedirs(AD_HOC_RESULTS_DIR, exist_ok=True)
    plt.savefig(PNG_OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  PNG: {PNG_OUT}")


# ── Interactive HTML viewer ──────────────────────────────────────────────
def export_html(rows, portfolio, episodes):
    last_date = portfolio.index[-1].strftime("%-m/%-d/%Y")

    # Build JSON data for the table
    json_rows = []
    for r in rows:
        json_rows.append({
            "period": r["period"],
            "days": r["days"],
            "high": r["high"],
            "low": r["low"],
            "decline": round(r["decline"], 1),
            "split_recov": r["split_recov"],
            "split_total": r["split_total"],
            "spy_dd": round(r["spy_dd"], 1),
            "spy_recov": r["spy_recov"],
            "spy_total": r["spy_total"],
            "ratio": round(r["ratio"], 1),
            "event": r["event"],
            "ongoing": r["ongoing"],
            "severe": r["severe"],
        })

    rows_json = json.dumps(json_rows)
    n_severe = sum(1 for r in rows if abs(r["decline"]) >= 25)
    n_moderate = sum(1 for r in rows if 15 <= abs(r["decline"]) < 25)
    n_mild = sum(1 for r in rows if 10 <= abs(r["decline"]) < 15)
    n_minor = sum(1 for r in rows if 5 <= abs(r["decline"]) < 10)

    html = _build_html_template(rows_json, last_date, len(rows), n_severe, n_moderate, n_mild, n_minor)

    with open(HTML_OUT, "w") as f:
        f.write(html)
    print(f"  HTML: {HTML_OUT}")


def _build_html_template(rows_json, last_date, total, n_severe, n_moderate, n_mild, n_minor):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NDXMEGASPLIT Corrections Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #111827; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }}
h1 {{ text-align: center; font-size: 28px; margin-bottom: 8px; }}
.subtitle {{ text-align: center; color: #94a3b8; font-size: 13px; margin-bottom: 20px; }}
.controls {{ text-align: center; margin-bottom: 16px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }}
.controls button {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; transition: all 0.2s; }}
.controls button:hover {{ background: #334155; }}
.controls button.active {{ background: #3b82f6; border-color: #3b82f6; color: white; }}
.stats {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }}
.stat-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 120px; }}
.stat-card .label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase; }}
.stat-card .value {{ color: #fbbf24; font-size: 22px; font-weight: bold; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
thead {{ position: sticky; top: 0; z-index: 10; }}
th {{ background: #1e293b; color: #fbbf24; padding: 10px 8px; text-align: right; font-size: 12px; border-bottom: 2px solid #334155; white-space: nowrap; }}
th:first-child, th:last-child {{ text-align: left; }}
td {{ padding: 8px 8px; border-bottom: 1px solid #1e293b; text-align: right; white-space: nowrap; }}
td:first-child {{ text-align: left; }}
td:last-child {{ text-align: left; white-space: normal; max-width: 350px; }}
tr {{ transition: background 0.15s; }}
tr:hover {{ background: #1e3a5f !important; }}
tr.even {{ background: #111827; }}
tr.odd {{ background: #162032; }}
tr.severe {{ background: #2d0a0a; }}
tr.ongoing {{ background: #1a1800; }}
tr.severe td:first-child, tr.ongoing td:first-child {{ font-weight: bold; color: #e2e8f0; }}
.decline-severe {{ color: #ef4444; font-weight: bold; }}
.decline-moderate {{ color: #f97316; font-weight: bold; }}
.decline-mild {{ color: #eab308; }}
.decline-minor {{ color: #94a3b8; }}
.ongoing-text {{ color: #ef4444; font-weight: bold; }}
.ratio-low {{ color: #34d399; }}
.ratio-high {{ color: #f97316; }}
.median-row {{ background: #0c2744 !important; border: 2px solid #3b82f6; }}
.median-row td {{ color: #fbbf24; font-weight: bold; font-size: 14px; }}
.footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 16px; }}
.group-header {{ text-align: center; color: #64748b; font-size: 11px; }}
.search {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 8px 14px; border-radius: 6px; font-size: 13px; width: 250px; }}
.search::placeholder {{ color: #64748b; }}
</style>
</head>
<body>
<h1>NDXMEGASPLIT (w/ ERs) Corrections &gt;5% since Jul 2000</h1>
<p class="subtitle">*as of {last_date} | 60% NDXMEGA 2x + 20% GLD + 15% VXUS + 5% QQQ 3x | Yearly Rebal | Data: Testfol SIM</p>

<div class="controls">
  <button class="active" onclick="filterRows('all')">All ({total})</button>
  <button onclick="filterRows('severe')">Severe &gt;25% ({n_severe})</button>
  <button onclick="filterRows('moderate')">Moderate 15-25% ({n_moderate})</button>
  <button onclick="filterRows('mild')">Mild 10-15% ({n_mild})</button>
  <button onclick="filterRows('minor')">Minor 5-10% ({n_minor})</button>
  <input type="text" class="search" placeholder="Search events..." oninput="searchRows(this.value)">
</div>

<div class="stats" id="stats"></div>

<table>
<thead>
<tr>
  <th style="text-align:left">Correction Period</th>
  <th>Days</th>
  <th>SPLIT High</th>
  <th>SPLIT Low</th>
  <th>% Decline</th>
  <th><span class="group-header">SPLIT</span><br>Recovery<br>from Bottom</th>
  <th>Decline +<br>Recovery Time</th>
  <th><span class="group-header">SPY</span><br>DD</th>
  <th>Recovery<br>from Bottom</th>
  <th>Decline +<br>Recovery Time</th>
  <th>Ratio</th>
  <th style="text-align:left">"Markets Fall On..."</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
<div class="footer" id="footer"></div>

<script>
const DATA = {rows_json};

function declineClass(d) {{
  var a = Math.abs(d);
  if (a >= 25) return 'decline-severe';
  if (a >= 15) return 'decline-moderate';
  if (a >= 10) return 'decline-mild';
  return 'decline-minor';
}}

function ratioClass(r) {{
  if (r < 1.5) return 'ratio-low';
  if (r >= 3) return 'ratio-high';
  return '';
}}

function escapeHtml(text) {{
  var div = document.createElement('div');
  div.textContent = text;
  return div.textContent;
}}

function renderRows(filtered) {{
  var tbody = document.getElementById('tbody');
  // Clear existing rows
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

  filtered.forEach(function(r, i) {{
    var tr = document.createElement('tr');
    var cls = r.ongoing ? 'ongoing' : (r.severe ? 'severe' : (i % 2 === 0 ? 'even' : 'odd'));
    tr.className = cls;

    var cells = [
      {{ text: r.period, align: 'left' }},
      {{ text: String(r.days) }},
      {{ text: '$' + r.high.toLocaleString('en-US', {{maximumFractionDigits:0}}) }},
      {{ text: '$' + r.low.toLocaleString('en-US', {{maximumFractionDigits:0}}) }},
      {{ text: r.decline.toFixed(1) + '%', cls: declineClass(r.decline) }},
      {{ text: r.split_recov, cls: r.split_recov.includes('ongoing') ? 'ongoing-text' : '' }},
      {{ text: r.split_total, cls: r.split_total.includes('ongoing') ? 'ongoing-text' : '' }},
      {{ text: r.spy_dd.toFixed(1) + '%', style: 'color:#94a3b8' }},
      {{ text: r.spy_recov, cls: r.spy_recov.includes('ongoing') ? 'ongoing-text' : '', style: 'color:#94a3b8' }},
      {{ text: r.spy_total, cls: r.spy_total.includes('ongoing') ? 'ongoing-text' : '', style: 'color:#94a3b8' }},
      {{ text: r.ratio.toFixed(1) + 'x', cls: ratioClass(r.ratio) }},
      {{ text: r.event, align: 'left' }}
    ];

    cells.forEach(function(c) {{
      var td = document.createElement('td');
      td.textContent = c.text;
      if (c.cls) td.className = c.cls;
      if (c.style) td.setAttribute('style', c.style);
      if (c.align) td.style.textAlign = c.align;
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});

  // Median row
  if (filtered.length > 0) {{
    var medDays = median(filtered.map(function(r) {{ return r.days; }}));
    var medDecline = median(filtered.map(function(r) {{ return r.decline; }}));
    var medSpy = median(filtered.map(function(r) {{ return r.spy_dd; }}));
    var medRatio = median(filtered.map(function(r) {{ return r.ratio; }}));
    var mtr = document.createElement('tr');
    mtr.className = 'median-row';
    var medCells = ['Median', Math.round(medDays), '', '', medDecline.toFixed(1)+'%', '', '', medSpy.toFixed(1)+'%', '', '', medRatio.toFixed(1)+'x', ''];
    medCells.forEach(function(text, idx) {{
      var td = document.createElement('td');
      td.textContent = text;
      if (idx === 0 || idx === 11) td.style.textAlign = 'left';
      mtr.appendChild(td);
    }});
    tbody.appendChild(mtr);
  }}

  // Stats cards
  var statsEl = document.getElementById('stats');
  while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);
  var severe = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 25; }}).length;
  var mod = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 15 && Math.abs(r.decline) < 25; }}).length;
  var ong = filtered.filter(function(r) {{ return r.ongoing; }}).length;
  var medD = median(filtered.map(function(r) {{ return r.decline; }}));

  var statData = [
    ['Corrections', filtered.length],
    ['Median Decline', medD.toFixed(1) + '%'],
    ['Severe (>25%)', severe],
    ['Moderate (15-25%)', mod],
    ['Ongoing', ong]
  ];
  statData.forEach(function(s) {{
    var card = document.createElement('div');
    card.className = 'stat-card';
    var lbl = document.createElement('div');
    lbl.className = 'label';
    lbl.textContent = s[0];
    var val = document.createElement('div');
    val.className = 'value';
    val.textContent = s[1];
    card.appendChild(lbl);
    card.appendChild(val);
    statsEl.appendChild(card);
  }});
}}

function median(arr) {{
  if (!arr.length) return 0;
  var s = arr.slice().sort(function(a, b) {{ return a - b; }});
  var mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}}

var currentFilter = 'all';
var currentSearch = '';

function applyFilters() {{
  var filtered = DATA;
  if (currentFilter === 'severe') filtered = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 25; }});
  else if (currentFilter === 'moderate') filtered = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 15 && Math.abs(r.decline) < 25; }});
  else if (currentFilter === 'mild') filtered = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 10 && Math.abs(r.decline) < 15; }});
  else if (currentFilter === 'minor') filtered = filtered.filter(function(r) {{ return Math.abs(r.decline) >= 5 && Math.abs(r.decline) < 10; }});
  if (currentSearch) {{
    var q = currentSearch.toLowerCase();
    filtered = filtered.filter(function(r) {{ return r.event.toLowerCase().indexOf(q) !== -1 || r.period.toLowerCase().indexOf(q) !== -1; }});
  }}
  renderRows(filtered);
}}

function filterRows(type) {{
  currentFilter = type;
  var buttons = document.querySelectorAll('.controls button');
  buttons.forEach(function(b) {{ b.classList.remove('active'); }});
  event.target.classList.add('active');
  applyFilters();
}}

function searchRows(q) {{
  currentSearch = q;
  applyFilters();
}}

// Initial render
renderRows(DATA);
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("NDXMEGASPLIT (w/ ERs) Corrections Analysis")
    print("=" * 70)

    portfolio, spy_norm = build_portfolio()

    dd = portfolio / portfolio.cummax() - 1.0
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / years) - 1

    print(f"\nPortfolio: {portfolio.index[0].strftime('%Y-%m-%d')} to {portfolio.index[-1].strftime('%Y-%m-%d')}")
    print(f"  ${portfolio.iloc[0]:,.0f} -> ${portfolio.iloc[-1]:,.0f} (CAGR: {cagr * 100:.1f}%)")
    print(f"  Max DD: {dd.min() * 100:.1f}% | Current DD: {dd.iloc[-1] * 100:.1f}%")

    rows, episodes = build_rows(portfolio, spy_norm)
    print(f"\nFound {len(rows)} corrections > {abs(THRESHOLD) * 100:.0f}%")

    print("\nExporting:")
    export_csv(rows)
    export_png(rows, portfolio, episodes)
    export_html(rows, portfolio, episodes)

    print("\nDone.")


if __name__ == "__main__":
    main()

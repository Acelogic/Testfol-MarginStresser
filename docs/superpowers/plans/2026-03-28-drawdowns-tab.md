# Drawdowns Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Drawdowns" tab to the Returns Analysis section showing all portfolio corrections >5% with SPY comparison, severity filtering, and market event labels.

**Architecture:** Calculation logic (`find_drawdown_episodes`, `build_drawdown_table`, `EVENT_MAP`) goes in `app/core/calculations/stats.py` alongside existing drawdown helpers. Rendering goes in `app/ui/charts/returns.py` as the 6th sub-tab. Uses SPYSIM for maximum date coverage.

**Tech Stack:** pandas, numpy, Streamlit (`st.tabs`, `st.radio`, `st.columns`, `st.metric`, `st.dataframe`), existing `fetch_component_data` for SPYSIM data.

**Spec:** `docs/superpowers/specs/2026-03-28-drawdowns-tab-design.md`

---

### Task 1: Add `find_drawdown_episodes()` to stats.py

**Files:**
- Modify: `app/core/calculations/stats.py` (append after line 292)
- Test: `tests/test_calculations.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_calculations.py`:

```python
from app.core.calculations import find_drawdown_episodes

# ---------------------------------------------------------------------------
# find_drawdown_episodes
# ---------------------------------------------------------------------------

def test_find_drawdown_episodes_basic():
    """Detects a single drawdown episode with recovery."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    # Rise to 120, drop to 80 (-33%), recover to 120
    values = np.concatenate([
        np.linspace(100, 120, 15),   # rise
        np.linspace(120, 80, 15),    # -33% drawdown
        np.linspace(80, 120, 15),    # recovery
        np.linspace(120, 130, 15),   # new high
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["peak_val"] == pytest.approx(120.0, abs=1.0)
    assert ep["trough_val"] == pytest.approx(80.0, abs=1.0)
    assert ep["dd"] == pytest.approx(-1/3, abs=0.05)
    assert ep["recovery"] is not None


def test_find_drawdown_episodes_ongoing():
    """Detects an ongoing drawdown (no recovery)."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    values = np.concatenate([
        np.linspace(100, 120, 15),
        np.linspace(120, 90, 15),   # -25% drop, no recovery
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 1
    assert episodes[0]["recovery"] is None


def test_find_drawdown_episodes_below_threshold():
    """Small drawdowns below threshold are ignored."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    values = np.concatenate([
        np.linspace(100, 105, 15),
        np.linspace(105, 102, 15),  # -2.8% drop, below 5% threshold
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 0


def test_find_drawdown_episodes_empty():
    """Empty series returns empty list."""
    series = pd.Series(dtype=float)
    assert find_drawdown_episodes(series) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calculations.py -k "drawdown_episodes" -v`
Expected: FAIL with `ImportError: cannot import name 'find_drawdown_episodes'`

- [ ] **Step 3: Write the implementation**

Add to the end of `app/core/calculations/stats.py`:

```python
def find_drawdown_episodes(series: pd.Series, threshold: float = -0.05) -> list[dict]:
    """Find all drawdown episodes exceeding threshold.

    Returns list of dicts with keys: peak_date, peak_val, trough_date,
    trough_val, dd (decimal, e.g. -0.25), recovery (date or None).
    """
    if series.empty:
        return []
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
```

- [ ] **Step 4: Add export to `__init__.py`**

In `app/core/calculations/__init__.py`, add `find_drawdown_episodes` to the imports from `stats` and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_calculations.py -k "drawdown_episodes" -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/calculations/stats.py app/core/calculations/__init__.py tests/test_calculations.py
git commit -m "feat: add find_drawdown_episodes() to calculations"
```

---

### Task 2: Add `fmt_duration()`, `EVENT_MAP`, and `get_market_event()` to stats.py

**Files:**
- Modify: `app/core/calculations/stats.py` (append after `find_drawdown_episodes`)
- Modify: `app/core/calculations/__init__.py` (add exports)
- Test: `tests/test_calculations.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_calculations.py`:

```python
from app.core.calculations import fmt_duration, get_market_event

# ---------------------------------------------------------------------------
# fmt_duration
# ---------------------------------------------------------------------------

def test_fmt_duration_years():
    assert fmt_duration(400) == "1.1yr"
    assert fmt_duration(730) == "2.0yr"


def test_fmt_duration_months():
    assert fmt_duration(90) == "3mo"
    assert fmt_duration(60) == "2mo"


def test_fmt_duration_days():
    assert fmt_duration(59) == "59d"
    assert fmt_duration(1) == "1d"


# ---------------------------------------------------------------------------
# get_market_event
# ---------------------------------------------------------------------------

def test_get_market_event_exact_match():
    """Exact (year, month) match returns event string."""
    date = pd.Timestamp("2020-02-19")
    event = get_market_event(date)
    assert "COVID" in event


def test_get_market_event_fuzzy_match():
    """Fuzzy match within +/- 2 months returns event string."""
    date = pd.Timestamp("2020-04-01")  # 2 months after Feb 2020
    event = get_market_event(date)
    assert "COVID" in event


def test_get_market_event_no_match():
    """No match returns empty string."""
    date = pd.Timestamp("1990-01-01")
    event = get_market_event(date)
    assert event == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calculations.py -k "fmt_duration or market_event" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Add to the end of `app/core/calculations/stats.py`:

```python
def fmt_duration(days: int) -> str:
    """Format a day count as human-readable duration (e.g. '5.0yr', '3mo', '59d')."""
    if days >= 365:
        return f"{days / 365.25:.1f}yr"
    if days >= 60:
        return f"{days // 30}mo"
    return f"{days}d"


MARKET_EVENT_MAP: dict[tuple[int, int], str] = {
    (2000, 7): "Dot-com Bubble Burst, Tech Wreck",
    (2007, 7): "Subprime Contagion, Quant Blowups",
    (2007, 11): "GFC: Lehman/AIG/Bear Stearns Collapse",
    (2010, 11): "Ireland Bailout, EU Debt Contagion",
    (2011, 2): "Arab Spring, Japan Earthquake/Fukushima",
    (2011, 4): "EU Debt (Portugal), S&P Warning",
    (2011, 7): "US Downgrade (AAA->AA+), EU Crisis",
    (2012, 3): "EU Debt (Spain/Italy), Austerity",
    (2012, 9): "Fiscal Cliff Fears, Election",
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


def get_market_event(peak_date) -> str:
    """Look up a market event by peak date, with +/- 2 month fuzzy matching."""
    key = (peak_date.year, peak_date.month)
    if key in MARKET_EVENT_MAP:
        return MARKET_EVENT_MAP[key]
    for offset in [1, -1, 2, -2]:
        m = peak_date.month + offset
        y = peak_date.year
        if m > 12: m -= 12; y += 1
        if m < 1: m += 12; y -= 1
        if (y, m) in MARKET_EVENT_MAP:
            return MARKET_EVENT_MAP[(y, m)]
    return ""
```

- [ ] **Step 4: Add exports to `__init__.py`**

In `app/core/calculations/__init__.py`, add `fmt_duration`, `get_market_event`, and `MARKET_EVENT_MAP` to the imports from `stats` and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_calculations.py -k "fmt_duration or market_event" -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/calculations/stats.py app/core/calculations/__init__.py tests/test_calculations.py
git commit -m "feat: add fmt_duration, EVENT_MAP, and get_market_event"
```

---

### Task 3: Add `build_drawdown_table()` to stats.py

**Files:**
- Modify: `app/core/calculations/stats.py` (append after `get_market_event`)
- Modify: `app/core/calculations/__init__.py` (add export)
- Test: `tests/test_calculations.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calculations.py`:

```python
from app.core.calculations import build_drawdown_table

# ---------------------------------------------------------------------------
# build_drawdown_table
# ---------------------------------------------------------------------------

def test_build_drawdown_table_basic():
    """Produces a DataFrame with expected columns from a series with one drawdown."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    port = pd.Series(np.concatenate([
        np.linspace(100, 120, 15),
        np.linspace(120, 80, 15),
        np.linspace(80, 120, 15),
        np.linspace(120, 130, 15),
    ]), index=dates)
    # Use same series as "SPY" for simplicity
    spy = port.copy()
    df = build_drawdown_table(port, spy)
    assert len(df) == 1
    assert "Correction Period" in df.columns
    assert "% Decline" in df.columns
    assert "Recovery from Bottom" in df.columns
    assert "Decline + Recovery Time" in df.columns
    assert "SPY DD" in df.columns
    assert "Ratio" in df.columns
    assert "Market Event" in df.columns
    assert "_severity" in df.columns
    assert "_ongoing" in df.columns
    # Decline should be around -33%
    assert df.iloc[0]["_decline_raw"] == pytest.approx(-33.3, abs=5.0)


def test_build_drawdown_table_empty():
    """No drawdowns returns empty DataFrame with correct columns."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    port = pd.Series(np.linspace(100, 130, 30), index=dates)
    spy = port.copy()
    df = build_drawdown_table(port, spy)
    assert len(df) == 0
    assert "Correction Period" in df.columns


def test_build_drawdown_table_ongoing():
    """Ongoing drawdown shows 'ongoing' in recovery columns."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    port = pd.Series(np.concatenate([
        np.linspace(100, 120, 15),
        np.linspace(120, 90, 15),
    ]), index=dates)
    spy = port.copy()
    df = build_drawdown_table(port, spy)
    assert len(df) == 1
    assert df.iloc[0]["_ongoing"] is True
    assert "ongoing" in df.iloc[0]["Recovery from Bottom"]
    assert "ongoing" in df.iloc[0]["Decline + Recovery Time"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calculations.py -k "build_drawdown_table" -v`
Expected: FAIL with `ImportError: cannot import name 'build_drawdown_table'`

- [ ] **Step 3: Write the implementation**

Add to the end of `app/core/calculations/stats.py`:

```python
def _classify_severity(decline_pct: float) -> str:
    """Classify drawdown severity by percentage (already negative)."""
    d = abs(decline_pct)
    if d >= 25: return "Severe"
    if d >= 15: return "Moderate"
    if d >= 10: return "Mild"
    return "Minor"


def build_drawdown_table(port_series: pd.Series, spy_series: pd.Series) -> pd.DataFrame:
    """Build a DataFrame of all drawdown episodes with SPY comparison.

    Args:
        port_series: Portfolio value series (DatetimeIndex).
        spy_series: SPY/SPYSIM value series, aligned to same dates as port_series.

    Returns:
        DataFrame with display columns and hidden _metadata columns for filtering.
    """
    columns = [
        "Correction Period", "Days", "% Decline",
        "Recovery from Bottom", "Decline + Recovery Time",
        "SPY DD", "SPY Recovery from Bottom", "SPY Decline + Recovery Time",
        "Ratio", "Market Event",
        "_ongoing", "_severity", "_decline_raw", "_spy_dd_raw", "_ratio_raw", "_days_raw",
    ]

    episodes = find_drawdown_episodes(port_series, threshold=-0.05)
    if not episodes:
        return pd.DataFrame(columns=columns)

    spy_norm = spy_series / spy_series.iloc[0] * port_series.iloc[0]
    last_date = port_series.index[-1]
    rows = []

    for ep in episodes:
        peak, trough, recovery = ep["peak_date"], ep["trough_date"], ep["recovery"]
        pdd = ep["dd"] * 100
        n_days = (trough - peak).days

        # SPY drawdown during same window
        end = recovery if recovery else last_date
        sw = spy_norm.loc[peak:end]
        if sw.empty:
            sdd = 0.0
            spy_trough = peak
        else:
            sddw = sw / sw.cummax() - 1.0
            sdd = sddw.min() * 100
            spy_trough = sddw.idxmin()
        ratio = abs(pdd / sdd) if sdd != 0 else 0.0

        # Portfolio recovery
        if recovery:
            split_recov = fmt_duration((recovery - trough).days)
            split_total = fmt_duration((recovery - peak).days)
        else:
            split_recov = f"ongoing ({fmt_duration((last_date - trough).days)})"
            split_total = f"ongoing ({fmt_duration((last_date - peak).days)})"

        # SPY recovery
        if not sw.empty:
            spy_peak_val = spy_norm.loc[peak]
            spy_after = spy_norm.loc[spy_trough:]
            spy_recovered = spy_after[spy_after >= spy_peak_val]
            if len(spy_recovered) > 0:
                spy_rd = spy_recovered.index[0]
                spy_recov = fmt_duration((spy_rd - spy_trough).days)
                spy_total = fmt_duration((spy_rd - peak).days)
            else:
                spy_recov = f"ongoing ({fmt_duration((last_date - spy_trough).days)})"
                spy_total = f"ongoing ({fmt_duration((last_date - peak).days)})"
        else:
            spy_recov = "N/A"
            spy_total = "N/A"

        period = f"{peak.strftime('%b %d, %Y')} - {trough.strftime('%b %d, %Y')}"
        if not recovery:
            period += "*"

        rows.append({
            "Correction Period": period,
            "Days": n_days,
            "% Decline": f"{pdd:.1f}%",
            "Recovery from Bottom": split_recov,
            "Decline + Recovery Time": split_total,
            "SPY DD": f"{sdd:.1f}%",
            "SPY Recovery from Bottom": spy_recov,
            "SPY Decline + Recovery Time": spy_total,
            "Ratio": f"{ratio:.1f}x",
            "Market Event": get_market_event(peak),
            "_ongoing": recovery is None,
            "_severity": _classify_severity(pdd),
            "_decline_raw": pdd,
            "_spy_dd_raw": sdd,
            "_ratio_raw": ratio,
            "_days_raw": n_days,
        })

    return pd.DataFrame(rows, columns=columns)
```

- [ ] **Step 4: Add export to `__init__.py`**

In `app/core/calculations/__init__.py`, add `build_drawdown_table` to the imports from `stats` and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_calculations.py -k "build_drawdown_table" -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/calculations/stats.py app/core/calculations/__init__.py tests/test_calculations.py
git commit -m "feat: add build_drawdown_table() with SPY comparison"
```

---

### Task 4: Add the Drawdowns tab to Returns Analysis

**Files:**
- Modify: `app/ui/charts/returns.py:536` (add 6th tab)
- Modify: `app/ui/charts/returns.py` (add tab content after `tab_daily` block, before the final closing of `render_returns_analysis`)

- [ ] **Step 1: Add the 6th tab definition**

In `app/ui/charts/returns.py`, change line 536 from:

```python
    tab_summary, tab_annual, tab_quarterly, tab_monthly, tab_daily = st.tabs(["📋 Summary", "📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily"])
```

to:

```python
    tab_summary, tab_annual, tab_quarterly, tab_monthly, tab_daily, tab_drawdowns = st.tabs(["📋 Summary", "📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily", "📉 Drawdowns"])
```

- [ ] **Step 2: Add the tab content**

After the `with tab_daily:` block (after line 680), add:

```python
    with tab_drawdowns:
        st.subheader(f"{portfolio_name} Corrections >5%")

        # Fetch SPYSIM for market comparison
        from app.services.data_service import fetch_component_data
        from app.core.calculations.stats import build_drawdown_table

        start_date = port_series.index[0].strftime("%Y-%m-%d")
        end_date = port_series.index[-1].strftime("%Y-%m-%d")

        try:
            spy_prices = fetch_component_data(["SPYSIM"], start_date, end_date)
            spy_col = spy_prices.columns[0]
            spy_raw = spy_prices[spy_col].reindex(port_series.index).ffill().bfill()
            spy_norm = spy_raw / spy_raw.iloc[0] * port_series.iloc[0]
        except Exception:
            spy_norm = port_series.copy()
            st.warning("Could not load SPY benchmark data. SPY columns may be inaccurate.")

        df = build_drawdown_table(port_series, spy_norm)

        if df.empty:
            st.info("No corrections >5% found in this period.")
        else:
            # Summary metrics
            n_total = len(df)
            median_decline = df["_decline_raw"].median()
            n_severe = (df["_severity"] == "Severe").sum()
            n_moderate = (df["_severity"] == "Moderate").sum()
            n_ongoing = df["_ongoing"].sum()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Corrections", n_total)
            c2.metric("Median Decline", f"{median_decline:.1f}%")
            c3.metric("Severe (>25%)", n_severe)
            c4.metric("Moderate (15-25%)", n_moderate)
            c5.metric("Ongoing", n_ongoing)

            # Severity filter
            n_mild = (df["_severity"] == "Mild").sum()
            n_minor = (df["_severity"] == "Minor").sum()
            filter_options = [
                f"All ({n_total})",
                f"Severe >25% ({n_severe})",
                f"Moderate 15-25% ({n_moderate})",
                f"Mild 10-15% ({n_mild})",
                f"Minor 5-10% ({n_minor})",
            ]
            selected = st.radio(
                "Filter by severity",
                filter_options,
                horizontal=True,
                label_visibility="collapsed",
                key=f"dd_filter_{unique_id}",
            )

            # Apply filter
            filtered = df
            if "Severe" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Severe"]
            elif "Moderate" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Moderate"]
            elif "Mild" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Mild"]
            elif "Minor" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Minor"]

            # Build median row
            if not filtered.empty:
                median_row = pd.DataFrame([{
                    "Correction Period": "Median",
                    "Days": int(filtered["_days_raw"].median()),
                    "% Decline": f"{filtered['_decline_raw'].median():.1f}%",
                    "Recovery from Bottom": "",
                    "Decline + Recovery Time": "",
                    "SPY DD": f"{filtered['_spy_dd_raw'].median():.1f}%",
                    "SPY Recovery from Bottom": "",
                    "SPY Decline + Recovery Time": "",
                    "Ratio": f"{filtered['_ratio_raw'].median():.1f}x",
                    "Market Event": "",
                    "_ongoing": False,
                    "_severity": "",
                    "_decline_raw": 0,
                    "_spy_dd_raw": 0,
                    "_ratio_raw": 0,
                    "_days_raw": 0,
                }])
                display_df = pd.concat([filtered, median_row], ignore_index=True)
            else:
                display_df = filtered.copy()

            # Drop hidden columns for display
            display_cols = [c for c in display_df.columns if not c.startswith("_")]
            display_df = display_df[display_cols]

            # Style function
            def style_drawdowns(styler):
                def color_decline(val):
                    if not isinstance(val, str) or not val.endswith("%"):
                        return ""
                    try:
                        v = float(val.replace("%", ""))
                    except ValueError:
                        return ""
                    if abs(v) >= 25: return "color: #ef4444; font-weight: bold"
                    if abs(v) >= 15: return "color: #f97316; font-weight: bold"
                    if abs(v) >= 10: return "color: #eab308"
                    return "color: #94a3b8"

                def color_ratio(val):
                    if not isinstance(val, str) or not val.endswith("x"):
                        return ""
                    try:
                        v = float(val.replace("x", ""))
                    except ValueError:
                        return ""
                    if v < 1.5: return "color: #34d399"
                    if v > 3.0: return "color: #f97316"
                    return ""

                def color_ongoing(val):
                    if isinstance(val, str) and "ongoing" in val:
                        return "color: #ef4444; font-weight: bold"
                    return ""

                def color_spy(val):
                    if isinstance(val, str) and "ongoing" in val:
                        return "color: #ef4444; font-weight: bold"
                    return "color: #94a3b8"

                def color_median_row(row):
                    if row["Correction Period"] == "Median":
                        return ["color: #3b82f6; font-weight: bold"] * len(row)
                    return [""] * len(row)

                styler.map(color_decline, subset=["% Decline"])
                styler.map(color_ratio, subset=["Ratio"])
                styler.map(color_ongoing, subset=["Recovery from Bottom", "Decline + Recovery Time"])
                styler.map(color_spy, subset=["SPY DD", "SPY Recovery from Bottom", "SPY Decline + Recovery Time"])
                styler.apply(color_median_row, axis=1)
                return styler

            st.dataframe(
                display_df.style.pipe(style_drawdowns),
                use_container_width=True,
                hide_index=True,
                height=min(800, 35 * (len(display_df) + 1) + 38),
            )
```

- [ ] **Step 3: Verify the app runs**

Run: `streamlit run testfol_charting.py --server.port 8501`

Open in browser, run a backtest, go to Returns Analysis > Drawdowns tab. Verify:
- Summary metrics appear
- Severity filter works
- Table renders with correct columns
- Colors are applied (decline severity, ratio, ongoing, SPY gray, median blue)

- [ ] **Step 4: Commit**

```bash
git add app/ui/charts/returns.py
git commit -m "feat: add Drawdowns tab to Returns Analysis"
```

---

### Task 5: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS (no regressions)

- [ ] **Step 2: Run drawdown-specific tests**

Run: `python -m pytest tests/test_calculations.py -k "drawdown or fmt_duration or market_event" -v`
Expected: All 13 tests PASS

- [ ] **Step 3: Commit any fixes if needed**

Only if test failures require fixes.

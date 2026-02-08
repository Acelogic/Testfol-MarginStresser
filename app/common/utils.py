import pandas as pd
import streamlit as st
import json
import os

def color_return(val):
    if pd.isna(val): return ""
    color = '#00CC96' if val >= 0 else '#EF553B'
    return f'color: {color}'

def num_input(label, key, default, step, **kwargs):
    return st.number_input(
        label,
        value=st.session_state.get(key, default),
        step=step,
        key=key,
        **kwargs
    )

def sync_equity():
    sv = st.session_state.get("g_start", 10000.0)
    loan = st.session_state.get("starting_loan", 0.0)
    if sv > 0:
        st.session_state.equity_init = 100 * max(0, 1 - loan / sv)

def sync_loan():
    sv = st.session_state.get("g_start", 10000.0)
    eq = st.session_state.get("equity_init", 100.0)
    st.session_state.starting_loan = sv * max(0, 1 - eq / 100)

def get_presets_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "../../data/presets.json")

def load_presets():
    path = get_presets_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return []
    return []

def save_preset(preset_data):
    """
    Saves a preset dict to presets.json.
    Updates existing if name matches, else appends.
    """
    path = get_presets_path()
    current_presets = load_presets()
    
    # Check if exists
    existing_idx = next((i for i, p in enumerate(current_presets) if p["name"] == preset_data["name"]), -1)
    
    if existing_idx >= 0:
        current_presets[existing_idx] = preset_data
    else:
        current_presets.append(preset_data)
        
    with open(path, "w") as f:
        json.dump(current_presets, f, indent=4)

def delete_preset(name):
    path = get_presets_path()
    current_presets = load_presets()
    current_presets = [p for p in current_presets if p["name"] != name]
    with open(path, "w") as f:
        json.dump(current_presets, f, indent=4)

def resample_data(series: pd.Series, timeframe: str, method="ohlc") -> pd.DataFrame:
    if timeframe == "1D":
        if method == "ohlc":
            df = series.to_frame(name="Close")
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            return df
        else:
            return series

    rule_map = {
        "1W": "W-FRI",
        "1M": "ME",
        "3M": "QE",
        "1Y": "YE"
    }
    rule = rule_map.get(timeframe, "ME")

    if method == "ohlc":
        ohlc = series.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })
        return ohlc.dropna()
    elif method == "max":
        return series.resample(rule).max().dropna()
    else:
        return series.resample(rule).last().dropna()

def render_documentation():
    st.sidebar.title("📚 Documentation")
    st.sidebar.markdown("---")
    
    docs_dir = "docs"
    available_docs = {
        "User Guide": "user_guide.md",
        "Methodology": "methodology.md",
        "FAQ & Troubleshooting": "faq.md"
    }
    
    doc_selection = st.sidebar.radio("Select Topic", list(available_docs.keys()))
    
    doc_file = available_docs[doc_selection]
    doc_path = os.path.join(docs_dir, doc_file)
    
    if os.path.exists(doc_path):
        with open(doc_path, "r") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error(f"Documentation file not found: {doc_path}")

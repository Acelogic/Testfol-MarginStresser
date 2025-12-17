import pandas as pd
import streamlit as st
import json
import os

PORTFOLIO_FILE = "data/saved_portfolios.json"

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

def load_saved_portfolios():
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_portfolio_to_disk(name, config):
    data = load_saved_portfolios()
    data[name] = config
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def delete_portfolio_from_disk(name):
    data = load_saved_portfolios()
    if name in data:
        del data[name]
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f, indent=4)

def sync_equity():
    sv = st.session_state.start_val
    loan = st.session_state.starting_loan
    st.session_state.equity_init = 100 * max(0, 1 - loan / sv)

def sync_loan():
    sv = st.session_state.start_val
    eq = st.session_state.equity_init
    st.session_state.starting_loan = sv * max(0, 1 - eq / 100)

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

import os

# --- General ---
# --- General ---
# Start from 'src' directory
# Base directory: data/etf_xray
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
ETF_CACHE_DIR = os.path.join(CACHE_DIR, "etf_filings")

DEBUG_DIR = os.path.join(BASE_DIR, "debug")

# Irrelevant but keeping for syntax safety if accessed (though shouldn't be)

# --- ETF X-Ray Settings ---
# (Add any ETF-specific settings here in the future)


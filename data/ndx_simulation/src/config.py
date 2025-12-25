import os

# --- General ---
# --- General ---
# Start from 'src' directory
# Base directory: data/ndx_simulation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ASSETS_DIR = os.path.join(DATA_DIR, "assets")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
NDX_CACHE_DIR = os.path.join(CACHE_DIR, "ndx_filings")
# Legacy alias
DOWNLOAD_DIR = NDX_CACHE_DIR

DEBUG_DIR = os.path.join(BASE_DIR, "debug") # Debug can stay at root of module

COMPONENTS_FILE = os.path.join(ASSETS_DIR, "nasdaq_components.csv")
WEIGHTS_FILE = os.path.join(RESULTS_DIR, "nasdaq_quarterly_weights.csv")
CHANGES_FILE = os.path.join(ASSETS_DIR, "nasdaq_changes.csv")
PRICE_CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.pkl")
BENCHMARK_TICKER = "^NDX"

# --- Common Methodology Constants ---
# (Can be overridden by specific strategies)

# --- NDX Mega 1.0 Settings ---
MEGA1_TARGET_THRESHOLD = 0.47  # Select top 47% cumulative weight
MEGA1_BUFFER_THRESHOLD = 0.50  # Buffer to 50%
MEGA1_SINGLE_STOCK_CAP = 0.35  # Cap single stock at 35%

# --- NDX Mega 2.0 Settings ---
MEGA2_TARGET_THRESHOLD = 0.47  # Select top 47% cumulative weight
MEGA2_BUFFER_THRESHOLD = 0.50  # Buffer to 50%
MEGA2_SINGLE_STOCK_CAP = 0.30  # Cap single stock at 30%
MEGA2_MIN_CONSTITUENTS = 9     # Minimum 9 stocks

# --- Validation Settings ---
WEIGHT_TOLERANCE = 0.01        # Allow 1% deviation (handling fillers etc)
MAX_CAP_ITERATIONS = 20        # Increase from 10 to ensure convergence

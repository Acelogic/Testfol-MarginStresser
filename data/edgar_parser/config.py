# NDX Mega Strategy Configuration

# --- General ---
DOWNLOAD_DIR = "downloads"
COMPONENTS_FILE = "nasdaq_components.csv"
WEIGHTS_FILE = "nasdaq_quarterly_weights.csv"
CHANGES_FILE = "nasdaq_changes.csv"
PRICE_CACHE_FILE = "prices_cache.pkl"
BENCHMARK_TICKER = "^NDX"

# --- Common Methodology Constants ---
# (Can be overridden by specific strategies)

# --- NDX Mega 1.0 Settings ---
MEGA1_TARGET_THRESHOLD = 0.47  # Select top 47% cumulative weight
MEGA1_BUFFER_THRESHOLD = 0.50  # Buffer to 50%
MEGA1_SINGLE_STOCK_CAP = 0.35  # Cap single stock at 35%

# --- NDX Mega 2.0 Settings ---
MEGA2_TARGET_THRESHOLD = 0.40  # Select top 40% cumulative weight
MEGA2_BUFFER_THRESHOLD = 0.45  # Buffer to 45%
MEGA2_SINGLE_STOCK_CAP = 0.30  # Cap single stock at 30%
MEGA2_MIN_CONSTITUENTS = 9     # Minimum 9 stocks

# --- Validation Settings ---
WEIGHT_TOLERANCE = 0.01        # Allow 1% deviation (handling fillers etc)
MAX_CAP_ITERATIONS = 20        # Increase from 10 to ensure convergence

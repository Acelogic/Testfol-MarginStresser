import subprocess
import os
import sys
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path, desc, env=None):
    """Run a Python script with optional environment variables."""
    logging.info(f"--- Starting: {desc} ---")
    try:
        # verify file exists
        if not os.path.exists(script_path):
             logging.error(f"Script not found: {script_path}")
             sys.exit(1)

        # Merge with current environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Run with python3
        result = subprocess.run([sys.executable, script_path], check=True, text=True, env=run_env)
        logging.info(f"--- Completed: {desc} ---\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Failed: {desc} (Exit Code: {e.returncode}) !!!")
        sys.exit(e.returncode)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Rebuild NDX/NDXMEGA/NDXMEGA2 simulation pipeline.',
        epilog='''
Examples:
  python rebuild_all.py                          # Use yfinance (default)
  python rebuild_all.py --stooq                  # Use Stooq (free, 20+ years)
  python rebuild_all.py --polygon YOUR_API_KEY   # Use Polygon.io (paid)
        '''
    )
    parser.add_argument(
        '--polygon', 
        metavar='API_KEY',
        help='Use Polygon.io as data source with the provided API key (paid)'
    )
    parser.add_argument(
        '--stooq',
        action='store_true',
        help='Use Stooq as data source (free, 20+ years of data)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip SEC filing download step (use existing filings)'
    )
    parser.add_argument(
        '--skip-reconstruct',
        action='store_true',
        help='Skip weight reconstruction step (use existing weights)'
    )
    args = parser.parse_args()

    # Build environment variables to pass to child scripts
    env_vars = {}
    
    if args.polygon:
        env_vars['NDX_DATA_SOURCE'] = 'polygon'
        env_vars['POLYGON_API_KEY'] = args.polygon
        logging.info(f"Data Source: Polygon.io (paid)")
    elif args.stooq:
        env_vars['NDX_DATA_SOURCE'] = 'stooq'
        logging.info(f"Data Source: Stooq (free)")
    else:
        env_vars['NDX_DATA_SOURCE'] = 'yfinance'
        logging.info(f"Data Source: yfinance (default)")
    
    # Base paths
    # This script is in data/ndx_simulation/scripts/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    module_root = os.path.dirname(scripts_dir)
    src_dir = os.path.join(module_root, "src")
    
    logging.info("Starting Full NDX Simulation Rebuild...")
    
    # 1. Download Filings (ndx_downloader.py is in src/)
    if not args.skip_download:
        downloader_script = os.path.join(src_dir, "ndx_downloader.py")
        run_script(downloader_script, "Download SEC Filings", env_vars)
    else:
        logging.info("--- Skipped: Download SEC Filings ---\n")
    
    # 2. Reconstruct Weights (scripts/reconstruct_weights.py)
    if not args.skip_reconstruct:
        reconstruct_script = os.path.join(scripts_dir, "reconstruct_weights.py")
        run_script(reconstruct_script, "Reconstruct Index Weights", env_vars)
    else:
        logging.info("--- Skipped: Reconstruct Index Weights ---\n")
    
    # 3. Backtest Mega 1.0 (scripts/backtest_ndx_mega.py)
    mega1_script = os.path.join(scripts_dir, "backtest_ndx_mega.py")
    run_script(mega1_script, "Backtest NDX Mega 1.0", env_vars)
    
    # 4. Backtest Mega 2.0 (scripts/backtest_ndx_mega2.py)
    mega2_script = os.path.join(scripts_dir, "backtest_ndx_mega2.py")
    run_script(mega2_script, "Backtest NDX Mega 2.0", env_vars)
    
    logging.info("All steps completed successfully.")
    logging.info("Dashboard data (NDXMEGASIM.csv / NDXMEGA2SIM.csv) has been updated.")
    
    # Summary
    if args.polygon:
        logging.info(f"Data source used: Polygon.io")
    else:
        logging.info(f"Data source used: yfinance")

if __name__ == "__main__":
    main()

import subprocess
import os
import sys
import logging
import argparse
import shutil
import glob

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
        description='Rebuild NDX/NDXMEGA/NDXMEGA2/NDX30 simulation pipeline.',
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
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='DEEP CLEAN: Deletes all downloaded data, caches, and results before running.'
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
    
    # Paths from config (replicated here or imported? safer to replicate relative to root to avoid ImportErrors if config moves)
    # But we can assume module structure.
    data_dir = os.path.join(module_root, "data")
    cache_dir = os.path.join(data_dir, "cache")
    assets_dir = os.path.join(data_dir, "assets")
    results_dir = os.path.join(data_dir, "results")
    charts_dir = os.path.join(results_dir, "charts")
    
    if args.clear_cache:
        logging.warning("!!! --clear-cache SET: Performing Deep Clean !!!")
        logging.info(f"Cleaning {data_dir} artifacts...")
        
        # 1. Price Caches
        for pattern in ["prices_cache.pkl", "prices_cache_*.pkl"]:
            for f in glob.glob(os.path.join(cache_dir, pattern)):
                try:
                    os.remove(f)
                    logging.info(f"Deleted: {f}")
                except OSError as e:
                    logging.error(f"Error deleting {f}: {e}")
                    
        # 2. NDX Filings (Downloads)
        filings_dir = os.path.join(cache_dir, "ndx_filings")
        if os.path.exists(filings_dir):
            try:
                shutil.rmtree(filings_dir)
                logging.info(f"Deleted Directory: {filings_dir}")
            except OSError as e:
                logging.error(f"Error deleting {filings_dir}: {e}")
        
        # 3. Assets (Components CSV)
        comp_file = os.path.join(assets_dir, "nasdaq_components.csv")
        if os.path.exists(comp_file):
            try:
                os.remove(comp_file)
                logging.info(f"Deleted: {comp_file}")
            except OSError as e:
                logging.error(f"Error deleting {comp_file}: {e}")

        # 4. Results (Weights)
        w_file = os.path.join(results_dir, "nasdaq_quarterly_weights.csv")
        if os.path.exists(w_file):
            try:
                os.remove(w_file)
                logging.info(f"Deleted: {w_file}")
            except OSError as e:
                logging.error(f"Error deleting {w_file}: {e}")
        
        # 5. Charts
        if os.path.exists(charts_dir):
            # Delete all pngs
            for f in glob.glob(os.path.join(charts_dir, "*.png")):
                try:
                    os.remove(f)
                    logging.info(f"Deleted: {f}")
                except OSError as e:
                     logging.error(f"Error deleting {f}: {e}")
                     
        # 6. Top Level Simulation Outputs (NDXMEGASIM.csv)
        # Assuming they are in data/ndx_simulation/.. (based on backtest scripts)
        # ../NDXMEGASIM.csv relative to module_root
        parent_dir = os.path.dirname(module_root) # Testfol-MarginStresser/data usually? No, module_root is data/ndx_simulation. Parent is data.
        # Actually backtest script uses: os.path.join(config.BASE_DIR, "..", "NDXMEGASIM.csv")
        # config.BASE_DIR is data/ndx_simulation
        # So it is in Testfol-MarginStresser/data/NDXMEGASIM.csv (if that's where module is)
        # Let's rely on relative path logic matching backtest
        
        for sim_file in ["NDXMEGASIM.csv", "NDXMEGA2SIM.csv", "NDX30SIM.csv"]:
            p = os.path.abspath(os.path.join(module_root, "..", sim_file))
            if os.path.exists(p):
                 try:
                    os.remove(p)
                    logging.info(f"Deleted: {p}")
                 except OSError as e:
                    logging.error(f"Error deleting {p}: {e}")
                    
        logging.info("--- Deep Clean Completed ---\n")
    
    logging.info("Starting Full NDX Simulation Rebuild...")
    
    # 1. Download Filings (ndx_downloader.py is in src/)
    if not args.skip_download:
        downloader_script = os.path.join(src_dir, "ndx_downloader.py")
        run_script(downloader_script, "Download SEC Filings", env_vars)
        
        # 1.5. Parse Filings (ndx_parser.py is in src/)
        # Must run after download to generate nasdaq_components.csv
        parser_script = os.path.join(src_dir, "ndx_parser.py")
        run_script(parser_script, "Parse SEC Filings", env_vars)
    else:
        logging.info("--- Skipped: Download & Parse SEC Filings ---\n")
    
    # 2. Update Name Mappings (src/mapper.py)
    # This ensures new filings are mapped to tickers before reconstruction
    mapper_script = os.path.join(src_dir, "mapper.py")
    run_script(mapper_script, "Update Name Mappings", env_vars)

    # 3. Reconstruct Weights (scripts/reconstruct_weights.py)
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

    # 4.5. Backtest NDX30 (scripts/backtest_ndx30.py)
    ndx30_script = os.path.join(scripts_dir, "backtest_ndx30.py")
    run_script(ndx30_script, "Backtest NDX30", env_vars)

    # 5. Validation (scripts/validate_ndx.py)
    validate_script = os.path.join(scripts_dir, "validate_ndx.py")
    run_script(validate_script, "Validate & Compare Results", env_vars)
    
    logging.info("All steps completed successfully.")
    logging.info("Dashboard data (NDXMEGASIM.csv / NDXMEGA2SIM.csv / NDX30SIM.csv) has been updated.")
    
    # Summary
    if args.polygon:
        logging.info(f"Data source used: Polygon.io")
    else:
        logging.info(f"Data source used: yfinance")

if __name__ == "__main__":
    main()

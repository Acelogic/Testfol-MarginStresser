import subprocess
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path, desc):
    logging.info(f"--- Starting: {desc} ---")
    try:
        # verify file exists
        if not os.path.exists(script_path):
             logging.error(f"Script not found: {script_path}")
             sys.exit(1)

        # Run with python3
        result = subprocess.run([sys.executable, script_path], check=True, text=True)
        logging.info(f"--- Completed: {desc} ---\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Failed: {desc} (Exit Code: {e.returncode}) !!!")
        sys.exit(e.returncode)

def main():
    # Base paths
    # This script is in data/ndx_simulation/scripts/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    module_root = os.path.dirname(scripts_dir)
    src_dir = os.path.join(module_root, "src")
    
    logging.info("Starting Full NDX Simulation Rebuild...")
    
    # 1. Download Filings (ndx_downloader.py is in src/)
    downloader_script = os.path.join(src_dir, "ndx_downloader.py")
    run_script(downloader_script, "Download SEC Filings")
    
    # 2. Reconstruct Weights (scripts/reconstruct_weights.py)
    reconstruct_script = os.path.join(scripts_dir, "reconstruct_weights.py")
    run_script(reconstruct_script, "Reconstruct Index Weights")
    
    # 3. Backtest Mega 1.0 (scripts/backtest_ndx_mega.py)
    mega1_script = os.path.join(scripts_dir, "backtest_ndx_mega.py")
    run_script(mega1_script, "Backtest NDX Mega 1.0")
    
    # 4. Backtest Mega 2.0 (scripts/backtest_ndx_mega2.py)
    mega2_script = os.path.join(scripts_dir, "backtest_ndx_mega2.py")
    run_script(mega2_script, "Backtest NDX Mega 2.0")
    
    logging.info("All steps completed successfully.")
    logging.info("Dashboard data (NDXMEGASIM.csv / NDXMEGA2SIM.csv) has been updated.")

if __name__ == "__main__":
    main()

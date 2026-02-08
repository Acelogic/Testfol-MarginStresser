#!/usr/bin/env python3
"""
Process launcher: starts FastAPI backend and Streamlit frontend.

Usage:
    python run.py
"""
import subprocess
import sys
import signal
import os

procs = []


def cleanup(*_):
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # Start FastAPI
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=root,
    )
    procs.append(api_proc)
    print("FastAPI started on http://localhost:8000")

    # Start Streamlit
    st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "testfol_charting.py", "--server.port", "8501"],
        cwd=root,
    )
    procs.append(st_proc)
    print("Streamlit started on http://localhost:8501")

    # Wait for either to exit
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"Process {p.args} exited with code {ret}")
                    cleanup()
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

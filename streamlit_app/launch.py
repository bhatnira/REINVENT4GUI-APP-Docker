#!/usr/bin/env python3
"""
Launch script for REINVENT4 Streamlit Web Application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch REINVENT4 Streamlit Web Application")
    parser.add_argument("--port", type=int, default=8502, help="Port to run the application on")
    parser.add_argument("--host", default="localhost", help="Host to run the application on")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies before launching")
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    app_dir = Path(__file__).parent
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing dependencies...")
        requirements_file = app_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            print("Warning: requirements.txt not found")
    
    # Set up environment
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
    
    if args.dev:
        os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "auto"
        os.environ["STREAMLIT_SERVER_RUNON_SAVE"] = "true"
    
    # Change to app directory
    os.chdir(app_dir)
    
    # Launch Streamlit
    app_file = app_dir / "app.py"
    
    print(f"Launching REINVENT4 Web Interface on {args.host}:{args.port}")
    print(f"Application file: {app_file}")
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_file),
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--server.headless", "false" if args.dev else "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nShutting down REINVENT4 Web Interface...")
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

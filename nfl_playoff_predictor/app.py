"""
Streamlit app launcher for NFL Playoff Predictor.

This module provides a function to launch the Streamlit app.
"""

import subprocess
import sys
from pathlib import Path


def launch_app():
    """Launch the Streamlit app."""
    # Get the path to streamlit_app.py (in the package directory)
    package_dir = Path(__file__).parent
    app_file = package_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"Error: Could not find streamlit_app.py at {app_file}")
        print("Please ensure the package is properly installed.")
        sys.exit(1)
    
    # Launch Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])


if __name__ == "__main__":
    launch_app()


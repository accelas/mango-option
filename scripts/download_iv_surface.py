#!/usr/bin/env python3
"""
Standalone wrapper script for IV surface calculator.

This script can be run directly from the scripts/ directory:
    python download_iv_surface.py AAPL

It automatically sets up the Python path and calls the main module.
"""

import sys
import os

# Add iv_surface module to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import and run the main function
from iv_surface.calculate_iv_surface import main

if __name__ == '__main__':
    main()

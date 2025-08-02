"""
This script analyzes the data inside of a given directory.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze data in a given directory.")
    parser.add_argument("directory", type=str, help="Directory containing the data files.")
    parser.add_argument("--star", action="store_true", help="Star network directory.")
    return parser.parse_args()

def main():
    # Get command line arguments
    args = parse_args()
    

    pass

if __name__ == "__main__":
    main()


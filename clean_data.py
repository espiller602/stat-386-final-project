#!/usr/bin/env python
"""
Script for cleaning NFL playoff passing data.

This script provides a command-line interface to the data cleaning functions
in the nfl_playoff_predictor package.

Usage:
    python clean_data.py [data_dir] [output_file] [start_year] [end_year]

Arguments:
    data_dir: Directory containing data files (default: "data sources")
    output_file: Path to output CSV file (default: "clean_playoff_passing.csv")
    start_year: First year to include (default: 2018)
    end_year: Last year to include (default: 2024)

Examples:
    python clean_data.py
    python clean_data.py "data sources" "output.csv"
    python clean_data.py "data sources" "output.csv" 2019 2023
"""

import sys
from pathlib import Path
from nfl_playoff_predictor.wrangling import process_and_save_dataset


def main():
    """Main function to run the data cleaning script."""
    # Parse command-line arguments
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data sources"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clean_playoff_passing.csv"
    start_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2018
    end_year = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        print(f"Please ensure the directory exists and contains the required data files.")
        sys.exit(1)
    
    print(f"Processing data from {data_dir}...")
    print(f"Years: {start_year} to {end_year}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        # Process and save the dataset
        df = process_and_save_dataset(
            data_dir=data_dir,
            output_file=output_file,
            start_year=start_year,
            end_year=end_year
        )
    
        print(f"✓ Successfully processed {len(df)} records")
        print(f"✓ Saved cleaned data to {output_file}")
        print(f"✓ Dataset contains {len(df.columns)} columns")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


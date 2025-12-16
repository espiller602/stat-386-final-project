"""
Data cleaning functions for NFL playoff passing statistics.

This module provides functions to load, clean, and prepare NFL playoff passing
data from Pro-Football-Reference HTML tables.
"""

import pandas as pd
import os


def build_advanced_dataset(data_dir, start_year=2018, end_year=2024):
    """
    Build a combined dataset from NFL playoff passing statistics.
    
    This function reads advanced and standard passing statistics from HTML files
    for multiple seasons, merges them, and extracts playoff wins information.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files. Files should be named in the format:
        '{year}_Playoffs_Advanced.xls' and '{year}_Standard_Passing.xls'
    start_year : int, default=2018
        First year to include in the dataset
    end_year : int, default=2024
        Last year to include in the dataset (inclusive)
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with advanced and standard passing statistics, including
        a 'playoff_games_won' column extracted from QBrec.
    
    Examples
    --------
    >>> df = build_advanced_dataset("data sources", start_year=2018, end_year=2024)
    >>> print(df.head())
    """
    adv_list = []
    std_list = []

    for year in range(start_year, end_year + 1):

        # --- Advanced passing ---
        adv_files = [f for f in os.listdir(data_dir) if f"{year}_Playoffs_Advanced" in f]
        if not adv_files:
            print(f"No advanced playoff file found for {year}, skipping")
            continue
        adv_file = adv_files[0]

        # Read with two-level header
        adv_df = pd.read_html(os.path.join(data_dir, adv_file), header=[0,1])[0]

        # Drop all columns under RPO or Play Action
        drop_sections = ['RPO', 'PlayAction']
        cols_to_drop = [col for col in adv_df.columns if col[0] in drop_sections]
        adv_df = adv_df.drop(columns=cols_to_drop)

        # Flatten MultiIndex header → single row of column names
        adv_df.columns = [
            (c[1] if c[1] not in ['', None] else c[0]) 
            for c in adv_df.columns
        ]
        adv_df.columns = adv_df.columns.astype(str).str.strip().str.replace('\xa0', '')

        # Clean fields
        adv_df['Player'] = adv_df['Player'].astype(str).str.strip()
        adv_df['Team'] = adv_df['Team'].astype(str).str.strip()
        adv_df['Season'] = year
        adv_list.append(adv_df)

        # --- Standard passing ---
        std_files = [f for f in os.listdir(data_dir) if f"{year}_Standard_Passing" in f]
        if not std_files:
            print(f"No standard passing file found for {year}, skipping")
            continue
        std_file = std_files[0]

        std_df = pd.read_html(os.path.join(data_dir, std_file), header=0)[0]
        std_df.columns = std_df.columns.astype(str).str.strip().str.replace('\xa0', '')
        std_df['Player'] = std_df['Player'].astype(str).str.strip()
        std_df['Team'] = std_df['Team'].astype(str).str.strip()
        std_df['Season'] = year
        std_list.append(std_df)

    # --- Concatenate all years ---
    adv_all = pd.concat(adv_list, ignore_index=True)
    std_all = pd.concat(std_list, ignore_index=True)
    
    # Filter out "League Average" rows before merging
    adv_all = adv_all[adv_all['Player'].astype(str).str.strip().str.lower() != 'league average']
    std_all = std_all[std_all['Player'].astype(str).str.strip().str.lower() != 'league average']

    # --- Merge standard onto advanced ---
    merged_df = adv_all.merge(std_all, on=['Player', 'Team', 'Season'], how='left')

    # --- Extract playoff wins from QBrec ---
    def extract_wins(qbrec):
        try:
            return int(str(qbrec).split('-')[0])
        except:
            return pd.NA

    merged_df['playoff_games_won'] = merged_df['QBrec'].apply(extract_wins)
    merged_df['playoff_games_won'] = pd.to_numeric(merged_df['playoff_games_won'], errors='coerce').astype('float')     
    return merged_df


def clean_column_names(df):
    """
    Clean column names by removing duplicate suffixes and renaming columns.
    
    After merging advanced and standard passing data, this function:
    1. Drops columns ending with '_y' (duplicates from standard passing)
    2. Removes '_x' suffix from remaining columns (from advanced passing)
    3. Renames 'Yds.1' to 'YdsLost' for clarity
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with merged advanced and standard passing statistics
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names
    
    Examples
    --------
    >>> df = build_advanced_dataset("data sources")
    >>> clean_df = clean_column_names(df)
    """
    # Drop columns ending with _y (duplicates from standard passing)
    cols_to_drop = [c for c in df.columns if c.endswith("_y")]
    clean_df = df.drop(columns=cols_to_drop)

    # Remove _x from the remaining columns
    clean_df.columns = clean_df.columns.str.replace("_x", "", regex=False)

    # Rename Yds.1 → YdsLost
    clean_df = clean_df.rename(columns={"Yds.1": "YdsLost"})

    return clean_df


def process_and_save_dataset(data_dir, output_file="clean_playoff_passing.csv", 
                            start_year=2018, end_year=2024):
    """
    Complete pipeline: build dataset, clean columns, and save to CSV.
    
    This is a convenience function that combines build_advanced_dataset() and
    clean_column_names() and saves the result to a CSV file.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files
    output_file : str, default="clean_playoff_passing.csv"
        Path to output CSV file
    start_year : int, default=2018
        First year to include in the dataset
    end_year : int, default=2024
        Last year to include in the dataset (inclusive)
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    
    Examples
    --------
    >>> df = process_and_save_dataset("data sources", "clean_playoff_passing.csv")
    """
    # Build the dataset
    df = build_advanced_dataset(data_dir, start_year=start_year, end_year=end_year)
    
    # Clean column names
    clean_df = clean_column_names(df)
    
    # Save to CSV
    clean_df.to_csv(output_file, index=False)
    
    return clean_df


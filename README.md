# NFL Playoff Predictor

A Python package for predicting NFL postseason wins using advanced passing statistics.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Package Structure

```
nfl_playoff_predictor/
├── __init__.py
├── wrangling/          # Data cleaning and preparation modules
│   └── __init__.py
└── analysis/           # Statistical analysis and modeling modules
    └── __init__.py
```

## Usage

```python
from nfl_playoff_predictor.wrangling import build_advanced_dataset

# Build dataset from raw data files
df = build_advanced_dataset("data sources", start_year=2018, end_year=2024)
```

## Authors

Eli Spiller, Zion Tippetts

## Project Status

This project is part of STAT 386 Final Project.
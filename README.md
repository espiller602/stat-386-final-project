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

### Data Wrangling

```python
from nfl_playoff_predictor.wrangling import (
    build_advanced_dataset,
    clean_column_names,
    process_and_save_dataset
)

# Option 1: Build and clean dataset in one step
df = process_and_save_dataset("data sources", "clean_playoff_passing.csv")

# Option 2: Build dataset, then clean columns separately
df = build_advanced_dataset("data sources", start_year=2018, end_year=2024)
clean_df = clean_column_names(df)
```

### Analysis

```python
from nfl_playoff_predictor.analysis import (
    get_default_model,
    predict_playoff_wins,
    train_poisson_model,
    evaluate_model
)

# Get the default trained model
model = get_default_model(df)

# Make a prediction
prediction = predict_playoff_wins(model, {
    'IAY_PA': 7.0,
    'YAC_Cmp': 5.0,
    'Int': 1,
    'DropPct': 5.0,
    'BadPct': 15.0
})
```

### Streamlit App

Run the interactive Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app provides:
- **Home**: Project information and navigation
- **Data Explorer**: Interactive data exploration with filters, statistics, and visualizations
- **Predictions**: Make predictions using the trained Poisson GLM model

## Authors

Eli Spiller, Zion Tippetts

## Project Status

This project is part of STAT 386 Final Project.
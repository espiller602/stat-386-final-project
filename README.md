# NFL Playoff Predictor

A Python package for predicting NFL postseason wins using advanced passing statistics.

## Installation

### From GitHub

Install directly from the GitHub repository:

```bash
pip install git+https://github.com/espiller602/stat-386-final-project.git
```

### Local Development

For local development and contributions:

```bash
git clone https://github.com/espiller602/stat-386-final-project.git
cd stat-386-final-project
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

### Data Cleaning Script

You can also use the standalone command-line script for data cleaning:

```bash
# Use defaults (data sources/, clean_playoff_passing.csv, 2018-2024)
python clean_data.py

# Specify custom directory and output file
python clean_data.py "data sources" "my_output.csv"

# Specify all parameters
python clean_data.py "data sources" "output.csv" 2019 2023
```

The script will:
- Process data from the specified directory
- Clean and merge advanced and standard passing statistics
- Save the cleaned dataset to a CSV file
- Display progress and summary information

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
    'IntPerAtt': 0.02  # IntPerAtt = Int / Att
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

## Documentation

Comprehensive documentation is available on [GitHub Pages](https://espiller602.github.io/stat-386-final-project/):

- [Home](https://espiller602.github.io/stat-386-final-project/) - Package overview and quick start
- [Data Wrangling](https://espiller602.github.io/stat-386-final-project/wrangling.html) - Data cleaning functions
- [Analysis](https://espiller602.github.io/stat-386-final-project/analysis.html) - Modeling and prediction functions
- [Tutorial](https://espiller602.github.io/stat-386-final-project/tutorial.html) - Step-by-step guide
- [Final Report](https://espiller602.github.io/stat-386-final-project/report.html) - Complete project report

The documentation is built using [Quarto](https://quarto.org/) and automatically deployed via GitHub Actions.

## Authors

Eli Spiller, Zion Tippetts

## Project Status

This project is part of STAT 386 Final Project.
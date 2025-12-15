"""
Analysis module for NFL playoff passing statistics.

This module contains functions for statistical analysis and modeling.
"""

from .modeling import (
    prepare_data,
    select_variables_lasso,
    train_poisson_model,
    predict_playoff_wins,
    evaluate_model,
    cross_validate_model,
    get_default_model
)

__all__ = [
    "prepare_data",
    "select_variables_lasso",
    "train_poisson_model",
    "predict_playoff_wins",
    "evaluate_model",
    "cross_validate_model",
    "get_default_model"
]


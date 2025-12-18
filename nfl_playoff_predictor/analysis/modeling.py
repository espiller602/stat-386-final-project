"""
Modeling functions for NFL playoff predictions.

This module contains functions for training, evaluating, and using Poisson GLM
models to predict NFL playoff wins based on advanced passing statistics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor


def prepare_data(df, predictors=None):
    """
    Prepare data for modeling by renaming columns and selecting predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with playoff passing statistics
    predictors : list, optional
        List of predictor column names. If None, uses default predictors.
    
    Returns
    -------
    pd.DataFrame
        Prepared dataframe with renamed columns
    list
        List of predictor names (with valid Python identifiers)
    """
    if predictors is None:
        predictors = [
            "IAY/PA",
            "CAY/PA",
            "YAC/Cmp",
            "Int",
            "Prss%",
            "PktTime",
            "Drop%",
            "Bad%"
        ]
    
    # Rename columns to valid Python identifiers
    rename_dict = {
        "IAY/PA": "IAY_PA",
        "CAY/PA": "CAY_PA",
        "YAC/Cmp": "YAC_Cmp",
        "Prss%": "PrssPct",
        "Drop%": "DropPct",
        "Bad%": "BadPct"
    }
    
    df_prep = df.rename(columns=rename_dict)
    
    # Map predictor names
    predictor_mapping = {old: rename_dict.get(old, old) for old in predictors}
    predictors_renamed = [predictor_mapping.get(p, p) for p in predictors]
    
    return df_prep, predictors_renamed


def select_variables_lasso(df, predictors, response="playoff_games_won", 
                          alpha=0.01, alphas=None):
    """
    Select variables using Poisson Lasso regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe
    predictors : list
        List of predictor column names (with valid Python identifiers)
    response : str, default="playoff_games_won"
        Response variable name
    alpha : float, default=0.01
        Final alpha value for variable selection
    alphas : list, optional
        List of alpha values to test. If None, uses default values.
    
    Returns
    -------
    list
        Selected variable names
    """
    if alphas is None:
        alphas = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    # Prepare data
    combined = df[predictors + [response]].dropna()
    X = combined[predictors].values
    y = combined[response].values
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    
    var_names = ['Intercept'] + predictors
    
    # Test different alphas
    print("Poisson Lasso Variable Selection:")
    for a in alphas:
        poisson_lasso = sm.GLM(y, X_scaled, family=sm.families.Poisson()).fit_regularized(
            method='elastic_net',
            alpha=a,
            L1_wt=1.0,
            maxiter=50000
        )
        coefs = poisson_lasso.params
        selected = [name for name, coef in zip(var_names, coefs) 
                   if coef != 0 and name != 'Intercept']
        print(f"alpha={a}: selected variables = {selected}")
    
    # Final selection with specified alpha
    poisson_lasso_final = sm.GLM(y, X_scaled, family=sm.families.Poisson()).fit_regularized(
        method='elastic_net',
        alpha=alpha,
        L1_wt=1.0,
        maxiter=50000
    )
    
    coefs_final = poisson_lasso_final.params
    selected_vars = [name for name, coef in zip(var_names, coefs_final) 
                   if coef != 0 and name != 'Intercept']
    
    print(f"\nFinal selected variables (alpha={alpha}): {selected_vars}")
    return selected_vars


def train_poisson_model(df, predictors, response="playoff_games_won"):
    """
    Train a Poisson GLM model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe
    predictors : list
        List of predictor column names
    response : str, default="playoff_games_won"
        Response variable name
    
    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted model
    """
    # Prepare data
    combined = df[predictors + [response]].dropna()
    
    # Build formula
    formula = response + " ~ " + " + ".join(predictors)
    
    # Fit Poisson GLM
    model = smf.glm(formula=formula, data=combined, family=sm.families.Poisson()).fit()
    
    return model


def predict_playoff_wins(model, predictors_dict):
    """
    Predict playoff wins for given predictor values.
    
    Parameters
    ----------
    model : statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted Poisson GLM model
    predictors_dict : dict
        Dictionary with predictor names as keys and values as values.
        Keys should match the model's predictor names.
    
    Returns
    -------
    float
        Predicted number of playoff wins
    """
    pred_df = pd.DataFrame([predictors_dict])

    prediction = model.predict(pred_df)

    return float(prediction.iloc[0])

def evaluate_model(model, df, predictors, response="playoff_games_won"):
    """
    Evaluate model performance and diagnostics.
    
    Parameters
    ----------
    model : statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted model
    df : pd.DataFrame
        Dataframe used for evaluation
    predictors : list
        List of predictor column names
    response : str, default="playoff_games_won"
        Response variable name
    
    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    # Prepare data
    combined = df[predictors + [response]].dropna()
    
    # Get residuals
    residuals = model.resid_pearson
    fitted = model.fittedvalues
    
    # Overdispersion check
    overdispersion = sum(residuals**2) / model.df_resid
    
    # VIF calculation
    X_selected = combined[predictors]
    X_const = sm.add_constant(X_selected)
    vif_values = [variance_inflation_factor(X_const.values, i+1) 
                  for i in range(len(predictors))]
    vif_df = pd.DataFrame({
        "Variable": predictors,
        "VIF": vif_values
    })
    
    # Model summary statistics
    results = {
        "overdispersion_ratio": overdispersion,
        "vif": vif_df,
        "residuals": residuals,
        "fitted": fitted,
        "pseudo_r_squared": model.pseudo_rsquared(kind='cs'),
        "aic": model.aic,
        "bic": model.bic
    }
    
    return results


def cross_validate_model(df, predictors, response="playoff_games_won", 
                         n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation for model selection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe
    predictors : list
        List of predictor column names to test
    response : str, default="playoff_games_won"
        Response variable name
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random state for reproducibility
    
    Returns
    -------
    list
        List of dictionaries with CV results, sorted by mean deviance
    """
    # Prepare data
    combined = df[predictors + [response]].dropna()
    
    # Set up k-fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_results = []
    
    # Generate all non-empty subsets of predictors
    for k in range(1, len(predictors) + 1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)
            fold_deviances = []
            
            for train_index, test_index in kf.split(combined):
                train = combined.iloc[train_index]
                test = combined.iloc[test_index]
                
                # Fit Poisson GLM
                model = smf.glm(formula=formula, data=train, 
                              family=sm.families.Poisson()).fit()
                mu_pred = model.predict(test)
                y_true = test[response].values
                
                # Compute Poisson deviance
                dev = 2 * np.sum(y_true * np.log(np.maximum(y_true, 1e-10) / mu_pred) 
                                - (y_true - mu_pred))
                fold_deviances.append(dev)
            
            # Save results
            cv_results.append({
                "predictors": subset,
                "fold_deviances": fold_deviances,
                "mean_deviance": np.mean(fold_deviances),
                "std_deviance": np.std(fold_deviances)
            })
    
    # Sort by mean deviance (lowest = best)
    cv_results_sorted = sorted(cv_results, key=lambda x: x['mean_deviance'])
    
    return cv_results_sorted


def get_default_model(df):
    """
    Get the default trained model using the selected variables.
    
    This function uses the final model configuration from the notebook:
    - Selected variables: ['IAY_PA', 'YAC_Cmp', 'IntPerAtt'] (from CV, Rank 1)
    - Model type: Poisson GLM
    - IntPerAtt is calculated as Int / Att
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with playoff passing statistics
    
    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted model
    """
    # Prepare data
    df_prep, _ = prepare_data(df)
    
    # Calculate IntPerAtt (Int / Att) - required for final model
    if 'Int' in df_prep.columns and 'Att' in df_prep.columns:
        df_prep['IntPerAtt'] = df_prep['Int'] / df_prep['Att']
    else:
        raise ValueError("Dataframe must contain 'Int' and 'Att' columns to calculate IntPerAtt")
    
    # Use the final selected variables from CV (best model: Rank 1)
    # This matches the final model in the notebook
    selected_vars = ['IAY_PA', 'YAC_Cmp', 'IntPerAtt']
    
    # Train model
    model = train_poisson_model(df_prep, selected_vars)
    
    return model


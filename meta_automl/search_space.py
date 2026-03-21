"""
search_space.py — Search Space Definitions
===========================================
Defines the full search space for the Genetic Algorithm, including:
  - Preprocessor choices (scaling)
  - Feature selector choices (PCA, SelectKBest)
  - Model choices with log-scale hyperparameter grids
  - Separate definitions for classification vs. regression tasks
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
SCALER_OPTIONS = ["None", "StandardScaler", "MinMaxScaler"]

FEATURE_SELECTOR_OPTIONS = ["None", "PCA", "SelectKBest"]

# Feature fraction: proportion of features to keep (used by PCA/SelectKBest)
FEATURE_FRACTION_RANGE = (0.3, 0.9)

# ──────────────────────────────────────────────────────────────────────────────
# MODELS — CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFICATION_MODELS = {
    "LogisticRegression": {
        # Log-scale regularization penalty
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [300, 500, 1000],
    },
    "SVC": {
        # Log-scale regularization and kernel bandwidth
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        # Log-scale learning rate
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        # Log-scale learning rate init (alpha = L2 penalty)
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "activation": ["relu", "tanh"],
        "max_iter": [300, 500],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# MODELS — REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
REGRESSION_MODELS = {
    "Ridge": {
        # Log-scale regularization penalty
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "lsqr"],
    },
    "SVR": {
        # Log-scale regularization and epsilon
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.001, 0.01, 0.1, 0.5],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        # Log-scale learning rate
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        # Log-scale alpha (L2 penalty) and learning rate init
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "activation": ["relu", "tanh"],
        "max_iter": [300, 500],
    },
}


def get_model_space(task_type: str) -> dict:
    """Return the model search space for the given task type.

    Args:
        task_type: 'classification' or 'regression'.

    Returns:
        Dict mapping model names to their hyperparameter grids.
    """
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    elif task_type == "regression":
        return REGRESSION_MODELS
    else:
        raise ValueError(f"Unknown task_type: '{task_type}'. Use 'classification' or 'regression'.")

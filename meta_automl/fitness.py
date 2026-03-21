"""
fitness.py — Fitness Evaluation with Timeout
=============================================
Builds a scikit-learn Pipeline from a Chromosome's genes and evaluates
its performance using K-Fold Cross Validation.

Key features:
  - Timeout constraint: pipelines exceeding the time limit are penalized.
  - try-except safety: invalid pipelines receive -inf fitness.
  - Supports both classification (f1_macro) and regression (neg RMSE).
"""

import signal
import warnings
import threading
import time
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

from meta_automl.chromosome import Chromosome

# ──────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY — maps model name strings to scikit-learn estimator classes
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFICATION_ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "MLPClassifier": MLPClassifier,
}

REGRESSION_ESTIMATORS = {
    "Ridge": Ridge,
    "SVR": SVR,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "MLPRegressor": MLPRegressor,
}


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def build_pipeline(
    chromosome: Chromosome,
    task_type: str,
    n_features: int,
    random_state: Optional[int] = None,
) -> Pipeline:
    """Construct a scikit-learn Pipeline from a chromosome's genes.

    Args:
        chromosome: The chromosome encoding the pipeline config.
        task_type: 'classification' or 'regression'.
        n_features: Number of features in the dataset (used to compute k/n_components).
        random_state: Seed for reproducibility in estimators.

    Returns:
        A scikit-learn Pipeline ready for fitting.
    """
    steps = []

    # ── Step 1: Feature Scaling ───────────────────────────────────────────
    if chromosome.scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    elif chromosome.scaler == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))
    # 'None' → no scaler step

    # ── Step 2: Feature Selection / Extraction ────────────────────────────
    # Dynamically compute the number of components/features from fraction
    n_selected = max(1, int(n_features * chromosome.feature_fraction))

    if chromosome.feature_selector == "PCA":
        # n_components must not exceed n_features
        n_components = min(n_selected, n_features)
        steps.append(("feature_selector", PCA(n_components=n_components, random_state=random_state)))
    elif chromosome.feature_selector == "SelectKBest":
        k = min(n_selected, n_features)
        score_func = f_classif if task_type == "classification" else f_regression
        steps.append(("feature_selector", SelectKBest(score_func=score_func, k=k)))
    # 'None' → no feature selection step

    # ── Step 3: ML Model ──────────────────────────────────────────────────
    estimator_map = (
        CLASSIFICATION_ESTIMATORS if task_type == "classification" else REGRESSION_ESTIMATORS
    )
    estimator_cls = estimator_map[chromosome.model_name]
    hp = dict(chromosome.hyperparameters)  # shallow copy

    # Inject random_state into models that support it
    _inject_random_state(estimator_cls, hp, random_state)

    # Suppress XGBoost verbose output
    if chromosome.model_name in ("XGBClassifier", "XGBRegressor"):
        hp.setdefault("verbosity", 0)
        hp.setdefault("use_label_encoder", False)

    estimator = estimator_cls(**hp)
    steps.append(("model", estimator))

    return Pipeline(steps)


def _inject_random_state(estimator_cls, hp: dict, random_state: Optional[int]):
    """Inject random_state into estimator hyperparameters if supported."""
    if random_state is None:
        return
    # Models with a random_state parameter
    rs_models = (
        LogisticRegression, RandomForestClassifier, RandomForestRegressor,
        MLPClassifier, MLPRegressor,
    )
    if XGBClassifier is not None:
        rs_models = rs_models + (XGBClassifier, XGBRegressor)

    if issubclass(estimator_cls, rs_models):
        hp["random_state"] = random_state


# ──────────────────────────────────────────────────────────────────────────────
# TIMEOUT MECHANISM (cross-platform using threading)
# ──────────────────────────────────────────────────────────────────────────────
class _TimeoutException(Exception):
    """Raised when a pipeline evaluation exceeds the time budget."""
    pass


def _evaluate_cv_with_timeout(pipeline, X, y, scoring, cv, timeout_seconds):
    """Run cross_val_score in a daemon thread with a timeout.

    Args:
        pipeline: The sklearn Pipeline to evaluate.
        X, y: Dataset.
        scoring: Scoring metric string.
        cv: Cross-validation splitter.
        timeout_seconds: Max seconds allowed.

    Returns:
        Mean CV score, or -inf if timed out or errored.
    """
    result = {"score": float("-inf"), "error": None}

    def _worker():
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, error_score="raise")
            result["score"] = float(np.mean(scores))
        except Exception as e:
            result["error"] = e
            result["score"] = float("-inf")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread still running → timeout
        return float("-inf"), "Timeout"
    if result["error"] is not None:
        return float("-inf"), str(result["error"])
    return result["score"], None


# ──────────────────────────────────────────────────────────────────────────────
# FITNESS EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_fitness(
    chromosome: Chromosome,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int = 5,
    timeout_seconds: int = 30,
    random_state: Optional[int] = None,
) -> float:
    """Evaluate the fitness of a chromosome via K-Fold Cross Validation.

    Args:
        chromosome: Candidate pipeline configuration.
        X: Feature matrix.
        y: Target vector.
        task_type: 'classification' or 'regression'.
        cv_folds: Number of CV folds (default 5).
        timeout_seconds: Max seconds per evaluation (default 30).
        random_state: Seed for CV splitting and model initialization.

    Returns:
        Fitness score (float). Returns -inf for invalid or timed-out pipelines.
    """
    try:
        n_features = X.shape[1]
        pipeline = build_pipeline(chromosome, task_type, n_features, random_state)

        # Scoring metric
        if task_type == "classification":
            scoring = "f1_macro"
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            scoring = "neg_root_mean_squared_error"
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Evaluate with timeout protection
        score, err_msg = _evaluate_cv_with_timeout(
            pipeline, X, y, scoring, cv, timeout_seconds
        )

        if err_msg == "Timeout":
            warnings.warn(
                f"⏱ Timeout ({timeout_seconds}s) for {chromosome.model_name}. "
                f"Penalizing with -inf fitness."
            )
        elif err_msg is not None:
            warnings.warn(f"⚠ Pipeline error for {chromosome.model_name}: {err_msg}")

        return score

    except Exception as e:
        # Catch-all: any remaining errors → penalize
        warnings.warn(f"⚠ Unexpected error for {chromosome.model_name}: {e}")
        return float("-inf")

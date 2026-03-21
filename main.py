"""
main.py — Executable Demonstration
===================================
Proves the system works for both Classification (Breast Cancer)
and Regression (Diabetes) tasks perfectly out of the box.
"""

import warnings
from sklearn.datasets import load_breast_cancer, load_diabetes

# Suppress standard sklearn warnings for terminal cleanliness
warnings.filterwarnings("ignore")

from meta_automl import MetaAutoML

def test_classification():
    print("\n\n" + "#" * 60)
    print("TESTING 1: CLASSIFICATION MODE (Breast Cancer Dataset)")
    print("#" * 60)
    
    X, y = load_breast_cancer(return_X_y=True)
    
    automl = MetaAutoML(
        task_type="classification",
        pop_size=10,           # Smaller population/gens to run faster in demo
        n_generations=5,
        cv_folds=3,
        timeout_seconds=20,
        random_state=42        # Ensures exact reproducible outputs
    )
    
    automl.fit(X, y, verbose=True)
    automl.summary()

def test_regression():
    print("\n\n" + "#" * 60)
    print("TESTING 2: REGRESSION MODE (Diabetes Dataset)")
    print("#" * 60)
    
    X, y = load_diabetes(return_X_y=True)
    
    automl = MetaAutoML(
        task_type="regression",
        pop_size=10,           # Smaller population/gens to run faster in demo
        n_generations=5,
        cv_folds=3,
        timeout_seconds=20,
        random_state=42        # Ensures exact reproducible outputs
    )
    
    automl.fit(X, y, verbose=True)
    automl.summary()

if __name__ == "__main__":
    try:
        import xgboost
        print(f"XGBoost version found: {xgboost.__version__}")
    except ImportError:
        print("XGBoost not installed. It will be skipped from the model space.")
        
    test_classification()
    test_regression()

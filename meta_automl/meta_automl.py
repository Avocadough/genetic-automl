"""
meta_automl.py — Top-Level API Interface
========================================
Main entry point for the Meta-AutoML library mimicking the scikit-learn API:
`model = MetaAutoML(...)`
`model.fit(X, y)`
`pipeline = model.get_best_pipeline()`
"""

import pprint
import numpy as np
from sklearn.pipeline import Pipeline

from meta_automl.ga_engine import GAEngine
from meta_automl.fitness import build_pipeline
from meta_automl.chromosome import Chromosome


class MetaAutoML:
    """Genetic Algorithm-based AutoML System."""

    def __init__(
        self,
        task_type: str = "classification",
        pop_size: int = 15,
        n_generations: int = 10,
        mutation_rate: float = 0.2,
        cv_folds: int = 5,
        timeout_seconds: int = 30,
        random_state: int = 42,
    ):
        """
        Args:
            task_type: 'classification' or 'regression'.
            pop_size: Number of pipelines per generation.
            n_generations: Number of generations to evolve.
            mutation_rate: Probability of mutating each gene [0.0, 1.0].
            cv_folds: Number of Cross-Validation folds for fitness evaluation.
            timeout_seconds: Max time limit per pipeline evaluation.
            random_state: Seed for total reproducibility across all stochastic operations.
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")
            
        self.task_type = task_type
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.cv_folds = cv_folds
        self.timeout_seconds = timeout_seconds
        self.random_state = random_state
        
        # State
        self.best_chromosome_: Chromosome = None
        self.best_pipeline_: Pipeline = None
        self.engine_: GAEngine = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Run the Genetic Algorithm to find the best ML pipeline for the provided data.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            verbose: If True, prints generation-level trace.
            
        Returns:
            self
        """
        X_np = np.asarray(X)
        y_np = np.asarray(y)
        
        if verbose:
            print("=" * 60)
            print(f"🚀 Starting MetaAutoML Search for {self.task_type.upper()}")
            print(f"Population: {self.pop_size} | Generations: {self.n_generations}")
            print(f"Random State: {self.random_state} | CV Folds: {self.cv_folds}")
            print("=" * 60)

        self.engine_ = GAEngine(
            task_type=self.task_type,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            mutation_rate=self.mutation_rate,
            cv_folds=self.cv_folds,
            timeout_seconds=self.timeout_seconds,
            random_state=self.random_state,
        )

        n_features = X_np.shape[1]
        
        # Run optimization
        self.best_chromosome_ = self.engine_.optimize(X_np, y_np, verbose=verbose)
        
        # Build scikit-learn Pipeline from best chromosome and physically fit it on full training data
        self.best_pipeline_ = build_pipeline(
            self.best_chromosome_, 
            self.task_type, 
            n_features, 
            self.random_state
        )
        
        if verbose:
            print("\nFitting Best Pipeline on entire dataset...")
            
        self.best_pipeline_.fit(X_np, y_np)
        self.is_fitted_ = True
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets using the optimized pipeline."""
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before predicting.")
        return self.best_pipeline_.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the optimized pipeline (classification only)."""
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before predicting.")
        if self.task_type != "classification":
            raise RuntimeError("predict_proba is only supported for classification tasks.")
        return self.best_pipeline_.predict_proba(np.asarray(X))

    def get_best_pipeline(self) -> Pipeline:
        """Return the fully trained best scikit-learn Pipeline."""
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before requesting the pipeline.")
        return self.best_pipeline_

    def summary(self):
        """Print a visually comprehensive summary of the discovered pipeline."""
        if not self.is_fitted_:
            print("Model has not been fitted yet.")
            return

        c = self.best_chromosome_
        print("\n" + "=" * 50)
        print("🏆  META-AUTOML RESULTS SUMMARY  🏆")
        print("=" * 50)
        
        metric = "F1-Macro" if self.task_type == "classification" else "Negative RMSE"
        
        print(f"Task Type          : {self.task_type.capitalize()}")
        print(f"Best CV Fitness    : {c.fitness:.4f} ({metric})")
        print(f"\n[ Pipeline Configuration ]")
        print(f"  • Scaler           : {c.scaler}")
        print(f"  • Feature Selector : {c.feature_selector}")
        print(f"  • Feature Fraction : {c.feature_fraction:.2%}")
        print(f"\n[ Estimator details ]")
        print(f"  • Model            : {c.model_name}")
        print(f"  • Hyperparameters  :")
        for param, val in c.hyperparameters.items():
            print(f"      - {param:<18}: {val}")
        print("=" * 50)

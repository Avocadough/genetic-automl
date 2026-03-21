"""
chromosome.py — Chromosome Representation
==========================================
Each chromosome encodes a full ML pipeline configuration:
  - Scaler gene          (str)
  - Feature selector gene (str)
  - Feature fraction gene (float in [0.3, 0.9])
  - Model name gene      (str)
  - Hyperparameters gene  (dict — coupled with model name)

Provides factory methods for random initialization with reproducible
random states propagated from the top-level seed.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from meta_automl.search_space import (
    SCALER_OPTIONS,
    FEATURE_SELECTOR_OPTIONS,
    FEATURE_FRACTION_RANGE,
    get_model_space,
)


class Chromosome:
    """Represents a single candidate ML pipeline configuration."""

    def __init__(
        self,
        scaler: str,
        feature_selector: str,
        feature_fraction: float,
        model_name: str,
        hyperparameters: dict,
    ):
        self.scaler = scaler
        self.feature_selector = feature_selector
        self.feature_fraction = feature_fraction
        self.model_name = model_name
        self.hyperparameters = copy.deepcopy(hyperparameters)
        # Fitness is set externally after evaluation
        self.fitness: float = float("-inf")

    # ── Factory: random chromosome ────────────────────────────────────────
    @classmethod
    def random(cls, task_type: str, rng: np.random.RandomState) -> Chromosome:
        """Create a randomly initialized chromosome.

        Args:
            task_type: 'classification' or 'regression'.
            rng: NumPy RandomState for reproducibility.

        Returns:
            A new Chromosome with randomly sampled genes.
        """
        model_space = get_model_space(task_type)

        scaler = rng.choice(SCALER_OPTIONS)
        feature_selector = rng.choice(FEATURE_SELECTOR_OPTIONS)
        feature_fraction = rng.uniform(*FEATURE_FRACTION_RANGE)

        # Randomly pick a model
        model_name = rng.choice(list(model_space.keys()))

        # Sample hyperparameters for the chosen model
        hyperparameters = _sample_hyperparameters(model_space[model_name], rng)

        return cls(
            scaler=scaler,
            feature_selector=feature_selector,
            feature_fraction=feature_fraction,
            model_name=model_name,
            hyperparameters=hyperparameters,
        )

    # ── Utility methods ───────────────────────────────────────────────────
    def copy(self) -> Chromosome:
        """Return a deep copy of this chromosome."""
        new = Chromosome(
            scaler=self.scaler,
            feature_selector=self.feature_selector,
            feature_fraction=self.feature_fraction,
            model_name=self.model_name,
            hyperparameters=self.hyperparameters,  # deepcopy in __init__
        )
        new.fitness = self.fitness
        return new

    def to_dict(self) -> dict:
        """Serialize chromosome to a dictionary."""
        return {
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "feature_fraction": round(self.feature_fraction, 4),
            "model_name": self.model_name,
            "hyperparameters": copy.deepcopy(self.hyperparameters),
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chromosome:
        """Deserialize a chromosome from a dictionary."""
        chrom = cls(
            scaler=data["scaler"],
            feature_selector=data["feature_selector"],
            feature_fraction=data["feature_fraction"],
            model_name=data["model_name"],
            hyperparameters=data["hyperparameters"],
        )
        chrom.fitness = data.get("fitness", float("-inf"))
        return chrom

    def __repr__(self) -> str:
        return (
            f"Chromosome(model={self.model_name}, scaler={self.scaler}, "
            f"selector={self.feature_selector}, frac={self.feature_fraction:.2f}, "
            f"fitness={self.fitness:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: sample hyperparameters from a model's grid
# ──────────────────────────────────────────────────────────────────────────────
def _sample_hyperparameters(hp_grid: dict, rng: np.random.RandomState) -> dict:
    """Randomly sample one value per hyperparameter from the grid.

    Args:
        hp_grid: Dict mapping HP names to lists of allowed values.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Dict with one sampled value per hyperparameter.
    """
    sampled = {}
    for param_name, choices in hp_grid.items():
        idx = rng.randint(0, len(choices))
        sampled[param_name] = choices[idx]
    return sampled

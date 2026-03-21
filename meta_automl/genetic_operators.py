"""
genetic_operators.py — GA Selection, Crossover, and Mutation
============================================================
Implements the three core genetic operators:
  - Tournament Selection (k=3)
  - Uniform Crossover with coupled model+HP genes
  - Random Mutation

All operators accept a NumPy RandomState for full reproducibility.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from meta_automl.chromosome import Chromosome, _sample_hyperparameters
from meta_automl.search_space import (
    SCALER_OPTIONS,
    FEATURE_SELECTOR_OPTIONS,
    FEATURE_FRACTION_RANGE,
    get_model_space,
)


# ──────────────────────────────────────────────────────────────────────────────
# TOURNAMENT SELECTION
# ──────────────────────────────────────────────────────────────────────────────
def tournament_selection(
    population: List[Chromosome],
    rng: np.random.RandomState,
    tournament_size: int = 3,
) -> Chromosome:
    """Select the best individual from a random tournament.

    Args:
        population: Current population of chromosomes.
        rng: NumPy RandomState for reproducibility.
        tournament_size: Number of individuals competing (default 3).

    Returns:
        A copy of the tournament winner.
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    competitors = [population[i] for i in indices]
    winner = max(competitors, key=lambda c: c.fitness)
    return winner.copy()


# ──────────────────────────────────────────────────────────────────────────────
# UNIFORM CROSSOVER  (model + hyperparameters are COUPLED)
# ──────────────────────────────────────────────────────────────────────────────
def uniform_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.RandomState,
) -> Tuple[Chromosome, Chromosome]:
    """Perform uniform crossover on two parents to produce two offspring.

    Each gene is swapped with 50% probability.
    **Model name and hyperparameters are always inherited together** to
    prevent parameter-mismatch errors.

    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Tuple of two offspring chromosomes.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Gene 1: Scaler
    if rng.random() < 0.5:
        child1.scaler, child2.scaler = child2.scaler, child1.scaler

    # Gene 2: Feature Selector
    if rng.random() < 0.5:
        child1.feature_selector, child2.feature_selector = (
            child2.feature_selector,
            child1.feature_selector,
        )

    # Gene 3: Feature Fraction
    if rng.random() < 0.5:
        child1.feature_fraction, child2.feature_fraction = (
            child2.feature_fraction,
            child1.feature_fraction,
        )

    # Gene 4+5: Model + Hyperparameters (COUPLED — always swap together)
    if rng.random() < 0.5:
        child1.model_name, child2.model_name = child2.model_name, child1.model_name
        child1.hyperparameters, child2.hyperparameters = (
            child2.hyperparameters,
            child1.hyperparameters,
        )

    return child1, child2


# ──────────────────────────────────────────────────────────────────────────────
# MUTATION
# ──────────────────────────────────────────────────────────────────────────────
def mutate(
    chromosome: Chromosome,
    task_type: str,
    rng: np.random.RandomState,
    mutation_rate: float = 0.2,
) -> Chromosome:
    """Mutate genes of a chromosome with a given probability.

    If the model gene mutates, its hyperparameters are re-sampled from
    the new model's search space (coupled mutation).

    Args:
        chromosome: The chromosome to mutate (modified in-place and returned).
        task_type: 'classification' or 'regression'.
        rng: NumPy RandomState for reproducibility.
        mutation_rate: Probability of mutating each gene (default 0.2).

    Returns:
        The mutated chromosome (same object, modified in place).
    """
    model_space = get_model_space(task_type)

    # Mutate scaler
    if rng.random() < mutation_rate:
        chromosome.scaler = rng.choice(SCALER_OPTIONS)

    # Mutate feature selector
    if rng.random() < mutation_rate:
        chromosome.feature_selector = rng.choice(FEATURE_SELECTOR_OPTIONS)

    # Mutate feature fraction
    if rng.random() < mutation_rate:
        chromosome.feature_fraction = rng.uniform(*FEATURE_FRACTION_RANGE)

    # Mutate model (+ re-sample hyperparameters for new model)
    if rng.random() < mutation_rate:
        new_model = rng.choice(list(model_space.keys()))
        chromosome.model_name = new_model
        chromosome.hyperparameters = _sample_hyperparameters(
            model_space[new_model], rng
        )
    else:
        # Mutate individual hyperparameters of the current model
        hp_grid = model_space[chromosome.model_name]
        for param_name, choices in hp_grid.items():
            if rng.random() < mutation_rate:
                idx = rng.randint(0, len(choices))
                chromosome.hyperparameters[param_name] = choices[idx]

    return chromosome

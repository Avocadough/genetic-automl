"""
ga_engine.py — Genetic Algorithm Master Loop
============================================
Orchestrates the evolutionary process:
  - Initializes population
  - Evaluates fitness using K-Fold CV
  - Explicit Elitism: unconditionally carries the absolute best individual forward
  - Applies selection, crossover, and mutation to generate offspring
"""

import sys
from typing import List, Tuple, Optional

import numpy as np

from meta_automl.chromosome import Chromosome
from meta_automl.fitness import evaluate_fitness
from meta_automl.genetic_operators import tournament_selection, uniform_crossover, mutate


class GAEngine:
    """Genetic Algorithm Engine for optimizing Auto-ML pipelines."""

    def __init__(
        self,
        task_type: str,
        pop_size: int = 20,
        n_generations: int = 10,
        mutation_rate: float = 0.2,
        cv_folds: int = 5,
        timeout_seconds: int = 30,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.cv_folds = cv_folds
        self.timeout_seconds = timeout_seconds
        
        # Core reproducibility anchor
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # State tracking
        self.population: List[Chromosome] = []
        self.best_history: List[float] = []

    def _initialize_population(self) -> None:
        """Create random initial population."""
        self.population = [
            Chromosome.random(self.task_type, self.rng) for _ in range(self.pop_size)
        ]

    def _evaluate_population(self, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate fitness for all individuals that don't have it yet."""
        for chrom in self.population:
            if chrom.fitness == float("-inf"):
                chrom.fitness = evaluate_fitness(
                    chrom,
                    X,
                    y,
                    self.task_type,
                    self.cv_folds,
                    self.timeout_seconds,
                    self.random_state, # Pass top-level seed down to estimators/CV
                )

    def optimize(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Chromosome:
        """Run the main GA optimization loop.

        Args:
            X: Feature matrix.
            y: Target vector.
            verbose: If True, prints generation-level progress.

        Returns:
            The best overall Chromosome discovered.
        """
        # 1. Initialize
        self._initialize_population()

        # 2. Evaluate Generation 0
        self._evaluate_population(X, y)
        
        best_overall = max(self.population, key=lambda c: c.fitness).copy()
        if verbose:
            print(f"Gen 0 | Best Fitness: {best_overall.fitness:.4f} | Model: {best_overall.model_name}")

        # 3. Evolution Loop
        for gen in range(1, self.n_generations + 1):
            new_population: List[Chromosome] = []

            # ── EXPLICIT ELITISM ──────────────────────────────────────────────
            # Carry over the best individual untouched
            current_best = max(self.population, key=lambda c: c.fitness)
            new_population.append(current_best.copy())

            # ── Crossover & Mutation ──────────────────────────────────────────
            while len(new_population) < self.pop_size:
                # Select parents
                p1 = tournament_selection(self.population, self.rng)
                p2 = tournament_selection(self.population, self.rng)

                # Re-select if they are identically the same object (rare)
                while id(p1) == id(p2):
                    p2 = tournament_selection(self.population, self.rng)

                # Crossover
                c1, c2 = uniform_crossover(p1, p2, self.rng)

                # Mutate (in-place)
                mutate(c1, self.task_type, self.rng, self.mutation_rate)
                mutate(c2, self.task_type, self.rng, self.mutation_rate)

                # Add to new pop (reset fitness since genes changed)
                c1.fitness = float("-inf")
                c2.fitness = float("-inf")
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            self.population = new_population

            # 4. Evaluate New Generation
            self._evaluate_population(X, y)

            # 5. Track Progress
            gen_best = max(self.population, key=lambda c: c.fitness)
            if gen_best.fitness > best_overall.fitness:
                best_overall = gen_best.copy()
            
            self.best_history.append(gen_best.fitness)

            if verbose:
                print(f"Gen {gen:02d} | Best Fitness: {gen_best.fitness:.4f} | "
                      f"Current Best Model: {gen_best.model_name}")

        if verbose:
            print("\n🏆 Evolution Complete!")
            print(f"Best Fitness Achieved: {best_overall.fitness:.4f}")

        return best_overall

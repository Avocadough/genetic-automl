# Meta-AutoML with a genetic algorithm

An experimental Python implementation of combined algorithm selection and hyperparameter search. It represents candidate scikit-learn pipelines as chromosomes, evaluates them with cross-validation, and uses genetic operators to evolve the population. Both classification and regression are supported.

This is a research and learning prototype, not a production AutoML service. It has no independent benchmark against established AutoML systems in this repository.

## How it works

1. Sample candidate preprocessing, feature selection, estimator and hyperparameter settings.
2. Score each candidate with stratified cross-validation and macro F1 for classification, or K-fold cross-validation and negative RMSE for regression.
3. Select, cross and mutate candidates; carry the current best candidate into the next generation.
4. Refit the selected pipeline on the supplied training data and expose `predict()` and `get_best_pipeline()`.

The search includes standard scalers, PCA or univariate feature selection, and several scikit-learn models. XGBoost is included when available.

## Quick start

```bash
git clone https://github.com/Avocadough/AutoML-with-Metaheuristic-Optimization.git
cd AutoML-with-Metaheuristic-Optimization
python -m pip install -r requirements.txt
```

The example below checks the API with a small Iris split. Zero evolved generations means only the initial candidates are evaluated; increase the population and generations for an actual search.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from meta_automl import MetaAutoML

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

search = MetaAutoML(
    task_type="classification",
    pop_size=2,
    n_generations=0,
    cv_folds=2,
    timeout_seconds=10,
    random_state=42,
)
search.fit(X_train, y_train, verbose=False)
predictions = search.predict(X_test)
print("Predictions:", len(predictions))
```

To run the smoke checks, install `pytest` and run `python -m pytest -q`. `main.py` contains larger classification and regression examples and can take substantially longer.

## Structure

| Path | Purpose |
| --- | --- |
| [`meta_automl/search_space.py`](meta_automl/search_space.py) | Candidate models and hyperparameter ranges. |
| [`meta_automl/chromosome.py`](meta_automl/chromosome.py) | Pipeline representation. |
| [`meta_automl/fitness.py`](meta_automl/fitness.py) | Pipeline construction and cross-validation fitness. |
| [`meta_automl/genetic_operators.py`](meta_automl/genetic_operators.py) | Selection, crossover and mutation. |
| [`meta_automl/ga_engine.py`](meta_automl/ga_engine.py) | Population and evolution loop. |
| [`meta_automl/meta_automl.py`](meta_automl/meta_automl.py) | Fit/predict interface. |
| [`tests/`](tests/) | Small API smoke checks. |

## Limits

- The search seeds several random operations, but identical results across dependency versions, hardware or timed-out runs are not guaranteed.
- The timeout stops **waiting** for a cross-validation thread; it does not terminate that thread. Repeated timeouts can continue consuming resources. It is not a hard execution or memory limit.
- For `n_generations > 0`, tournament selection requires at least 3 candidates. At exactly 3, every tournament contains the whole population and selects the same best candidate, limiting crossover diversity. Use at least 4 for exploration; repeated parent selection can still occur. The API does not currently validate all configuration errors.
- Cross-validation scores guide selection and are not an independent test score. Reserve a holdout suited to your data; random K-fold is unsuitable for many temporal or grouped datasets.
- `predict_proba()` depends on whether the selected estimator supports it. Regressors do not provide class probabilities. No model-quality or runtime guarantee is made.

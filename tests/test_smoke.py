import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from meta_automl import MetaAutoML


def test_prediction_before_fit_explains_the_problem():
    with pytest.raises(RuntimeError, match="fit"):
        MetaAutoML().predict([[1, 2]])


def test_small_classification_search_fits_and_predicts():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    search = MetaAutoML(
        task_type="classification", pop_size=2, n_generations=0,
        cv_folds=2, timeout_seconds=10, random_state=42,
    )
    search.fit(X_train, y_train, verbose=False)
    predictions = search.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1, 2})


def test_one_generation_produces_a_fitted_pipeline():
    X, y = load_iris(return_X_y=True)
    search = MetaAutoML(
        task_type="classification", pop_size=4, n_generations=1,
        cv_folds=2, timeout_seconds=10, random_state=42,
    )
    search.fit(X, y, verbose=False)
    assert len(search.engine_.best_history) == 1
    assert len(search.predict(X[:5])) == 5

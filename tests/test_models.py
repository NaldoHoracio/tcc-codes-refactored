import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from src.evaluation.model_comparison import compare_regressors
from src.analysis.descriptive import aggregate_question_importance
from src.models.regressors import (
    build_regressors,
    build_individual_experiment_regressors,
    extract_model_importance,
)


def test_build_regressors_contains_original_methods():
    models = build_regressors(random_forest_estimators=10)
    assert {"decision_tree", "random_forest", "lasso", "linear_regression"} == set(models)
    assert isinstance(models["decision_tree"], DecisionTreeRegressor)
    assert isinstance(models["lasso"], Lasso)
    assert models["decision_tree"].min_samples_split == 320
    assert models["random_forest"].n_estimators == 10
    assert models["lasso"].alpha == 0.005


def test_extract_importance_keeps_signed_coefficients():
    class LinearMock:
        coef_ = np.array([-0.5, 0.25])

    result = extract_model_importance(LinearMock(), ["a", "b"])
    assert result.loc[result["feature"] == "a", "importance"].iloc[0] == -0.5
    assert result.loc[result["feature"] == "a", "absolute_importance"].iloc[0] == 0.5


def test_compare_regressors_returns_metrics_and_importance():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=80)
    models = {
        "tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    }
    metrics, importances, fitted = compare_regressors(
        X, y, models=models, cv=4, test_size=0.25
    )
    assert metrics.loc[0, "model"] == "tree"
    assert {"test_r2", "test_mae", "cv_r2_mean"}.issubset(metrics.columns)
    assert set(importances["tree"]["feature"]) == {"a", "b", "c"}
    assert "tree" in fitted



def test_aggregate_question_importance_groups_one_hot_columns():
    data = pd.DataFrame({
        "feature": ["QE_I01_A", "QE_I01_B", "QE_I02_C", "other_feature"],
        "importance": [0.1, 0.2, -0.3, 0.4],
        "absolute_importance": [0.1, 0.2, 0.3, 0.4],
    })
    result = aggregate_question_importance(data)
    assert result["question"].tolist() == ["QE_I01", "QE_I02"]
    assert np.allclose(result["importance"], [0.3, -0.3])
    assert np.allclose(result["absolute_importance"], [0.3, 0.3])



def test_individual_experiment_hyperparameters_match_original_scripts():
    models = build_individual_experiment_regressors(random_forest_estimators=25)
    assert models["decision_tree"].random_state == 0
    assert models["random_forest"].n_estimators == 25
    assert models["random_forest"].random_state == 0
    assert models["lasso"].alpha == 0.1
    assert models["lasso"].positive is True

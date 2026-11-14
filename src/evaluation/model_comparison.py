"""Treinamento, validação cruzada e comparação dos modelos de regressão."""
from __future__ import annotations

import time
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.evaluation.metrics import regression_metrics
from src.models.regressors import build_regressors, extract_model_importance


def compare_regressors(
    features: pd.DataFrame | np.ndarray,
    labels,
    feature_names: list[str] | None = None,
    models: Mapping[str, object] | None = None,
    cv: int = 10,
    test_size: float = 0.33,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    """Compara os modelos seguindo o desenho do script compare_methods original.

    O conjunto é dividido com test_size=0.33 e random_state=42. A validação
    cruzada KFold (sem embaralhamento) é executada somente no conjunto de
    treinamento, usando R², como no código original.

    Retorna:
        metrics: uma linha por modelo, com métricas de CV e teste.
        importances: importância/coefs médios entre os estimadores dos folds.
        fitted_models: estimadores ajustados no conjunto de treinamento.
    """
    X = features
    y = np.asarray(labels, dtype=float)
    if len(X) != len(y):
        raise ValueError("Features e labels precisam ter o mesmo número de linhas.")
    if len(y) < cv:
        raise ValueError(f"São necessárias pelo menos {cv} amostras para {cv}-fold CV.")
    if feature_names is None:
        feature_names = (
            list(features.columns)
            if isinstance(features, pd.DataFrame)
            else [f"feature_{i}" for i in range(np.asarray(features).shape[1])]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if len(y_train) < cv:
        raise ValueError(
            f"O conjunto de treino tem {len(y_train)} amostras, insuficientes para "
            f"validação cruzada com {cv} folds."
        )

    estimators = dict(models or build_regressors(random_state=random_state))
    splitter = KFold(n_splits=cv, shuffle=False)
    rows: list[dict[str, float | str]] = []
    all_importances: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, object] = {}

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }

    for name, estimator in estimators.items():
        started = time.perf_counter()
        cv_result = cross_validate(
            clone(estimator),
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            return_estimator=True,
            n_jobs=n_jobs,
        )
        elapsed = time.perf_counter() - started

        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        test_metrics = regression_metrics(y_test, predictions)

        cv_r2 = cv_result["test_r2"]
        cv_mae = -cv_result["test_mae"]
        cv_mse = -cv_result["test_mse"]

        rows.append({
            "model": name,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2, ddof=1)) if len(cv_r2) > 1 else 0.0,
            "cv_mae_mean": float(np.mean(cv_mae)),
            "cv_mse_mean": float(np.mean(cv_mse)),
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "legacy_100_minus_mae": test_metrics["legacy_100_minus_mae"],
            "cv_seconds": float(elapsed),
        })

        fold_tables = [
            extract_model_importance(fold_estimator, feature_names)
            for fold_estimator in cv_result["estimator"]
        ]
        # Mantém o tipo de importância e calcula a média fold a fold.
        stacked = pd.concat(
            [table.set_index("feature")["importance"].rename(i)
             for i, table in enumerate(fold_tables)],
            axis=1,
        ).fillna(0.0)
        mean_values = stacked.mean(axis=1)
        importance_type = fold_tables[0]["importance_type"].iloc[0]
        all_importances[name] = pd.DataFrame({
            "feature": mean_values.index,
            "importance": mean_values.values,
            "absolute_importance": np.abs(mean_values.values),
            "importance_type": importance_type,
        }).sort_values("absolute_importance", ascending=False, ignore_index=True)

        fitted_models[name] = fitted

    metrics = pd.DataFrame(rows).sort_values(
        "cv_r2_mean", ascending=False, ignore_index=True
    )
    return metrics, all_importances, fitted_models

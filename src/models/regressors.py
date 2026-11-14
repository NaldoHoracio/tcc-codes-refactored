"""Fábricas dos modelos de regressão usados no TCC.

Os hiperparâmetros dos três modelos centrais reproduzem a configuração
registrada em compare_methods/compare_methods_al.py e
compare_methods/compare_methods_br.py do repositório original.
A regressão linear simples também é incluída porque aparece nos experimentos
separados da versão original.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor


def build_regressors(
    random_state: int = 42,
    random_forest_estimators: int = 1000,
) -> dict[str, object]:
    """Retorna os regressores com os parâmetros documentados no código original.

    Args:
        random_state: semente usada nos estimadores estocásticos.
        random_forest_estimators: número de árvores da Random Forest.
            O valor padrão (1000) corresponde ao experimento de comparação.
            Pode ser reduzido em testes rápidos.
    """
    return {
        "decision_tree": DecisionTreeRegressor(
            min_samples_split=320,
            min_samples_leaf=200,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=random_forest_estimators,
            min_samples_split=40,
            min_samples_leaf=20,
            random_state=random_state,
        ),
        "lasso": Lasso(
            alpha=0.005,
            positive=True,
            random_state=random_state,
            max_iter=10000,
        ),
        "linear_regression": LinearRegression(),
    }


def extract_model_importance(model: object, feature_names: list[str]):
    """Extrai importância de árvores ou coeficientes de modelos lineares.

    O campo ``importance`` mantém o sinal dos coeficientes lineares, como
    acontecia no código original. ``absolute_importance`` facilita ordenar
    a magnitude sem perder a informação de sinal.
    """
    import numpy as np
    import pandas as pd

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float).reshape(-1)
        kind = "coefficient"
    else:
        raise TypeError(
            f"O estimador {type(model).__name__} não expõe importâncias nem coeficientes."
        )

    if len(values) != len(feature_names):
        raise ValueError(
            "A quantidade de importâncias não corresponde à quantidade de atributos."
        )

    return pd.DataFrame({
        "feature": feature_names,
        "importance": values,
        "absolute_importance": np.abs(values),
        "importance_type": kind,
    }).sort_values("absolute_importance", ascending=False, ignore_index=True)



def build_individual_experiment_regressors(
    random_state: int = 0,
    random_forest_estimators: int = 1000,
) -> dict[str, object]:
    """Modelos com parâmetros dos experimentos individuais antigos.

    As rotinas separadas usavam Decision Tree padrão, Random Forest com
    1000 árvores e Lasso com alpha=0.1 e positive=True. Esta fábrica mantém
    essa configuração distinta da comparação consolidada (alpha=0.005).
    """
    return {
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=random_forest_estimators,
            random_state=random_state,
        ),
        "lasso": Lasso(
            alpha=0.1,
            positive=True,
            random_state=random_state,
            max_iter=10000,
        ),
        "linear_regression": LinearRegression(),
    }

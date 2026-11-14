"""Métricas de avaliação para os problemas de regressão do TCC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula R², MAE, MSE e RMSE."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        # Métrica histórica do código antigo; não é acurácia estatística.
        "legacy_100_minus_mae": float(100 - mae),
    }

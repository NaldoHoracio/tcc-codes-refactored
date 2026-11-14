"""Executa a comparação dos modelos sobre os dados pré-processados.

Uso:
    python -m src.run_experiments
"""
from __future__ import annotations

import logging

import pandas as pd

from src.analysis.descriptive import aggregate_question_importance, describe_target
from src.analysis.plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_target_distribution,
)
from src.config import PROCESSED_DATA_DIR
from src.evaluation.model_comparison import compare_regressors

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"
    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            "Dados pré-processados não encontrados. Execute primeiro: python -m src.main"
        )

    features = pd.read_csv(features_path, low_memory=False)
    labels = pd.read_csv(labels_path)["NT_GER"]

    metrics, importances, _ = compare_regressors(
        features=features,
        labels=labels,
        feature_names=features.columns.tolist(),
        cv=10,
        test_size=0.33,
        random_state=42,
    )

    results_dir = PROCESSED_DATA_DIR / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False)

    for model_name, table in importances.items():
        table.to_csv(results_dir / f"importance_{model_name}.csv", index=False)
        question_table = aggregate_question_importance(table)
        question_table.to_csv(results_dir / f"question_importance_{model_name}.csv", index=False)
        plot_feature_importance(
            table,
            results_dir / f"importance_{model_name}.png",
            top_n=20,
            title=f"Importância das variáveis — {model_name}",
        )

    plot_model_comparison(metrics, results_dir / "model_comparison_r2.png")
    plot_target_distribution(labels, results_dir / "target_distribution.png")
    describe_target(labels).to_csv(results_dir / "target_statistics.csv", header=["value"])

    logger.info("Comparação concluída. Resultados em %s", results_dir)
    logger.info("\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()

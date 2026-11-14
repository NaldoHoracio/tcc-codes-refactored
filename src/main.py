"""Ponto de entrada para integração e pré-processamento dos dados selecionados."""
from __future__ import annotations

import logging

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.processing_data.ingestion import load_selected_datasets
from src.processing_data.preprocessing import preprocess_enade

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Executa a integração e o pré-processamento dos conjuntos selecionados."""
    datasets = load_selected_datasets()
    features, labels, feature_names, dataset = preprocess_enade(datasets)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features, columns=feature_names).to_csv(
        PROCESSED_DATA_DIR / "features.csv", index=False
    )
    pd.Series(labels, name="NT_GER").to_csv(
        PROCESSED_DATA_DIR / "labels.csv", index=False
    )
    dataset.to_csv(PROCESSED_DATA_DIR / "dataset_preprocessed.csv", index=False)

    logger.info("Pré-processamento concluído.")
    logger.info(
        "Amostras: %s | Features após encoding: %s",
        features.shape[0],
        features.shape[1],
    )
    logger.info("Arquivos salvos em: %s", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()

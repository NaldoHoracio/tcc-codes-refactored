"""Leitura dos conjuntos selecionados do ENADE."""
from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from src.config import AVAILABLE_YEARS, SELECTED_DATA_DIR

logger = logging.getLogger(__name__)


def load_selected_datasets(
    data_dir: str | Path = SELECTED_DATA_DIR,
    years: tuple[int, ...] = AVAILABLE_YEARS,
    prefix: str = "BR",
) -> list[pd.DataFrame]:
    """Carrega os CSVs selecionados para os anos configurados.

    A função não exige que os anos sejam consecutivos; carrega exatamente
    os anos informados em ``years``.
    """
    directory = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for year in years:
        path = directory / f"{prefix}_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Não foi encontrado o conjunto selecionado de {year}: {path}"
            )
        logger.info("Carregando ENADE %s: %s", year, path)
        frames.append(pd.read_csv(path, low_memory=False))
    return frames

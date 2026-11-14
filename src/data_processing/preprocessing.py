"""Pré-processamento dos conjuntos selecionados do ENADE (Alagoas)."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import QUESTION_COLUMNS, TARGET_COLUMN


def _select_relevant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Mantém o intervalo original NT_GER..QE_I26 e remove blocos intermediários."""
    if TARGET_COLUMN not in frame.columns or "QE_I26" not in frame.columns:
        raise KeyError(
            f"Cada conjunto precisa conter as colunas '{TARGET_COLUMN}' e 'QE_I26'."
        )

    columns = list(frame.columns)
    start, end = columns.index(TARGET_COLUMN), columns.index("QE_I26")
    selected = frame.iloc[:, start : end + 1].copy()

    # Mantém a regra da implementação original: remover CO_RS_I1..CO_RS_I9
    # e NT_FG..NT_CE_D3, quando esses intervalos estiverem presentes.
    for first, last in (("CO_RS_I1", "CO_RS_I9"), ("NT_FG", "NT_CE_D3")):
        if first in columns and last in columns:
            first_idx, last_idx = columns.index(first), columns.index(last)
            drop_columns = columns[first_idx : last_idx + 1]
            selected.drop(columns=[c for c in drop_columns if c in selected], inplace=True)

    return selected


def preprocess_enade(dataframes: Iterable[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Integra anos, trata NT_GER e aplica one-hot encoding às QE_I01..QE_I26.

    Retorna (features, labels, nomes_das_features, dataset_antes_do_encoding).
    A imputação da variável-alvo pela média e as variáveis do questionário
    codificadas por one-hot mantêm a abordagem presente no código original.
    """
    frames = list(dataframes)
    if not frames:
        raise ValueError("Informe ao menos um DataFrame para processamento.")

    selected_frames = [_select_relevant_columns(frame) for frame in frames]
    dataset = pd.concat(selected_frames, ignore_index=True, sort=False)

    missing_questions = [column for column in QUESTION_COLUMNS if column not in dataset.columns]
    if missing_questions:
        raise KeyError(f"Colunas do questionário ausentes: {missing_questions}")

    # Trata números armazenados com vírgula decimal e converte valores inválidos em NaN.
    target = dataset[TARGET_COLUMN].astype("string").str.replace(",", ".", regex=False)
    dataset[TARGET_COLUMN] = pd.to_numeric(target, errors="coerce")

    target_mean = round(float(dataset[TARGET_COLUMN].mean()), 2)
    if pd.isna(target_mean):
        raise ValueError(f"Não há valores numéricos válidos em '{TARGET_COLUMN}'.")
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].fillna(target_mean)

    labels = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    features_frame = dataset.drop(columns=[TARGET_COLUMN])
    encoded_features = pd.get_dummies(
        features_frame,
        columns=list(QUESTION_COLUMNS),
        dtype=np.uint8,
    )
    feature_names = encoded_features.columns.tolist()
    features = encoded_features.to_numpy()

    return features, labels, feature_names, dataset


# Nome legado para facilitar a migração de códigos que importavam a função antiga.
def processing_set_al(data_al: Iterable[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Compatibilidade com o nome utilizado na versão anterior do projeto."""
    return preprocess_enade(data_al)


# Compatibilidade com chamadas da versão intermediária do repositório.
def preprocess_enade_al(dataframes):
    return preprocess_enade(dataframes)

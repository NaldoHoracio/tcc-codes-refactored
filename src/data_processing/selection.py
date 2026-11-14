"""Seleção dos microdados do ENADE e exportação dos conjuntos selecionados."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def select_enade_microdata(
    input_paths: Mapping[int, str | Path],
    output_dir: str | Path,
    uf_to_exclude: int = 27,
    samples: Mapping[int, int] | None = None,
    separator: str = ";",
    output_prefix: str = "BR",
    random_state: int = 42,
) -> list[Path]:
    """Seleciona microdados por ano, preservando a lógica do processamento original.

    Para cada arquivo: lê os dados, exclui a UF indicada, remove linhas com
    valores ausentes, aplica amostragem (quando configurada) e salva um CSV.
    Retorna os caminhos dos arquivos gerados.
    """
    if not input_paths:
        raise ValueError("Informe ao menos um arquivo de entrada.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for year, input_path in sorted(input_paths.items()):
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Arquivo do ENADE {year} não encontrado: {source}")

        frame = pd.read_csv(source, sep=separator, low_memory=False)
        if "QE_I16" not in frame.columns:
            raise KeyError(f"A coluna 'QE_I16' não existe no arquivo de {year}.")

        frame = frame.loc[frame["QE_I16"] != uf_to_exclude].dropna(axis=0)

        if samples is not None and year in samples:
            sample_size = samples[year]
            if sample_size < 0:
                raise ValueError(f"A amostra de {year} não pode ser negativa.")
            if sample_size > len(frame):
                raise ValueError(
                    f"A amostra solicitada para {year} ({sample_size}) excede "
                    f"o total disponível após os filtros ({len(frame)})."
                )
            frame = frame.sample(n=sample_size, random_state=random_state)

        output_path = destination / f"{output_prefix}_{year}.csv"
        frame.to_csv(output_path, index=False)
        generated_files.append(output_path)

    return generated_files

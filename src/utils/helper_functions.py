"""Funções auxiliares genéricas do projeto."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def seconds_transform(seconds_time: float) -> str:
    """Converte segundos em uma representação legível e a retorna como texto."""
    if seconds_time < 0:
        raise ValueError("O tempo não pode ser negativo.")
    hours, remainder = divmod(float(seconds_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.2f} s"


def version_file(
    name_file: str | Path,
    fields: Sequence[str],
    rows_version: Mapping[str, object],
) -> int:
    """Cria ou acrescenta uma versão a um CSV e retorna o número gerado."""
    path = Path(name_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    if "Version" not in fieldnames:
        raise ValueError("A lista de campos precisa conter 'Version'.")

    version = 1
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            previous_rows = list(reader)
        if previous_rows:
            try:
                version = int(previous_rows[-1]["Version"]) + 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("O último registro não possui uma versão válida.") from exc

    row = dict(rows_version)
    row["Version"] = version
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return version

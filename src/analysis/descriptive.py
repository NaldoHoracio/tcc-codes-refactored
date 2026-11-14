"""Estatísticas descritivas para os dados e para a variável-alvo."""
from __future__ import annotations

import pandas as pd


def describe_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas descritivas de todas as colunas."""
    return dataset.describe(include="all").transpose()


def describe_target(labels, target_name: str = "NT_GER") -> pd.Series:
    """Calcula estatísticas básicas da variável-alvo."""
    values = pd.Series(labels, name=target_name).dropna()
    if values.empty:
        raise ValueError("A variável-alvo não contém valores válidos.")
    return pd.Series({
        "count": int(values.count()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "variance": float(values.var()),
        "standard_deviation": float(values.std()),
    }, name=target_name)



def aggregate_question_importance(importances: pd.DataFrame) -> pd.DataFrame:
    """Agrupa importâncias das categorias one-hot por questão QE_I01..QE_I26.

    A implementação original separava manualmente os vetores de importância
    em blocos correspondentes às perguntas. Aqui o agrupamento é feito pelo
    prefixo dos nomes das colunas, evitando índices fixos que podem mudar
    quando uma categoria não aparece na amostra.
    """
    required = {"feature", "importance", "absolute_importance"}
    missing = required.difference(importances.columns)
    if missing:
        raise KeyError(f"Colunas ausentes na tabela de importâncias: {sorted(missing)}")

    data = importances.copy()
    data["question"] = data["feature"].astype(str).str.extract(r"^(QE_I\d{2})_", expand=False)
    data = data.dropna(subset=["question"])
    if data.empty:
        return pd.DataFrame(columns=["question", "importance", "absolute_importance"])

    grouped = data.groupby("question", as_index=False).agg(
        importance=("importance", "sum"),
        absolute_importance=("absolute_importance", "sum"),
    )
    return grouped.sort_values("question", ignore_index=True)

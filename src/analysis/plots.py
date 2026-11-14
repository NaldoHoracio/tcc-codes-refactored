"""Gráficos para distribuição da variável-alvo e comparação dos modelos."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(labels, output_path: str | Path, title: str = "Distribuição de NT_GER") -> Path:
    """Salva histograma da variável-alvo."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(labels, bins=30, edgecolor="black")
    ax.set(title=title, xlabel="Nota ENADE (NT_GER)", ylabel="Frequência")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_model_comparison(
    metrics: pd.DataFrame,
    output_path: str | Path,
    metric: str = "cv_r2_mean",
) -> Path:
    """Salva gráfico de comparação dos modelos por uma métrica."""
    if metric not in metrics.columns:
        raise KeyError(f"Métrica inexistente: {metric}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["model"], ordered[metric])
    ax.set(title=f"Comparação dos modelos — {metric}", xlabel="Modelo", ylabel=metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_feature_importance(
    importances: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
    title: str = "Importância das variáveis",
) -> Path:
    """Salva gráfico das top_n variáveis por magnitude da importância."""
    if top_n < 1:
        raise ValueError("top_n precisa ser maior que zero.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = importances.nlargest(top_n, "absolute_importance").sort_values(
        "absolute_importance"
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(data))))
    ax.barh(data["feature"], data["absolute_importance"])
    ax.set(title=title, xlabel="Magnitude da importância", ylabel="Variável")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output

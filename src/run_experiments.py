
"""Execução dos experimentos de aprendizagem de máquina por região."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALAGOAS_IMAGES_DIR,
    BRASIL_IMAGES_DIR,
    EXPERIMENTS_IMAGES_DIR,
    EXPERIMENTS_TABLES_DIR,
    RESULTS_STATISTICS_DIR,
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.analysis.descriptive import (
    describe_dataset,
    describe_target,
    aggregate_question_importance,
)

from src.analysis.plots import (
    plot_target_distribution,
    plot_model_comparison,
    plot_feature_importance,
    plot_question_distributions,
)

from src.evaluation.model_comparison import compare_regressors


def _to_dataframe(data, columns=None):
    """Converte os dados para DataFrame quando necessário."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, pd.Series):
        return data.to_frame()

    return pd.DataFrame(data, columns=columns)


def _normalize_metrics(metrics):
    """Garante que o nome dos modelos esteja em uma coluna."""
    metrics = metrics.copy()

    if "model" not in metrics.columns:
        metrics.index.name = "model"
        metrics = metrics.reset_index()

    return metrics


def _safe_filename(name):
    """Normaliza nomes para utilização em arquivos."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def run_region_experiment(
    region,
    features,
    labels,
    dataset,
    feature_names=None,
    cv=CV_FOLDS,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Executa treinamento, avaliação e geração de resultados para uma região.

    Parameters
    ----------
    region : str
        Nome da região: 'alagoas' ou 'brasil'.

    features : DataFrame ou array
        Variáveis independentes.

    labels : DataFrame, Series ou array
        Variável-alvo NT_GER.

    dataset : DataFrame
        Dataset após o pré-processamento, antes da codificação das
        variáveis categóricas.

    feature_names : list, optional
        Nomes das variáveis utilizadas no treinamento.

    Returns
    -------
    dict
        Dicionário contendo métricas, importâncias e modelos treinados.
    """
    region = _safe_filename(region)

    if region not in ("alagoas", "brasil"):
        raise ValueError(
            "A região deve ser 'alagoas' ou 'brasil'."
        )

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENTOS: {region.upper()}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 1. Preparação dos diretórios
    # ---------------------------------------------------------

    if region == "alagoas":
        region_images_dir = ALAGOAS_IMAGES_DIR
    else:
        region_images_dir = BRASIL_IMAGES_DIR

    for directory in (
        region_images_dir,
        EXPERIMENTS_IMAGES_DIR,
        EXPERIMENTS_TABLES_DIR,
        RESULTS_STATISTICS_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Conversão e validação dos dados
    # ---------------------------------------------------------

    features = _to_dataframe(features, columns=feature_names)
    labels = _to_dataframe(labels)
    dataset = _to_dataframe(dataset)

    if features.empty or labels.empty:
        raise ValueError(
            f"Os dados de treinamento de {region} estão vazios."
        )

    if len(features) != len(labels):
        raise ValueError(
            f"A quantidade de amostras de features e labels "
            f"não coincide para {region}."
        )

    if feature_names is None:
        feature_names = list(features.columns)

    # ---------------------------------------------------------
    # 3. Salvamento dos dados processados
    # ---------------------------------------------------------

    features.to_csv(
        EXPERIMENTS_TABLES_DIR / f"features_{region}.csv",
        index=False,
    )

    labels.to_csv(
        EXPERIMENTS_TABLES_DIR / f"labels_{region}.csv",
        index=False,
    )

    dataset.to_csv(
        EXPERIMENTS_TABLES_DIR / f"dataset_preprocessed_{region}.csv",
        index=False,
    )

    print(f"Dados preparados: {region}")

    # ---------------------------------------------------------
    # 4. Estatísticas descritivas
    # ---------------------------------------------------------

    descriptive_statistics = describe_dataset(dataset)

    descriptive_statistics.to_csv(
        RESULTS_STATISTICS_DIR / f"descriptive_statistics_{region}.csv"
    )

    # target_statistics = describe_target(labels)
    target_statistics = describe_target(labels.iloc[:, 0])

    target_statistics.to_csv(
        RESULTS_STATISTICS_DIR / f"target_statistics_{region}.csv"
    )

    print("Estatísticas descritivas geradas.")

    # ---------------------------------------------------------
    # 5. Gráficos de distribuição
    # ---------------------------------------------------------

    plot_target_distribution(
        labels,
        region_images_dir / f"target_distribution_{region}.png",
        title=f"Distribuição de NT_GER - {region.capitalize()}",
    )

    plot_question_distributions(
        dataset,
        region_images_dir / f"question_distributions_{region}.png",
        title=(
            f"Distribuição das questões do questionário - "
            f"{region.capitalize()}"
        ),
    )

    print("Gráficos de distribuição gerados.")

    # ---------------------------------------------------------
    # 6. Treinamento e avaliação dos modelos
    # ---------------------------------------------------------

    print("Iniciando treinamento e avaliação dos modelos...")

    metrics, importances, fitted_models = compare_regressors(
        features=features,
        labels=labels,
        feature_names=feature_names,
        cv=cv,
        test_size=test_size,
        random_state=random_state,
    )

    metrics = _normalize_metrics(metrics)

    metrics.to_csv(
        EXPERIMENTS_TABLES_DIR / f"model_metrics_{region}.csv",
        index=False,
    )

    print("Treinamento e avaliação concluídos.")

    # ---------------------------------------------------------
    # 7. Gráfico de comparação dos modelos
    # ---------------------------------------------------------

    plot_model_comparison(
        metrics,
        EXPERIMENTS_IMAGES_DIR / f"model_comparison_{region}.png",
        metric="cv_r2_mean",
        title=f"Comparação dos modelos - {region.capitalize()}",
    )

    # ---------------------------------------------------------
    # 8. Importância das variáveis
    # ---------------------------------------------------------

    for model_name, importance_data in importances.items():

        model_filename = _safe_filename(model_name)

        if importance_data is None or importance_data.empty:
            continue

        importance_data.to_csv(
            EXPERIMENTS_TABLES_DIR
            / f"importance_{region}_{model_filename}.csv",
            index=False,
        )

        plot_feature_importance(
            importance_data,
            EXPERIMENTS_IMAGES_DIR
            / f"importance_{region}_{model_filename}.png",
            top_n=20,
            title=(
                f"Importância das variáveis - "
                f"{model_name} - {region.capitalize()}"
            ),
        )

    print("Importância das variáveis calculada.")

    # ---------------------------------------------------------
    # 9. Importância agregada das questões
    # ---------------------------------------------------------

    for model_name, importance_data in importances.items():

        if importance_data is None or importance_data.empty:
            continue

        question_importance = aggregate_question_importance(
            importance_data
        )

        if question_importance.empty:
            continue

        model_filename = _safe_filename(model_name)

        question_importance.to_csv(
            EXPERIMENTS_TABLES_DIR
            / f"question_importance_{region}_{model_filename}.csv",
            index=False,
        )

    print("Importância agregada das questões calculada.")

    # ---------------------------------------------------------
    # 10. Resumo dos resultados
    # ---------------------------------------------------------

    print(f"\nResultados de {region.upper()}:")

    columns_to_show = [
        column
        for column in (
            "model",
            "cv_r2_mean",
            "cv_r2_std",
            "cv_mae_mean",
            "cv_rmse_mean",
        )
        if column in metrics.columns
    ]

    if columns_to_show:
        print(metrics[columns_to_show].to_string(index=False))
    else:
        print(metrics.to_string(index=False))

    return {
        "region": region,
        "metrics": metrics,
        "importances": importances,
        "models": fitted_models,
        "features": features,
        "labels": labels,
        "dataset": dataset,
    }


def run_regional_comparison(results_al, results_br):
    """
    Gera gráficos comparativos entre os resultados de Alagoas e Brasil.
    """
    print("\nGerando comparação entre Alagoas e Brasil...")

    metrics_al = results_al["metrics"]
    metrics_br = results_br["metrics"]

    plot_model_comparison(
        metrics_al,
        EXPERIMENTS_IMAGES_DIR / "model_comparison_alagoas.png",
        metric="cv_r2_mean",
        title="Desempenho dos modelos - Alagoas",
    )

    plot_model_comparison(
        metrics_br,
        EXPERIMENTS_IMAGES_DIR / "model_comparison_brasil.png",
        metric="cv_r2_mean",
        title="Desempenho dos modelos - Brasil",
    )

    # Comparação da distribuição da variável-alvo.
    plot_target_distribution(
        results_al["labels"],
        EXPERIMENTS_IMAGES_DIR / "target_distribution_alagoas.png",
        title="Distribuição de NT_GER - Alagoas",
    )

    plot_target_distribution(
        results_br["labels"],
        EXPERIMENTS_IMAGES_DIR / "target_distribution_brasil.png",
        title="Distribuição de NT_GER - Brasil",
    )

    from src.analysis.plots import (
        plot_target_comparison,
        plot_regional_model_comparison,
    )

    plot_target_comparison(
        results_al["labels"],
        results_br["labels"],
        EXPERIMENTS_IMAGES_DIR / "target_comparison_alagoas_brasil.png",
    )

    plot_regional_model_comparison(
        metrics_al,
        metrics_br,
        EXPERIMENTS_IMAGES_DIR / "regional_model_comparison.png",
        metric="cv_r2_mean",
    )

    print("Comparações regionais concluídas.")

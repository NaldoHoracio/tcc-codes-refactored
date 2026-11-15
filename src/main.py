"""Ponto de entrada principal do projeto ENADE.

Pipeline:
1. Carregamento dos dados selecionados de Alagoas e Brasil.
2. Pré-processamento independente por região.
3. Geração das distribuições gaussianas das variáveis socioeconômicas.
4. Treinamento e avaliação dos modelos.
5. Geração de estatísticas, gráficos e comparação regional.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    SELECTED_DATA_DIR,
    AVAILABLE_AL_YEARS,
    AVAILABLE_BR_YEARS,
    RANDOM_STATE,
    CV_FOLDS,
    TEST_SIZE,
    RESULTS_DIR,
    RESULTS_TABLES_DIR,
    EXPERIMENTS_TABLES_DIR,
    RESULTS_IMAGES_DIR,
    ALAGOAS_IMAGES_DIR,
    BRASIL_IMAGES_DIR,
    EXPERIMENTS_IMAGES_DIR,
    RESULTS_STATISTICS_DIR,
)

from src.data_processing.ingestion import load_selected_datasets
from src.data_processing.preprocessing import preprocess_enade

from src.analysis.plots import plot_all_socioeconomic_distributions

from src.run_experiments import (
    run_region_experiment,
    run_regional_comparison,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_output_directories() -> None:
    """Cria a estrutura de diretórios destinada aos resultados."""

    directories = [
        RESULTS_DIR,
        RESULTS_TABLES_DIR,
        EXPERIMENTS_TABLES_DIR,
        RESULTS_IMAGES_DIR,
        ALAGOAS_IMAGES_DIR,
        BRASIL_IMAGES_DIR,
        EXPERIMENTS_IMAGES_DIR,
        RESULTS_STATISTICS_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("Diretórios de resultados verificados.")


def load_region_data(
    region: str,
    years: tuple[int, ...],
):
    """
    Carrega os arquivos selecionados de uma região.

    Os arquivos devem seguir o padrão:
        AL_2014.csv, AL_2015.csv, ...
        BR_2014.csv, BR_2015.csv, ...
    """

    logger.info(
        "Carregando dados de %s para os anos: %s",
        region,
        years,
    )

    datasets = load_selected_datasets(
        data_dir=SELECTED_DATA_DIR,
        years=years,
        prefix=region,
    )

    total_samples = sum(len(frame) for frame in datasets)

    logger.info(
        "%s: %d arquivos carregados, totalizando %d registros.",
        region,
        len(datasets),
        total_samples,
    )

    return datasets


def preprocess_region(
    region: str,
    datasets: list,
):
    """Executa o pré-processamento de uma região independentemente."""

    logger.info("Iniciando pré-processamento de %s.", region)

    features, labels, feature_names, dataset = preprocess_enade(
        datasets
    )

    logger.info(
        "Pré-processamento de %s concluído.",
        region,
    )

    logger.info(
        "Amostras: %d | Variáveis após encoding: %d",
        features.shape[0],
        features.shape[1],
    )

    return {
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
        "dataset": dataset,
    }


def generate_socioeconomic_plots(
    processed_al: dict,
    processed_br: dict,
) -> None:
    """
    Gera as curvas gaussianas de NT_GER para as variáveis
    socioeconômicas descritas no arquivo de mapeamento.
    """

    logger.info(
        "\nETAPA 3 - DISTRIBUIÇÕES GAUSSIANAS DAS VARIÁVEIS SOCIOECONÔMICAS"
    )

    mapping_path = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "variaveis_socioeconomicas.csv"
    )

    if not mapping_path.is_file():
        raise FileNotFoundError(
            "Arquivo de mapeamento socioeconômico não encontrado: "
            f"{mapping_path}"
        )

    files_al = plot_all_socioeconomic_distributions(
        dataset=processed_al["dataset"],
        mapping_path=mapping_path,
        output_dir=ALAGOAS_IMAGES_DIR,
        region="Alagoas",
    )

    files_br = plot_all_socioeconomic_distributions(
        dataset=processed_br["dataset"],
        mapping_path=mapping_path,
        output_dir=BRASIL_IMAGES_DIR,
        region="Brasil",
    )

    logger.info(
        "Gráficos gaussianos gerados para Alagoas: %d",
        len(files_al),
    )
    logger.info(
        "Gráficos gaussianos gerados para o Brasil: %d",
        len(files_br),
    )


def run_pipeline() -> None:
    """Executa todas as etapas do projeto."""

    logger.info("=" * 65)
    logger.info("INÍCIO DO PIPELINE DE ANÁLISE DO ENADE")
    logger.info("=" * 65)

    # ---------------------------------------------------------
    # 1. Preparação dos diretórios
    # ---------------------------------------------------------

    create_output_directories()

    # ---------------------------------------------------------
    # 2. Carregamento dos dados selecionados
    # ---------------------------------------------------------

    logger.info("\nETAPA 1 - CARREGAMENTO DOS DADOS")

    datasets_al = load_region_data(
        region="AL",
        years=AVAILABLE_AL_YEARS,
    )

    datasets_br = load_region_data(
        region="BR",
        years=AVAILABLE_BR_YEARS,
    )

    # ---------------------------------------------------------
    # 3. Pré-processamento independente
    # ---------------------------------------------------------

    logger.info("\nETAPA 2 - PRÉ-PROCESSAMENTO")

    processed_al = preprocess_region(
        region="Alagoas",
        datasets=datasets_al,
    )

    processed_br = preprocess_region(
        region="Brasil",
        datasets=datasets_br,
    )

    # ---------------------------------------------------------
    # 4. Distribuições gaussianas socioeconômicas
    # ---------------------------------------------------------

    generate_socioeconomic_plots(
        processed_al=processed_al,
        processed_br=processed_br,
    )

    # ---------------------------------------------------------
    # 5. Experimentos de Alagoas
    # ---------------------------------------------------------

    logger.info("\nETAPA 4 - EXPERIMENTOS DE ALAGOAS")

    results_al = run_region_experiment(
        region="alagoas",
        features=processed_al["features"],
        labels=processed_al["labels"],
        dataset=processed_al["dataset"],
        feature_names=processed_al["feature_names"],
        cv=CV_FOLDS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # 6. Experimentos do Brasil
    # ---------------------------------------------------------

    logger.info("\nETAPA 5 - EXPERIMENTOS DO BRASIL")

    results_br = run_region_experiment(
        region="brasil",
        features=processed_br["features"],
        labels=processed_br["labels"],
        dataset=processed_br["dataset"],
        feature_names=processed_br["feature_names"],
        cv=CV_FOLDS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # 7. Comparação regional
    # ---------------------------------------------------------

    logger.info("\nETAPA 6 - COMPARAÇÃO ENTRE AS REGIÕES")

    run_regional_comparison(
        results_al=results_al,
        results_br=results_br,
    )

    # ---------------------------------------------------------
    # 8. Encerramento
    # ---------------------------------------------------------

    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
    logger.info("=" * 65)

    logger.info(
        "Resultados disponíveis em: %s",
        RESULTS_DIR,
    )


def main() -> None:
    """Executa o pipeline e registra eventuais erros."""

    try:
        run_pipeline()

    except FileNotFoundError:
        logger.exception(
            "Não foi possível localizar um ou mais arquivos de entrada."
        )
        raise

    except Exception:
        logger.exception(
            "O pipeline foi interrompido devido a um erro."
        )
        raise


if __name__ == "__main__":
    main()
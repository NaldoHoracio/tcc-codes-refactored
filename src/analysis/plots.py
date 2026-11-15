
"""Funções para geração de gráficos e visualizações dos experimentos."""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# Funções auxiliares
# =============================================================

def _prepare_output_path(output_path):
    """Cria o diretório de destino e retorna o caminho do arquivo."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _extract_values(labels):
    """Extrai os valores numéricos da variável-alvo em formato 1D."""

    if isinstance(labels, pd.DataFrame):
        values = labels.iloc[:, 0].to_numpy()

    elif isinstance(labels, pd.Series):
        values = labels.to_numpy()

    else:
        values = np.asarray(labels)

    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _normal_pdf(x, mean, std):
    """Calcula a função densidade de probabilidade normal."""

    if std <= 0:
        raise ValueError(
            "O desvio padrão precisa ser maior que zero."
        )

    return (
        1 / (std * np.sqrt(2 * np.pi))
    ) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


# =============================================================
# Distribuição gaussiana da variável-alvo
# =============================================================


def plot_target_distribution(
    labels,
    output_path,
    title="Distribuição das notas do ENADE",
    xlabel="Notas do ENADE",
    xlim=(0, 100),
    color="royalblue",
):
    """
    Plota a distribuição gaussiana da variável-alvo.

    Apresenta:
    - Curva gaussiana;
    - Área preenchida em azul;
    - Linha vertical tracejada na média;
    - Média e desvio padrão à direita da curva.
    """

    output_path = _prepare_output_path(output_path)

    values = _extract_values(labels)

    if len(values) < 2:
        raise ValueError(
            "São necessários pelo menos dois valores para calcular "
            "a distribuição."
        )

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if std <= 0:
        raise ValueError(
            "Não é possível gerar a curva: desvio padrão igual a zero."
        )

    # Domínio da curva gaussiana.
    if xlim is not None:
        x = np.linspace(xlim[0], xlim[1], 1000)
    else:
        x = np.linspace(
            mean - 4 * std,
            mean + 4 * std,
            1000,
        )

    # Densidade da distribuição normal.
    y = _normal_pdf(x, mean, std)

    max_ylim = y.max()

    fig, ax = plt.subplots(figsize=(11, 7))

    # Curva gaussiana.
    ax.plot(
        x,
        y,
        color="black",
        linewidth=1.5,
    )

    # Preenchimento da curva.
    ax.fill_between(
        x,
        y,
        color=color,
    )

    # Linha vertical indicando a média.
    ax.axvline(
        mean,
        color="black",
        linestyle="dashed",
        linewidth=1.5,
    )

    # Posicionamento das informações estatísticas.
    text_x = mean * 1.4

    ax.text(
        text_x,
        max_ylim * 0.90,
        f"Média: {mean:.2f}",
        fontsize=12,
        style="italic",
        weight="bold",
        va="top"
    )

    ax.text(
        text_x,
        max_ylim * 0.83,
        f"Desvio padrão: {std:.2f}",
        fontsize=12,
        style="italic",
        weight="bold",
        va="top"
    )

    # Título e rótulos.
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Distribuição", fontsize=14)

    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.015,
    )

    plt.close(fig)


# =============================================================
# Distribuição das questões do questionário
# =============================================================

def plot_question_distributions(
    dataset,
    output_path,
    title="Distribuição das respostas do questionário",
    question_columns=None,
):
    """
    Gera um painel com a distribuição das respostas das questões
    QE_I01 até QE_I26.

    Cada questão é representada em um subplot.
    """

    output_path = _prepare_output_path(output_path)

    if question_columns is None:
        question_columns = [
            f"QE_I{i:02d}" for i in range(1, 27)
        ]

    available_columns = [
        column
        for column in question_columns
        if column in dataset.columns
    ]

    if not available_columns:
        raise ValueError(
            "Nenhuma coluna de questão foi encontrada no dataset."
        )

    n_columns = 4
    n_rows = math.ceil(len(available_columns) / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18, 4 * n_rows),
    )

    axes = np.asarray(axes).flatten()

    for index, column in enumerate(available_columns):

        ax = axes[index]

        values = dataset[column].dropna().astype(str)
        frequencies = values.value_counts().head(8)
        frequencies = frequencies.sort_index()

        ax.bar(
            frequencies.index,
            frequencies.values,
            edgecolor="black",
        )

        ax.set_title(column)
        ax.set_xlabel("Resposta")
        ax.set_ylabel("Frequência")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    for index in range(len(available_columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================
# Comparação de distribuição: Alagoas x Brasil
# =============================================================

def plot_target_comparison(
    labels_al,
    labels_br,
    output_path,
    title="Distribuição das notas do ENADE: Alagoas x Brasil",
    xlabel="Notas do ENADE",
    xlim=(0, 100),
):
    """
    Compara as distribuições gaussianas de NT_GER de Alagoas e Brasil.

    Apresenta as duas curvas, suas médias e desvios padrão.
    """

    output_path = _prepare_output_path(output_path)

    values_al = _extract_values(labels_al)
    values_br = _extract_values(labels_br)

    if len(values_al) < 2 or len(values_br) < 2:
        raise ValueError(
            "Cada região precisa ter pelo menos dois valores."
        )

    mean_al = np.mean(values_al)
    std_al = np.std(values_al, ddof=1)

    mean_br = np.mean(values_br)
    std_br = np.std(values_br, ddof=1)

    if std_al <= 0 or std_br <= 0:
        raise ValueError(
            "O desvio padrão das duas regiões precisa ser maior que zero."
        )

    if xlim is not None:
        x = np.linspace(xlim[0], xlim[1], 1000)
    else:
        lower = min(
            mean_al - 4 * std_al,
            mean_br - 4 * std_br,
        )
        upper = max(
            mean_al + 4 * std_al,
            mean_br + 4 * std_br,
        )
        x = np.linspace(lower, upper, 1000)

    y_al = _normal_pdf(x, mean_al, std_al)
    y_br = _normal_pdf(x, mean_br, std_br)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Curva de Alagoas.
    ax.fill_between(
        x,
        y_al,
        alpha=0.35,
        color="royalblue",
    )

    ax.plot(
        x,
        y_al,
        color="royalblue",
        linewidth=2.5,
        label="Alagoas",
    )

    # Curva do Brasil.
    ax.fill_between(
        x,
        y_br,
        alpha=0.25,
        color="darkorange",
    )

    ax.plot(
        x,
        y_br,
        color="darkorange",
        linewidth=2.5,
        label="Brasil",
    )

    # Médias.
    ax.axvline(
        mean_al,
        color="royalblue",
        linestyle="--",
        linewidth=2,
    )

    ax.axvline(
        mean_br,
        color="darkorange",
        linestyle="--",
        linewidth=2,
    )

    # Estatísticas.
    annotation = (
        f"Alagoas\n"
        f"Média: {mean_al:.2f}\n"
        f"Desvio padrão: {std_al:.2f}\n\n"
        f"Brasil\n"
        f"Média: {mean_br:.2f}\n"
        f"Desvio padrão: {std_br:.2f}"
    )

    ax.text(
        0.65,
        0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left"
    )

    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Distribuição", fontsize=14)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend()
    ax.grid(False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================
# Comparação do desempenho dos modelos
# =============================================================

def plot_model_comparison(
    metrics,
    output_path,
    metric="cv_r2_mean",
    title="Comparação de desempenho dos modelos",
):
    """Compara o desempenho dos modelos para uma métrica específica."""

    output_path = _prepare_output_path(output_path)

    if metric not in metrics.columns:
        raise ValueError(
            f"A métrica '{metric}' não foi encontrada na tabela."
        )

    data = metrics.copy()

    if "model" in data.columns:
        data = data.set_index("model")

    if data.empty:
        raise ValueError("A tabela de métricas está vazia.")

    values = pd.to_numeric(
        data[metric],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    values.sort_values().plot(
        kind="barh",
        ax=ax,
        edgecolor="black",
    )

    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Modelo")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_regional_model_comparison(
    metrics_al,
    metrics_br,
    output_path,
    metric="cv_r2_mean",
    title="Comparação dos modelos: Alagoas x Brasil",
):
    """Compara o desempenho dos modelos treinados em cada região."""

    output_path = _prepare_output_path(output_path)

    def prepare_metrics(metrics):

        data = metrics.copy()

        if "model" in data.columns:
            data = data.set_index("model")

        if metric not in data.columns:
            raise ValueError(
                f"A métrica '{metric}' não foi encontrada."
            )

        return pd.to_numeric(
            data[metric],
            errors="coerce",
        )

    alagoas = prepare_metrics(metrics_al)
    brasil = prepare_metrics(metrics_br)

    comparison = pd.concat(
        [
            alagoas.rename("Alagoas"),
            brasil.rename("Brasil"),
        ],
        axis=1,
    )

    comparison = comparison.dropna(how="all")

    fig, ax = plt.subplots(figsize=(11, 6))

    comparison.plot(
        kind="bar",
        ax=ax,
        edgecolor="black",
    )

    ax.set_title(title)
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Região")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================
# Importância das variáveis
# =============================================================

def plot_feature_importance(
    importances,
    output_path,
    top_n=20,
    title="Importância das variáveis",
):
    """Gera um gráfico com as variáveis mais importantes."""

    output_path = _prepare_output_path(output_path)

    if importances.empty:
        raise ValueError("A tabela de importâncias está vazia.")

    data = importances.copy()

    importance_column = (
        "absolute_importance"
        if "absolute_importance" in data.columns
        else "importance"
    )

    if importance_column not in data.columns:
        raise ValueError(
            "A tabela precisa conter 'importance' ou "
            "'absolute_importance'."
        )

    data[importance_column] = pd.to_numeric(
        data[importance_column],
        errors="coerce",
    )

    data = data.dropna(subset=[importance_column])

    data = data.nlargest(
        top_n,
        importance_column,
    )

    data = data.sort_values(importance_column)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        data["feature"],
        data[importance_column],
        edgecolor="black",
    )

    ax.set_title(title)
    ax.set_xlabel("Importância absoluta")
    ax.set_ylabel("Variável")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)





# =============================================================
# Distribuições gaussianas por variável socioeconômica
# =============================================================

def _normalize_question_name(name):
    """Converte QE_02 e QE_I02 para a nomenclatura QE_I02."""
    name = str(name).strip().upper()
    match = re.fullmatch(r"QE_(?:I)?(\d{1,2})", name)

    if match:
        return f"QE_I{int(match.group(1)):02d}"

    return name


def _normalize_answer_code(value):
    """Padroniza o código da alternativa para comparação com o CSV."""
    if pd.isna(value):
        return None

    return str(value).strip().upper()


def plot_socioeconomic_distribution(
    dataset,
    mapping,
    question,
    output_path,
    region,
    target_column="NT_GER",
):
    """
    Plota uma curva gaussiana das notas NT_GER para cada alternativa
    da variável socioeconômica informada.
    """

    question_normalized = _normalize_question_name(question)

    question_column = next(
        (
            column
            for column in dataset.columns
            if _normalize_question_name(column) == question_normalized
        ),
        None,
    )

    if question_column is None:
        raise KeyError(
            f"Variável {question} (equivalente a {question_normalized}) "
            "não encontrada no dataset."
        )

    if target_column not in dataset.columns:
        raise KeyError(f"Coluna {target_column} não encontrada no dataset.")

    question_mapping = mapping[
        mapping["_variavel_normalizada"] == question_normalized
    ].copy()

    if question_mapping.empty:
        raise ValueError(
            f"Não há alternativas cadastradas para {question} no CSV."
        )

    data = dataset[[target_column, question_column]].copy()
    data[target_column] = pd.to_numeric(
        data[target_column].astype("string").str.replace(
            ",", ".", regex=False
        ),
        errors="coerce",
    )
    data["_codigo"] = data[question_column].map(_normalize_answer_code)
    data = data.dropna(subset=[target_column, "_codigo"])

    alternatives = question_mapping.to_dict("records")
    n_plots = len(alternatives)
    n_columns = 2
    n_rows = math.ceil(n_plots / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13, 3.8 * n_rows),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)
    x = np.linspace(0, 100, 1000)

    for index, alternative in enumerate(alternatives):
        ax = axes[index]
        code = _normalize_answer_code(alternative["Código"])
        description = str(alternative["Descrição"]).strip()

        values = data.loc[
            data["_codigo"] == code,
            target_column,
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        ax.set_title(
            f"{code}: {description}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Nota do ENADE (NT_GER)")
        ax.set_ylabel("Densidade")
        ax.grid(axis="y", alpha=0.2)

        if len(values) < 2:
            ax.text(
                0.5,
                0.5,
                f"Dados insuficientes (n={len(values)})",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        if std <= 0:
            ax.axvline(mean, color="black", linestyle="--", linewidth=1.2)
            ax.text(
                0.5,
                0.5,
                "Desvio padrão igual a zero",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        y = _normal_pdf(x, mean, std)

        ax.plot(x, y, color="black", linewidth=1.3)
        ax.fill_between(x, y, color="royalblue", alpha=0.85)
        ax.axvline(
            mean,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

        max_y = float(np.max(y))
        ax.text(
            0.85,
            0.93,
            f"Média: {mean:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
            weight="bold",
            ha="right",
            va="top",
        )
        ax.text(
            0.85,
            0.84,
            f"Desvio padrão: {std:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
            weight="bold",
            ha="right",
            va="top",
        )

    for index in range(n_plots, len(axes)):
        fig.delaxes(axes[index])

    fig.suptitle(
        f"Distribuição das notas do ENADE por {question} — {region}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    output_path = _prepare_output_path(output_path)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_all_socioeconomic_distributions(
    dataset,
    mapping_path,
    output_dir,
    region,
):
    """
    Gera uma figura para cada variável listada no CSV de mapeamento.
    """

    mapping = pd.read_csv(
        mapping_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        dtype=str,
    )

    required_columns = {"Variável", "Código", "Descrição"}
    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(
            "O CSV de mapeamento precisa conter as colunas "
            f"{sorted(required_columns)}. Ausentes: {sorted(missing_columns)}"
        )

    mapping = mapping.dropna(
        subset=["Variável", "Código", "Descrição"]
    ).copy()

    mapping["_variavel_normalizada"] = mapping["Variável"].map(
        _normalize_question_name
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for question in mapping["Variável"].drop_duplicates():
        normalized_question = _normalize_question_name(question)

        output_path = output_dir / (
            f"distribuicao_gaussiana_{normalized_question.lower()}.png"
        )

        generated_files.append(
            plot_socioeconomic_distribution(
                dataset=dataset,
                mapping=mapping,
                question=question,
                output_path=output_path,
                region=region,
            )
        )

    return generated_files



# =============================================================
# Execução independente dos gráficos
# =============================================================

def main():
    """
    Executa a geração dos gráficos sem realizar treinamento
    ou avaliação dos modelos.

    Gera:
    - Distribuição gaussiana das notas de Alagoas;
    - Distribuição gaussiana das notas do Brasil;
    - Comparação das distribuições de Alagoas e Brasil;
    - Distribuição das questões do questionário para cada região;\n    - Distribuições gaussianas de NT_GER para as variáveis socioeconômicas do CSV.
    """

    from src.config import (
        PROJECT_ROOT,
        SELECTED_DATA_DIR,
        AVAILABLE_AL_YEARS,
        AVAILABLE_BR_YEARS,
        ALAGOAS_IMAGES_DIR,
        BRASIL_IMAGES_DIR,
        EXPERIMENTS_IMAGES_DIR,
    )

    from src.data_processing.ingestion import load_selected_datasets
    from src.data_processing.preprocessing import preprocess_enade

    print("=" * 60)
    print("GERAÇÃO DOS GRÁFICOS DO ENADE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Preparação dos diretórios
    # ---------------------------------------------------------

    for directory in (
        ALAGOAS_IMAGES_DIR,
        BRASIL_IMAGES_DIR,
        EXPERIMENTS_IMAGES_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Carregamento dos dados de Alagoas
    # ---------------------------------------------------------

    print("\nCarregando dados de Alagoas...")

    datasets_al = load_selected_datasets(
        data_dir=SELECTED_DATA_DIR,
        years=AVAILABLE_AL_YEARS,
        prefix="AL",
    )

    features_al, labels_al, feature_names_al, dataset_al = (
        preprocess_enade(datasets_al)
    )

    print(f"Alagoas: {len(labels_al)} registros.")

    # ---------------------------------------------------------
    # 3. Carregamento dos dados do Brasil
    # ---------------------------------------------------------

    print("\nCarregando dados do Brasil...")

    datasets_br = load_selected_datasets(
        data_dir=SELECTED_DATA_DIR,
        years=AVAILABLE_BR_YEARS,
        prefix="BR",
    )

    features_br, labels_br, feature_names_br, dataset_br = (
        preprocess_enade(datasets_br)
    )

    print(f"Brasil: {len(labels_br)} registros.")

    # ---------------------------------------------------------
    # 4. Distribuição gaussiana de Alagoas
    # ---------------------------------------------------------

    print("\nGerando gráfico de Alagoas...")

    plot_target_distribution(
        labels_al,
        ALAGOAS_IMAGES_DIR / "target_distribution_alagoas.png",
        title="Distribuição de notas do ENADE em Alagoas: 2014 a 2018",
    )

    plot_question_distributions(
        dataset_al,
        ALAGOAS_IMAGES_DIR / "question_distributions_alagoas.png",
        title="Distribuição das questões do ENADE - Alagoas",
    )

    mapping_path = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "variaveis_socioeconomicas.csv"
    )

    files_al = plot_all_socioeconomic_distributions(
        dataset=dataset_al,
        mapping_path=mapping_path,
        output_dir=ALAGOAS_IMAGES_DIR,
        region="Alagoas",
    )
    print(f"Gráficos socioeconômicos de Alagoas: {len(files_al)}")

    # ---------------------------------------------------------
    # 5. Distribuição gaussiana do Brasil
    # ---------------------------------------------------------

    print("\nGerando gráfico do Brasil...")

    plot_target_distribution(
        labels_br,
        BRASIL_IMAGES_DIR / "target_distribution_brasil.png",
        title="Distribuição de notas do ENADE no Brasil: 2014 a 2018",
    )

    plot_question_distributions(
        dataset_br,
        BRASIL_IMAGES_DIR / "question_distributions_brasil.png",
        title="Distribuição das questões do ENADE - Brasil",
    )

    files_br = plot_all_socioeconomic_distributions(
        dataset=dataset_br,
        mapping_path=mapping_path,
        output_dir=BRASIL_IMAGES_DIR,
        region="Brasil",
    )
    print(f"Gráficos socioeconômicos do Brasil: {len(files_br)}")

    # ---------------------------------------------------------
    # 6. Comparação entre Alagoas e Brasil
    # ---------------------------------------------------------

    print("\nGerando comparação entre Alagoas e Brasil...")

    plot_target_comparison(
        labels_al,
        labels_br,
        EXPERIMENTS_IMAGES_DIR / "target_comparison_alagoas_brasil.png",
        title="Distribuição de notas do ENADE: Alagoas x Brasil (2014 a 2018)",
    )

    # ---------------------------------------------------------
    # 7. Encerramento
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GERAÇÃO DOS GRÁFICOS CONCLUÍDA!")
    print("=" * 60)

    print(f"\nAlagoas: {ALAGOAS_IMAGES_DIR}")
    print(f"Brasil: {BRASIL_IMAGES_DIR}")
    print(f"Comparações: {EXPERIMENTS_IMAGES_DIR}")


if __name__ == "__main__":
    main()
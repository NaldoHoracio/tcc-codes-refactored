"""Gráficos animados (GIF) e versões estáticas (PNG).

Lê as tabelas geradas pelo pipeline em ``resultados/tabelas/experiments`` e
grava os arquivos em ``resultados/imagens/animacoes``:

- ``distribuicao_nt_ger.gif`` / ``.png``: distribuição de NT_GER em Alagoas e no Brasil;
- ``comparacao_modelos.gif`` / ``.png``: R² médio da validação cruzada por modelo;
- ``importancia_questoes.gif`` / ``.png``: questões mais importantes (Random Forest).

Uso (a partir da raiz do projeto):
    python src/analysis/animations.py
    python src/analysis/animations.py --graficos distribuicao modelos
    python -m src.analysis.animations --saida outra/pasta
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite executar o arquivo diretamente (python src/analysis/animations.py),
# sem depender de "python -m": adiciona a raiz do projeto ao sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from src.config import EXPERIMENTS_TABLES_DIR, RESULTS_IMAGES_DIR  # noqa: E402

logger = logging.getLogger(__name__)

ANIMATIONS_DIR = RESULTS_IMAGES_DIR / "animacoes"

# Paleta validada (cores categóricas de alto contraste, seguras para daltonismo).
REGION_COLORS = {"alagoas": "#2a78d6", "brasil": "#eb6834"}
REGION_LABELS = {"alagoas": "Alagoas", "brasil": "Brasil"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

FPS = 10
FIGSIZE = (8, 4.5)
GIF_DPI = 100
PNG_DPI = 150

MODEL_LABELS = {
    "linear_regression": "Regressão Linear",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "decision_tree": "Árvore de Decisão",
}

QUESTION_LABELS = {
    "QE_I01": "Estado civil",
    "QE_I02": "Cor/raça",
    "QE_I03": "Nacionalidade",
    "QE_I04": "Escolaridade do pai",
    "QE_I05": "Escolaridade da mãe",
    "QE_I06": "Com quem mora",
    "QE_I07": "Pessoas na casa",
    "QE_I08": "Renda familiar",
    "QE_I09": "Situação financeira",
    "QE_I10": "Situação de trabalho",
    "QE_I11": "Bolsa/financiamento",
    "QE_I12": "Auxílio permanência",
    "QE_I13": "Bolsa acadêmica",
    "QE_I14": "Intercâmbio",
    "QE_I15": "Ação afirmativa",
    "QE_I16": "UF do ensino médio",
    "QE_I17": "Tipo de escola no EM",
    "QE_I18": "Modalidade do EM",
    "QE_I19": "Quem incentivou a graduação",
    "QE_I20": "Grupo de apoio",
    "QE_I21": "Familiar com nível superior",
    "QE_I22": "Livros lidos no ano",
    "QE_I23": "Horas de estudo semanais",
    "QE_I24": "Atividade de idioma",
    "QE_I25": "Motivo da escolha do curso",
    "QE_I26": "Motivo da escolha da IES",
}


# =============================================================
# Utilidades
# =============================================================

def _ease(t: float) -> float:
    """Suavização cúbica (ease-in-out) para t em [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return 3 * t**2 - 2 * t**3


def _style_axes(ax: plt.Axes) -> None:
    """Aplica eixos e grade discretos."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.95, title, fontsize=13, fontweight="bold",
             color=TEXT_PRIMARY, ha="left", va="top")
    fig.text(0.02, 0.885, subtitle, fontsize=9.5,
             color=TEXT_SECONDARY, ha="left", va="top")


def _save(fig: plt.Figure, anim: FuncAnimation, name: str,
          total_frames: int, output_dir: Path,
          static_figure: plt.Figure | None = None) -> tuple[Path, Path]:
    """Salva o GIF e uma versão PNG (quadro final ou figura estática própria)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{name}.gif"
    png_path = output_dir / f"{name}.png"

    anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=GIF_DPI,
              savefig_kwargs={"facecolor": SURFACE})

    if static_figure is None:
        anim._func(total_frames - 1)  # quadro final na versão estática
        static_figure = fig
    static_figure.savefig(png_path, dpi=PNG_DPI, facecolor=SURFACE)
    plt.close(fig)
    plt.close(static_figure)

    logger.info("Gerados: %s e %s", gif_path, png_path)
    return gif_path, png_path


def _read_labels(region: str, tables_dir: Path) -> np.ndarray:
    path = tables_dir / f"labels_{region}.csv"
    values = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    return values[np.isfinite(values)]


# =============================================================
# 1. Distribuição de NT_GER
# =============================================================

def animate_target_distribution(tables_dir: Path = EXPERIMENTS_TABLES_DIR,
                                output_dir: Path = ANIMATIONS_DIR):
    """Histogramas de NT_GER que crescem até a distribuição final."""
    bins = np.arange(0, 102, 2)
    data = {}
    for region in REGION_COLORS:
        values = _read_labels(region, tables_dir)
        counts, _ = np.histogram(values, bins=bins)
        data[region] = {"density": counts / counts.sum() * 100,
                        "mean": values.mean()}

    y_max = max(d["density"].max() for d in data.values()) * 1.12

    fig, ax = _new_figure()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.13)
    _title(fig, "Distribuição da nota geral do ENADE (NT_GER)",
           "Percentual de estudantes - ENADE 2014–2018")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Nota Geral do ENADE", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("% de estudantes", color=TEXT_SECONDARY, fontsize=9.5)
    ax.grid(axis="x", visible=False)

    stairs, mean_lines, mean_texts = {}, {}, {}
    for region, color in REGION_COLORS.items():
        stairs[region] = ax.stairs(np.zeros_like(data[region]["density"]), bins,
                                   fill=True, color=color, alpha=0.22,
                                   linewidth=0)
        stairs[region + "_line"] = ax.stairs(
            np.zeros_like(data[region]["density"]), bins, color=color,
            linewidth=2, label=REGION_LABELS[region])
        mean_lines[region] = ax.axvline(data[region]["mean"], color=color,
                                        linewidth=1.5, linestyle="--",
                                        alpha=0.0)
        mean_texts[region] = ax.text(0, 0, "", fontsize=9,
                                     color=TEXT_PRIMARY, alpha=0.0)

    ax.legend(loc="upper right", frameon=False, fontsize=9.5,
              labelcolor=TEXT_PRIMARY)

    grow, fade, hold = 40, 15, 30
    total = grow + fade + hold

    def update(frame: int):
        k = _ease(frame / (grow - 1))
        appear = _ease((frame - grow) / fade)
        for i, region in enumerate(REGION_COLORS):
            density = data[region]["density"] * k
            stairs[region].set_data(density)
            stairs[region + "_line"].set_data(density)
            mean = data[region]["mean"]
            mean_lines[region].set_alpha(appear)
            text = mean_texts[region]
            offset = 1.5 if i == 1 else -1.5
            text.set_position((mean + offset, y_max * 0.94))
            text.set_ha("left" if offset > 0 else "right")
            text.set_text(f"média {REGION_LABELS[region]}: {mean:.1f}".replace(".", ","))
            text.set_alpha(appear)
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS)
    return _save(fig, anim, "distribuicao_nt_ger", total, output_dir)


# =============================================================
# 2. Comparação dos modelos
# =============================================================

def animate_model_comparison(tables_dir: Path = EXPERIMENTS_TABLES_DIR,
                             output_dir: Path = ANIMATIONS_DIR):
    """Barras horizontais do R² médio (CV) por modelo e região."""
    metrics = {
        region: pd.read_csv(tables_dir / f"model_metrics_{region}.csv")
        .set_index("model")
        for region in REGION_COLORS
    }
    order = (metrics["alagoas"]["cv_r2_mean"] + metrics["brasil"]["cv_r2_mean"]
             ).sort_values().index.tolist()

    fig, ax = _new_figure()
    fig.subplots_adjust(left=0.22, right=0.95, top=0.78, bottom=0.13)
    _title(fig, "Desempenho dos modelos de regressão",
           "R² médio e barras de erro (desvio-padrão)")

    x_max = max(m["cv_r2_mean"].max() + m["cv_r2_std"].max()
                for m in metrics.values()) * 1.18
    ax.set_xlim(0, x_max)
    ax.set_xlabel("R²", color=TEXT_SECONDARY, fontsize=9.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
    ax.grid(axis="y", visible=False)

    positions = np.arange(len(order))
    height = 0.36
    bars, errors, labels = {}, {}, {}
    for i, (region, color) in enumerate(REGION_COLORS.items()):
        y = positions + (height / 2 + 0.02) * (1 if i == 0 else -1)
        bars[region] = ax.barh(y, np.zeros(len(order)), height=height,
                               color=color, label=REGION_LABELS[region])
        values = metrics[region].loc[order]
        errors[region] = ax.errorbar(values["cv_r2_mean"], y,
                                     xerr=values["cv_r2_std"], fmt="none",
                                     ecolor=TEXT_SECONDARY, elinewidth=1,
                                     capsize=3, alpha=0.0)
        labels[region] = [
            ax.text(0, yy, "", va="center", fontsize=8.5,
                    color=TEXT_PRIMARY, alpha=0.0)
            for yy in y
        ]

    ax.set_yticks(positions, [MODEL_LABELS.get(m, m) for m in order],
                  color=TEXT_PRIMARY, fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=TEXT_PRIMARY)

    grow, fade, hold = 40, 15, 30
    total = grow + fade + hold

    def update(frame: int):
        k = _ease(frame / (grow - 1))
        appear = _ease((frame - grow) / fade)
        for region in REGION_COLORS:
            values = metrics[region].loc[order]
            for bar, value in zip(bars[region], values["cv_r2_mean"]):
                bar.set_width(value * k)
            for artist in errors[region].lines[1] + errors[region].lines[2]:
                artist.set_alpha(appear)
            for text, mean, std in zip(labels[region], values["cv_r2_mean"],
                                       values["cv_r2_std"]):
                text.set_x(mean + std + x_max * 0.012)
                text.set_text(f"{mean:.3f}".replace(".", ","))
                text.set_alpha(appear)
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS)
    return _save(fig, anim, "comparacao_modelos", total, output_dir)


# =============================================================
# 3. Importância das questões (Random Forest)
# =============================================================

def animate_question_importance(tables_dir: Path = EXPERIMENTS_TABLES_DIR,
                                output_dir: Path = ANIMATIONS_DIR,
                                model: str = "random_forest",
                                top_n: int = 8):
    """Ranking das questões mais importantes, alternando Alagoas e Brasil."""
    importance = {}
    for region in REGION_COLORS:
        table = pd.read_csv(
            tables_dir / f"question_importance_{region}_{model}.csv"
        ).set_index("question")["absolute_importance"]
        importance[region] = table / table.sum() * 100  # % da importância total

    union = pd.concat(importance, axis=1).fillna(0.0)
    questions = (union.max(axis=1).sort_values(ascending=False)
                 .head(top_n).index.tolist())
    union = union.loc[questions]
    regions = list(REGION_COLORS)

    def ranks(region: str) -> pd.Series:
        # posição 0 = topo do gráfico (maior importância)
        order = union[region].sort_values(ascending=False).index
        return pd.Series(np.arange(len(order)), index=order).loc[questions]

    rank = {region: ranks(region) for region in regions}

    fig, ax = _new_figure()
    fig.subplots_adjust(left=0.33, right=0.95, top=0.78, bottom=0.13)
    _title(fig, f"Fatores socioeconômicos que mais impactaram no {MODEL_LABELS.get(model, model)}",
           f"Impacto de cada fator socioeconômico na obtenção da nota do unade usando {MODEL_LABELS.get(model, model)} (em %)")
    x_max = union.to_numpy().max() * 1.2
    ax.set_xlim(0, x_max)
    ax.set_ylim(top_n - 0.4, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% da importância total", color=TEXT_SECONDARY, fontsize=9.5)

    bars = ax.barh(np.arange(top_n), np.zeros(top_n), height=0.68)
    names = [
        ax.text(-x_max * 0.015, 0, f"{QUESTION_LABELS.get(q, q)}  ({q})",
                ha="right", va="center", fontsize=9.5, color=TEXT_PRIMARY)
        for q in questions
    ]
    values_text = [ax.text(0, 0, "", va="center", fontsize=8.5,
                           color=TEXT_PRIMARY) for _ in questions]
    region_tag = ax.text(0.98, 0.06, "", transform=ax.transAxes, ha="right",
                         fontsize=15, fontweight="bold")

    hold, move = 35, 20
    segment = hold + move
    total = segment * len(regions)  # A → B → A (loop contínuo)

    def update(frame: int):
        index, local = divmod(frame, segment)
        current = regions[index % len(regions)]
        following = regions[(index + 1) % len(regions)]
        t = _ease((local - hold) / move) if local >= hold else 0.0
        shown = current if t < 0.5 else following

        for i, question in enumerate(questions):
            value = (1 - t) * union.loc[question, current] + t * union.loc[question, following]
            y = (1 - t) * rank[current][question] + t * rank[following][question]
            bar = bars[i]
            bar.set_y(y - bar.get_height() / 2)
            bar.set_width(value)
            bar.set_color(REGION_COLORS[shown])
            # Rótulos de barras que trocam de posição somem durante a troca.
            moving = rank[current][question] != rank[following][question]
            alpha = 1 - np.sin(np.pi * t) if moving else 1.0
            bar.set_alpha(0.35 + 0.65 * alpha)
            names[i].set_y(y)
            names[i].set_alpha(alpha)
            values_text[i].set_alpha(alpha)
            values_text[i].set_position((value + x_max * 0.012, y))
            values_text[i].set_text(f"{value:.1f}%".replace(".", ","))
        region_tag.set_text(REGION_LABELS[shown])
        region_tag.set_color(REGION_COLORS[shown])
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS)
    static = _static_question_importance(union, model)
    return _save(fig, anim, "importancia_questoes", total, output_dir,
                 static_figure=static)


def _static_question_importance(union: pd.DataFrame, model: str) -> plt.Figure:
    """Versão estática: barras agrupadas de Alagoas e Brasil lado a lado."""
    data = union.loc[union.max(axis=1).sort_values().index]
    fig, ax = _new_figure()
    fig.subplots_adjust(left=0.33, right=0.95, top=0.78, bottom=0.13)
    _title(fig, f"Fatores socioeconômicos que mais impactaram no {MODEL_LABELS.get(model, model)}",
           f"Impacto de cada fator socioeconômico na obtenção da nota do unade usando {MODEL_LABELS.get(model, model)} (em %)")
    positions = np.arange(len(data))
    height = 0.36
    x_max = data.to_numpy().max() * 1.15
    for i, (region, color) in enumerate(REGION_COLORS.items()):
        y = positions + (height / 2 + 0.02) * (1 if i == 0 else -1)
        ax.barh(y, data[region], height=height, color=color,
                label=REGION_LABELS[region])
        for yy, value in zip(y, data[region]):
            ax.text(value + x_max * 0.01, yy, f"{value:.1f}%".replace(".", ","),
                    va="center", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xlim(0, x_max)
    ax.set_yticks(positions, [f"{QUESTION_LABELS.get(q, q)}  ({q})"
                              for q in data.index],
                  color=TEXT_PRIMARY, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% da importância total", color=TEXT_SECONDARY, fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=TEXT_PRIMARY)
    return fig


# =============================================================
# Execução
# =============================================================

CHARTS = {
    "distribuicao": animate_target_distribution,
    "modelos": animate_model_comparison,
    "importancia": animate_question_importance,
}


def generate_all_animations(tables_dir: Path = EXPERIMENTS_TABLES_DIR,
                            output_dir: Path = ANIMATIONS_DIR,
                            charts: list[str] | None = None) -> list[Path]:
    """Gera os GIFs e PNGs escolhidos (todos, por padrão) e retorna os arquivos."""
    files: list[Path] = []
    for name in charts or list(CHARTS):
        files.extend(CHARTS[name](tables_dir=tables_dir, output_dir=output_dir))
    return files


def main(argv: list[str] | None = None) -> int:
    """Executa somente a geração das animações, sem rodar o pipeline completo."""
    parser = argparse.ArgumentParser(
        description="Gera os gráficos animados (GIF) e estáticos (PNG).")
    parser.add_argument("--graficos", nargs="+", choices=list(CHARTS),
                        default=list(CHARTS),
                        help="Gráficos a gerar (padrão: todos).")
    parser.add_argument("--tabelas", type=Path, default=EXPERIMENTS_TABLES_DIR,
                        help="Pasta com as tabelas geradas pelo pipeline.")
    parser.add_argument("--saida", type=Path, default=ANIMATIONS_DIR,
                        help="Pasta onde os GIFs e PNGs serão gravados.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    if not args.tabelas.is_dir():
        logger.error("Pasta de tabelas não encontrada: %s. "
                     "Execute antes o pipeline (python -m src.main).", args.tabelas)
        return 1

    try:
        files = generate_all_animations(args.tabelas, args.saida, args.graficos)
    except FileNotFoundError as exc:
        logger.error("Arquivo de entrada ausente: %s", exc.filename or exc)
        return 1

    logger.info("%d arquivos gerados em %s", len(files), args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
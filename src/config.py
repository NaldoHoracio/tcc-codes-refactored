
"""Configurações centrais do projeto ENADE."""

from pathlib import Path


# =============================================================
# Diretórios principais
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "data-raw"
SELECTED_DATA_DIR = DATA_DIR / "data-selected"

# Mantido para compatibilidade com códigos antigos.
# O novo pipeline não salva resultados neste diretório.
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# =============================================================
# Diretórios de resultados
# =============================================================

RESULTS_DIR = PROJECT_ROOT / "resultados"

RESULTS_TABLES_DIR = RESULTS_DIR / "tabelas"
EXPERIMENTS_TABLES_DIR = RESULTS_TABLES_DIR / "experiments"

RESULTS_IMAGES_DIR = RESULTS_DIR / "imagens"

ALAGOAS_IMAGES_DIR = RESULTS_IMAGES_DIR / "alagoas"
BRASIL_IMAGES_DIR = RESULTS_IMAGES_DIR / "brasil"
EXPERIMENTS_IMAGES_DIR = RESULTS_IMAGES_DIR / "experiments"

RESULTS_STATISTICS_DIR = RESULTS_DIR / "estatisticas"


# =============================================================
# Configurações dos dados
# =============================================================

AVAILABLE_YEARS = (2014, 2015, 2016, 2017, 2018)

AVAILABLE_AL_YEARS = AVAILABLE_YEARS
AVAILABLE_BR_YEARS = AVAILABLE_YEARS

TARGET_COLUMN = "NT_GER"

QUESTION_COLUMNS = tuple(
    f"QE_I{i:02d}" for i in range(1, 27)
)


# =============================================================
# Configurações dos experimentos
# =============================================================

RANDOM_STATE = 42

CV_FOLDS = 10

TEST_SIZE = 0.33


# =============================================================
# Preparação dos diretórios
# =============================================================

def create_result_directories():
    """Cria todos os diretórios destinados aos resultados."""

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
        directory.mkdir(parents=True, exist_ok=True)

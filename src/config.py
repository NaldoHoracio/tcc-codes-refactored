"""Configurações centrais do projeto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "data-raw"
SELECTED_DATA_DIR = DATA_DIR / "data-selected"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Anos para os quais o projeto possui microdados selecionados.
AVAILABLE_YEARS = (2014, 2015, 2016, 2018)
RANDOM_STATE = 42
TARGET_COLUMN = "NT_GER"
QUESTION_COLUMNS = tuple(f"QE_I{i:02d}" for i in range(1, 27))

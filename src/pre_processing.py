
"""Compatibilidade com funções de pré-processamento anteriores.

O módulo principal de processamento está em:
src.data_processing.preprocessing
"""

from src.data_processing.preprocessing import (
    preprocess_enade,
    preprocess_enade_al,
    processing_set_al,
)

__all__ = [
    "preprocess_enade",
    "preprocess_enade_al",
    "processing_set_al",
]

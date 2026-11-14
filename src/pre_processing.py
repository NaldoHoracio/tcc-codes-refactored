"""Compatibilidade com o módulo de pré-processamento da versão anterior.

Novos códigos devem importar de ``src.processing_data.preprocessing``.
"""
from src.processing_data.preprocessing import preprocess_enade_al, processing_set_al

__all__ = ["preprocess_enade_al", "processing_set_al"]

"""Compatibilidade com funções auxiliares da versão anterior.

Novos códigos devem importar de ``src.utils.helper_functions``.
"""
from src.utils.helper_functions import seconds_transform, version_file

__all__ = ["seconds_transform", "version_file"]

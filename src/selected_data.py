"""Compatibilidade com o módulo de seleção da versão anterior.

Novos códigos devem importar de ``src.processing_data.selection``.
"""
from pathlib import Path

from src.config import RAW_DATA_DIR, SELECTED_DATA_DIR
from src.processing_data.selection import select_enade_microdata
from src.utils.helper_functions import seconds_transform


def seleciona_microdados_enade(
    paths_in: dict,
    uf_excluir: int = 27,
    samples: dict | None = None,
    sep: str = ";",
    prefix_saida: str = "BR",
    paths_out: str | Path = SELECTED_DATA_DIR,
):
    return select_enade_microdata(
        input_paths=paths_in,
        output_dir=paths_out,
        uf_to_exclude=uf_excluir,
        samples=samples,
        separator=sep,
        output_prefix=prefix_saida,
    )


def run():
    """Executa a seleção usando os caminhos convencionais dos microdados brutos."""
    paths = {
        2014: RAW_DATA_DIR / "microdados_enade_2014/3.DADOS/MICRODADOS_ENADE_2014.txt",
        2015: RAW_DATA_DIR / "microdados_enade_2015/3.DADOS/MICRODADOS_ENADE_2015.txt",
        2016: RAW_DATA_DIR / "microdados_enade_2016/3.DADOS/MICRODADOS_ENADE_2016.txt",
        2018: RAW_DATA_DIR / "microdados_enade_2018/3.DADOS/microdados_enade_2018.txt",
    }
    samples = {2014: 7257, 2015: 5469, 2016: 3559, 2018: 5482}
    return seleciona_microdados_enade(paths, samples=samples)


if __name__ == "__main__":
    run()

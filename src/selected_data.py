"""
selected_data.py
---------------------------
Este script processa os microdados do ENADE, realizando seleções e salvando os dados tratados.

"""
from pathlib import Path
import time
import pandas as pd
from helper_functions import seconds_transform


def seleciona_microdados_enade(
    paths_in: dict,
    uf_excluir: int = 27,
    samples: dict | None = None,
    sep: str = ";",
    prefix_saida: str = "BR",
    paths_out: str = "/data/data-raw/"
):
    """
    Seleciona microdados do ENADE para múltiplos anos.

    Etapas:
    - Leitura dos arquivos
    - Exclusão de uma UF específica
    - Remoção de valores NaN
    - Amostragem aleatória
    - Salvamento em CSV

    :param paths_in: dicionário {ano: caminho_do_arquivo}
    :param uf_excluir: código da UF a ser excluída (default = 27 / Alagoas)
    :param samples: dicionário {ano: tamanho_da_amostra}
    :param sep: separador do arquivo CSV
    :param prefix_saida: prefixo dos arquivos de saída
    :param paths_out: diretório de saída dos arquivos processados
    """

    start_time = time.time()
    output_dir = Path(paths_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ano, path in paths_in.items():
        print(f"Processando ENADE {ano}...")

        # 1. Leitura do arquivo
        df = pd.read_csv(path, sep=sep)

        # 2. Excluir UF específica
        df = df[df["QE_I16"] != uf_excluir]

        # 3. Remover linhas com NaN
        df = df.dropna(axis=0)

        # 4. Amostragem aleatória (se definida)
        if samples and ano in samples:
            df = df.sample(samples[ano], random_state=42)

        # 5. Salvar CSV
        output_file = f"{output_dir}/{prefix_saida}_{ano}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nArquivo salvo: {output_file}")

    elapsed = time.time() - start_time
    print(f"\nProcessamento finalizado em {elapsed:.2f} segundos")
    seconds_transform(elapsed)

def run():
    """
    Função de execução padrão
    """
    paths = {
        2014: "./data/data-raw/microdados_enade_2014/3.DADOS/MICRODADOS_ENADE_2014.txt",
        2015: "./data/data-raw/microdados_enade_2015/3.DADOS/MICRODADOS_ENADE_2015.txt",
        2016: "./data/data-raw/microdados_enade_2016/3.DADOS/MICRODADOS_ENADE_2016.txt",
        2018: "./data/data-raw/microdados_enade_2018/3.DADOS/microdados_enade_2018.txt",
    }

    samples = {
        2014: 7257,
        2015: 5469,
        2016: 3559,
        2018: 5482,
    }

    seleciona_microdados_enade(
        paths_in=paths,
        uf_excluir=27,
        samples=samples,
        prefix_saida="BR",
        paths_out="./data/data-selected/"
    )
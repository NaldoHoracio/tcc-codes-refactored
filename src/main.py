"""
main.py
---------------------------
Este é o script principal para processar todo processo, desde a seleção dos dados até a análise (plotagem de graficos)

"""
from selection.selected_data import processa_microdados_enade

def main():
    """
    Docstring for main
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

    processa_microdados_enade(
        paths_in=paths,
        uf_excluir=27,
        samples=samples,
        prefix_saida="BR",
        paths_out="./data/data-selected/"
    )


if __name__ == "__main__":
    main()
"""
Docstring for src.pre_processing

Arquivo: processing_al.py
Descrição:
    Módulo responsável pelo pré-processamento dos dados do ENADE (Alagoas),
    incluindo limpeza, seleção de atributos, integração de múltiplos anos,
    tratamento de valores ausentes e transformação dos dados para uso em
    modelos de Aprendizado de Máquina.

Autor: Edvonaldo Horacio
Criado em: 2026-01-28
Versão: 1.0
"""
import pandas as pd
import numpy as np

def processing_set_al(data_al: list) -> tuple:
    """
    Realiza o processamento completo dos dados do ENADE (AL):
    - Limpeza
    - Seleção de colunas
    - Junção dos datasets
    - Tratamento de valores ausentes
    - Transformação para Machine Learning (One-Hot Encoding)
    
    Retorna:
    - features_al: array numpy com as features codificadas
    - labels_al: array numpy com os rótulos (NT_GER)
    - features_al_list_oh: lista com os nomes das features após one-hot encoding
    - dataset_al: DataFrame final processado (antes da codificação)
    """

    # =========================================================
    # 1. LIMPEZA DOS DADOS
    # =========================================================

    # Remove a coluna 'Unnamed: 0', geralmente criada ao salvar CSVs com índice
    for i in range(len(data_al)):
        if 'Unnamed: 0' in data_al[i].columns:
            del data_al[i]['Unnamed: 0']

    # =========================================================
    # 2. SELEÇÃO DE COLUNAS DE INTERESSE
    # =========================================================
    # Mantém apenas as colunas entre NT_GER e QE_I26
    for i in range(len(data_al)):
        data_al[i] = data_al[i].loc[:, 'NT_GER':'QE_I26']

    # Remove colunas de respostas inválidas ou auxiliares (CO_RS_I1 a CO_RS_I9)
    for i in range(len(data_al)):
        data_al[i] = data_al[i].drop(data_al[i].loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

    # Remove colunas de notas intermediárias (NT_FG a NT_CE_D3)
    for i in range(len(data_al)):
        data_al[i] = data_al[i].drop(data_al[i].loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)

    # =========================================================
    # 3. JUNÇÃO DOS DADOS (MERGE)
    # =========================================================
    # Concatena os dados de todos os anos em um único DataFrame
    #frames = [data_al2014, data_al2015, data_al2016, data_al2017, data_al2018]
    #data_al = pd.concat(frames)

    # =========================================================
    # 4. ENRIQUECIMENTO E TRATAMENTO DE DADOS
    # =========================================================

    # Ajusta o formato decimal da coluna NT_GER
    data_al['NT_GER'] = data_al['NT_GER'].str.replace(',', '.')
    data_al['NT_GER'] = data_al['NT_GER'].astype(float)

    # Calcula a média da nota geral
    data_al_media = round(data_al['NT_GER'].mean(), 2)

    # Preenche valores ausentes com a média
    data_al['NT_GER'] = data_al['NT_GER'].fillna(data_al_media)

    # Salva o dataset final (antes das transformações)
    dataset_al = data_al

    # Estatísticas descritivas (opcional, útil para análise exploratória)
    describe_al = data_al.describe()

    # =========================================================
    # 5. TRANSFORMAÇÃO PARA MACHINE LEARNING
    # =========================================================

    # Define os rótulos (variável alvo)
    labels_al = np.array(data_al['NT_GER'])

    # Remove a variável alvo das features
    data_al = data_al.drop(['NT_GER'], axis=1)

    # Lista das features antes da codificação
    features_al_list = list(data_al.columns)

    # =========================================================
    # 6. ONE-HOT ENCODING DAS VARIÁVEIS CATEGÓRICAS
    # =========================================================

    # Aplica one-hot encoding nas questões do questionário (QE_I01 a QE_I26)
    features_al = pd.get_dummies(
        data=data_al,
        columns=[
            'QE_I01','QE_I02','QE_I03','QE_I04','QE_I05','QE_I06','QE_I07','QE_I08',
            'QE_I09','QE_I10','QE_I11','QE_I12','QE_I13','QE_I14','QE_I15','QE_I16',
            'QE_I17','QE_I18','QE_I19','QE_I20','QE_I21','QE_I22','QE_I23','QE_I24',
            'QE_I25','QE_I26'
        ]
    )

    # Armazena os nomes das features após one-hot encoding
    features_al_list_oh = list(features_al.columns)

    # Converte as features para array NumPy
    features_al = np.array(features_al)

    # =========================================================
    # 7. RETORNO DOS DADOS PROCESSADOS
    # =========================================================
    return features_al, labels_al, features_al_list_oh, dataset_al


if __name__ == "__main__":
    #processing_set_al([])  # Exemplo de chamada da função
    print("Módulo de pré-processamento do ENADE (AL) carregado com sucesso.")
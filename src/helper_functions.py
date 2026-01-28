"""
helper_functions.py
---------------------------
Este modulo contem funcoes auxiliares para serem usadas em outros scripts.

"""
import os
import csv
import pandas as pd

def seconds_transform(seconds_time:float):
    """
     Esta funcao transforma segundos em horas, minutos e segundos
    """
    hours = int(seconds_time/3600)
    rest_1 = seconds_time%3600
    minutes = int(rest_1/60)
    seconds = rest_1 - 60*minutes
    #print(seconds)
    print("Time: ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")


def version_file(name_file:str, fields:str, rows_version):
    """
    Esta funcao cria ou atualiza o arquivo de versao do build
    
    :param name_file: Description
    :param fields: Description
    :param rows_version: Description
    """
    rows_aux = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file)
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value += 1
        rows_version['Version'] = value
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the data rows  
            csvwriter.writerows(rows_aux) 
    else:
        file_version_py = name_file
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the fields
            csvwriter.writeheader()
            # writing the data rows 
            csvwriter.writerows(rows_aux)
            #print ("File not exist")


"""
helper_functions.py
---------------------------
Este modulo contem funcoes auxiliares para serem usadas em outros scripts.

"""

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
# classificadorDeNomes/helpers.py
import numpy as np
import string
import unicodedata
from django.apps import apps

def preparar_input_para_modelo(nome, max_posicoes=20):
    nome = nome.upper()
    alfabeto = string.ascii_uppercase
    input_vector = np.zeros(max_posicoes * len(alfabeto), dtype=int)
    for posicao, letra in enumerate(nome):
        if posicao < max_posicoes and letra in alfabeto:
            indice_letra = alfabeto.index(letra)
            indice_vetor = posicao * len(alfabeto) + indice_letra
            input_vector[indice_vetor] = 1
    return input_vector.reshape(1, -1)

def remove_acentos(string):
    nfkd = unicodedata.normalize('NFKD', string)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def preveEmTodosModelos(nome):
    retorno = {}
    config = apps.get_app_config('classificadorDeNomes')
    modelos = config.modelos
    le = config.le
    data = config.data

    nome = remove_acentos(nome)
    inputDados = preparar_input_para_modelo(nome)

    for nome_modelo, modelo in modelos.items():
        if isinstance(modelo, list):  # Bagging models
            previsoes_agregadas = np.zeros((inputDados.shape[0],))
            for estimador in modelo:
                previsoes = estimador.predict(inputDados).flatten()
                previsoes_agregadas += previsoes
            previsoes_agregadas /= len(modelo)
            resultado = int(round(previsoes_agregadas[0]))
        elif isinstance(modelo, dict):  # Bagging k-fold models
            previsoes_agregadas = np.zeros((inputDados.shape[0],))
            total_estimadores = 0
            for fold, estimadores in modelo.items():
                for estimador in estimadores:
                    previsoes = estimador.predict(inputDados).flatten()
                    previsoes_agregadas += previsoes
                    total_estimadores += 1
            previsoes_agregadas /= total_estimadores
            resultado = int(round(previsoes_agregadas[0]))
        else:
            previsao = modelo.predict(inputDados)
            if isinstance(previsao, np.ndarray):
                previsao = previsao.flatten()[0]
            resultado = int(round(previsao))

        retorno[nome_modelo] = le.inverse_transform([resultado])[0]

    # Buscar no DataFrame
    resultado_df = data[data['nome'].str.upper() == nome.upper()]
    if not resultado_df.empty:
        resultado_df = resultado_df.iloc[0].to_dict()
    else:
        resultado_df = None

    return {'models': retorno, 'data': resultado_df}
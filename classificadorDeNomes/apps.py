from django.apps import AppConfig
import os
import json
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class ClassificadorDeNomesConfig(AppConfig):
    name = 'classificadorDeNomes'
    modelos = {}
    metrics = {}
    data = None
    le = LabelEncoder()

    def ready(self):
        print("Carregando modelos e dataset...")
        print("Pasta atual: ", os.getcwd())
        diretorio_modelos = './modelos_e_resultados/'
        caminho_data = './data/data.csv'

        # Carregar DataFrame
        self.data = pd.read_csv(caminho_data)
        print("DataFrame carregado com sucesso.")

        # Treinar Label Encoder manualmente
        if 'classificacao' in self.data.columns:
            self.le.fit(self.data['classificacao'])
            print("Label Encoder treinado com sucesso.")
        else:
            print("Coluna 'classificacao' não encontrada no DataFrame.")

        for nome_pasta in os.listdir(diretorio_modelos):
            print("Nome da pasta: ", nome_pasta)
            caminho_pasta = os.path.join(diretorio_modelos, nome_pasta)
            if os.path.isdir(caminho_pasta):
                # Carregar o modelo
                if nome_pasta == 'rede_neural':
                    caminho_modelo = os.path.join(caminho_pasta, 'modelo.h5')
                    self.modelos[nome_pasta] = load_model(caminho_modelo)
                elif nome_pasta == 'bagging':
                    modelos_bagging = []
                    for i in range(10):
                        caminho_modelo = os.path.join(caminho_pasta, f'modelo_{i}.keras')
                        modelos_bagging.append(load_model(caminho_modelo))
                    self.modelos[nome_pasta] = modelos_bagging
                elif nome_pasta == 'bagging_kfold':
                    modelos_bagging_kfold = {}
                    for fold in range(1, 6):
                        estimadores = []
                        for estimador in range(1, 11):
                            caminho_modelo = os.path.join(caminho_pasta, f'modelo_fold{fold}_estimador{estimador}.h5')
                            estimadores.append(load_model(caminho_modelo))
                        modelos_bagging_kfold[f'fold{fold}'] = estimadores
                    self.modelos[nome_pasta] = modelos_bagging_kfold
                else:
                    caminho_modelo = os.path.join(caminho_pasta, 'modelo.joblib')
                    self.modelos[nome_pasta] = load(caminho_modelo)

                # Carregar as métricas
                caminho_resultados = os.path.join(caminho_pasta, 'resultados.json')
                if os.path.exists(caminho_resultados):
                    with open(caminho_resultados, 'r') as f:
                        resultados = json.load(f)

                    if isinstance(resultados, dict):
                        classification_report = resultados.get('classification_report', {})
                        if isinstance(classification_report, dict):
                            if 'media_accuracia' in classification_report:
                                self.metrics[nome_pasta] = {
                                    'accuracy': classification_report.get('media_accuracia', 'N/A'),
                                    'precision': classification_report.get('precision', 'N/A'),
                                    'recall': classification_report.get('recall', 'N/A'),
                                    'f1': classification_report.get('f1-score', 'N/A')
                                }
                            elif 'macro avg' in classification_report:
                                self.metrics[nome_pasta] = {
                                    'accuracy': classification_report.get('accuracy', 'N/A'),
                                    'precision': classification_report.get('macro avg', {}).get('precision', 'N/A'),
                                    'recall': classification_report.get('macro avg', {}).get('recall', 'N/A'),
                                    'f1': classification_report.get('macro avg', {}).get('f1-score', 'N/A')
                                }
                            else:
                                self.metrics[nome_pasta] = {
                                    'accuracy': classification_report.get('accuracy', 'N/A'),
                                    'precision': classification_report.get('macro avg', {}).get('precision', 'N/A'),
                                    'recall': classification_report.get('macro avg', {}).get('recall', 'N/A'),
                                    'f1': classification_report.get('media_f1_score', 'N/A')
                                }
                        else:
                            self.metrics[nome_pasta] = {
                                'accuracy': 'N/A',
                                'precision': 'N/A',
                                'recall': 'N/A',
                                'f1': 'N/A'
                            }
                    else:
                        self.metrics[nome_pasta] = {
                            'accuracy': 'N/A',
                            'precision': 'N/A',
                            'recall': 'N/A',
                            'f1': 'N/A'
                        }
                else:
                    self.metrics[nome_pasta] = {
                        'accuracy': 'N/A',
                        'precision': 'N/A',
                        'recall': 'N/A',
                        'f1': 'N/A'
                    }

        print("Modelos e métricas carregados com sucesso.")
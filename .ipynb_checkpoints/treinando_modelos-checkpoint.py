#!/usr/bin/env python
# coding: utf-8

# # 1. Importação das bibliotecas 

# Aqui é feito a importação de todos os pacotes que serão utilizado nos estudos

# In[160]:


import os
import string
import unicodedata
import json
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold , RandomizedSearchCV , GridSearchCV , cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score , make_scorer, r2_score , f1_score , confusion_matrix
from scipy.stats import randint, uniform
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential , clone_model , load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier


# # 2.  Configuração das bibliotecas

# essa linha abaixo serve para resetar a largura das colunas na hora de exibir, em alguns cenarios estava tendo bug e tive que usar isso para adaptar

# In[161]:


pd.reset_option('display.max_colwidth')


# # 3. importação dos CSV e transformando em datasets pandas

# In[162]:


gruposDFnomeDasColunas = [
    "nome",
    "classificacao",
    "frequencia_feminina",
    "frequencia_masculina",
    "frequencia_total",
    "proporcao",
    "nomes_alternativos"    
]
gruposDF = pd.read_csv('grupos.csv', names=gruposDFnomeDasColunas, header=0)
gruposDF.head()


# In[163]:


nomesDSnomeDasColunas = [
    "nomes_alternativos",
    "classificacao",
    "primeiro_nome",
    "frequencia_feminina",
    "frequencia_masculina",
    "frequencia_total",
    "frequencia_grupo",
    "nome_grupo",
    "proporcao"
]
nomesDF= pd.read_csv("nomes.csv",names=nomesDSnomeDasColunas,header=0)
nomesDF.head()


# # 4. Limpeza de nulos e n/a

# nas limpezas dos dados nulos, foram verificados que os dados "NAN" na verdade, são dados que vazios, pois por exemplo, na linha AALINE, temos a frequencia_feminina de 66 e  frequencia_total de 66 também, então sobraria 0 para a frequencia_masculina

# In[164]:


gruposDF.fillna(0, inplace=True)
nomesDF.fillna(0, inplace=True)

gruposDF.drop_duplicates(inplace=True)
nomesDF.drop_duplicates(inplace=True)


# # 5. Criando novos dados 

# ## 5.1 Porcentagem de cada classe
# para melhor visualização da frequencia, é necessario a criação dos dados de porcentagem 

# In[165]:


gruposDF.head()


# In[166]:


# gruposDF.drop(columns=["nomes_alternativos"],inplace=True)
# nomesDF.drop(columns=["nomes_alternativos","frequencia_grupo","nome_grupo"],inplace=True)


# In[167]:


gruposDF["porcentagem_feminina"]  = 0
gruposDF["porcentagem_masculina"] = 0
nomesDF["porcentagem_feminina"] = 0
nomesDF["porcentagem_masculina"] = 0


# In[168]:


nomesDF.rename(columns=
               {"primeiro_nome": "nome"}
               ,inplace=True)


# In[169]:


gruposDF["porcentagem_masculina"] = round(gruposDF["frequencia_masculina"] / gruposDF["frequencia_total"], 7)
gruposDF["porcentagem_feminina"] =  round(gruposDF["frequencia_feminina"]  / gruposDF["frequencia_total"], 7)
    


# In[170]:


gruposDF.head()


# In[171]:


nomesDF["porcentagem_masculina"] = round(nomesDF["frequencia_masculina"] / nomesDF["frequencia_total"], 7)
nomesDF["porcentagem_feminina"] = round(nomesDF["frequencia_feminina"] / nomesDF["frequencia_total"], 7)
    


# In[172]:


nomesDF[(nomesDF["porcentagem_feminina"] == 0) & (nomesDF["porcentagem_masculina"] == 0)]


# In[173]:


gruposDF[(gruposDF["porcentagem_feminina"] == 0) & (gruposDF["porcentagem_masculina"] == 0)]


# In[174]:


nomesDF.set_index('nome', inplace=True)
gruposDF.set_index('nome', inplace=True)


# In[175]:


data = gruposDF.combine_first(nomesDF).reset_index()


# In[176]:


data.head()


# In[209]:


data.to_csv("data.csv")


# ## 5.2 Criando colunas binarias
# 
# Para alimentar os modelos, foi pensado a estrategia de criar colunas binarias para cada posição do nome , por exemplo, 
# 
# a letra 1 ( primeira letra) é igual a A ? ou seja , a coluna LETRA_1_A , e assim por diante 
# 

# In[177]:


def processar_dataset(data, max_posicoes=20):
    alfabeto = string.ascii_uppercase
    novas_colunas = [f"LETRA_{i}_{letra}" for i in range(1, max_posicoes + 1) for letra in alfabeto]
    novas_colunas_df = pd.DataFrame(0, index=data.index, columns=novas_colunas)
    data = pd.concat([data, novas_colunas_df], axis=1)
    for index, row in data.iterrows():
        nome = row['nome'].upper()
        for posicao, letra in enumerate(nome):
            if posicao < max_posicoes and letra in alfabeto:
                coluna = f"LETRA_{posicao + 1}_{letra}"
                if coluna in data.columns:
                    data.at[index, coluna] = 1

    return data
data = processar_dataset(data)
data


# ## 6. Criando e executando label encoder
# o label Encoder faz a função de codificador , para transformar os dados qualitativos em numeros, para assim que a maquina prever , o label encoder poder utilizar a função de inverse transform e trazer a real classe da previsão 

# In[178]:


le = LabelEncoder()
le = le.fit(data['classificacao'])
data['classificacao'] = le.fit_transform(data['classificacao'])
data.head()


# # separando dados de treino e teste
# separa as colunas de input e a coluna de target (Classificacao) 

# In[179]:


X = data.drop(columns=["nome",	"classificacao"	,"frequencia_feminina",	"frequencia_masculina"	,"frequencia_total"	,"proporcao"	,"porcentagem_feminina"	,"porcentagem_masculina"])
y = data['classificacao']


# # 7. Criando Funções Auxiliares

# ## 7.1 Criando função de input de dados
# a função de input de dados recebe um nome, e tranforma em colunas binarias, assim como os dados de treino, para encaixar com o shape dos dados

# In[180]:


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


# ## 7.2 Criando Função para plotar Matriz de Confusão 
# 
# essa vai ser uma função auxiliar que vai nos ajudar a evitar repetição de codigo

# In[181]:


def plotar_matriz_confusao(cm, classes=['F', 'M'], title='Matriz de Confusão', cmap='Blues'):
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, cbar=False)
    
    # Adicionar anotações de TP, FP, FN, TN
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                annotation = f'{cm[i, j]} (TP)'  # Verdadeiro Positivo
            elif i > j:
                annotation = f'{cm[i, j]} (FN)'  # Falso Negativo
            else:
                annotation = f'{cm[i, j]} (FP)'  # Falso Positivo
            ax.text(j + 0.5, i + 0.5, annotation, color='black', ha='center', va='center')

    plt.xlabel('Previsto')
    plt.ylabel('Valor Real')
    plt.title(title)
    plt.show()


# ## 7.3 Criando Função Auxiliar  para criar diretorios dinamicos 
# 
# isso ajudara a organizar as pastas

# In[182]:


def criar_diretorio(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)


# ## 7.4 Criando Objeto Auxiliar para manter todos os modelos 
# 
# esse objeto servirar para manter todos os modelos salvos em um lugar para testar de forma manual futuramente 

# In[183]:


modelos = {}


# ## 7.5 Criação Função Auxiliar para gerar relatorios de classificação
# essa função auxiliar ajuda a evitar repetição de codigo toda vez que for gerar um relatorio de classificação

# In[184]:


def gerar_relatorio_classificacao(Y_teste, previsoes, le=le):
    Y_teste_decodificado = le.inverse_transform(Y_teste)
    previsoes_decodificadas = le.inverse_transform(previsoes)
    class_report = classification_report(Y_teste_decodificado, previsoes_decodificadas,output_dict=True)
    print(class_report)
    print(f'Acuracia {class_report["accuracy"]}')
    return class_report, Y_teste_decodificado, previsoes_decodificadas


# ## 7.6 Criação de função para treinar|Carregar modelos "simples"
# essa é a principal das funções auxiliares, ela é responsavel por treinar ou carregar os dados dos modelos do scikit learn e consumir as outras funções auxiliares para trazer os dados 

# In[185]:


def treinar_e_avaliar_modelo(modelo, X, y, model_name, classes=['F', 'M'],modelos=modelos):
    base_path = f'./modelos_e_resultados/{model_name}/'
    model_path = os.path.join(base_path, 'modelo.joblib')
    results_path = os.path.join(base_path, 'resultados.json')
    
    # Criar diretório para o modelo se não existir
    criar_diretorio(base_path)
    
    if os.path.exists(model_path) and os.path.exists(results_path):
        # Carregar modelo e resultados
        modelo = load(model_path)
        with open(results_path, 'r') as f:
            resultados = json.load(f)
        
        # Exibir os resultados carregados
        print(f"Resultados carregados:")
        print(f"F1-scores de cada fold: {resultados['f1_scores']}")
        print(f"Média do F1-score: {resultados['media_f1_score']}")
        print("Classification Report:")
        print(resultados['classification_report'])
        
        # Exibir a matriz de confusão carregada
        cm = np.array(resultados['confusion_matrix'])
        plotar_matriz_confusao(cm, classes=classes)
    else:
        # Treinar o modelo e calcular os resultados
        X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scorer = make_scorer(f1_score, average='macro')
        f1_scores = cross_val_score(modelo, X_treino, Y_treino, cv=kf, scoring=f1_scorer)
        
        print(f'F1-scores de cada fold: {f1_scores}')
        print(f'Média do F1-score: {f1_scores.mean()}')
        
        
        modelo.fit(X_treino, Y_treino)
        dump(modelo, model_path)
        
        previsoes = modelo.predict(X_teste)
        acc = accuracy_score(Y_teste, previsoes)
        print(f'{model_name} - Acurácia: {acc}')
        
   
        class_report, Y_teste_decodificado, previsoes_decodificadas = gerar_relatorio_classificacao(Y_teste, previsoes, le)
        
    
        cm = confusion_matrix(Y_teste_decodificado, previsoes_decodificadas)
        plotar_matriz_confusao(cm, classes=classes)
        modelos[model_name] = modelo
   
        resultados = {
            "f1_scores": f1_scores.tolist(),
            "media_f1_score": f1_scores.mean(),
            "classification_report": class_report,
            "confusion_matrix": cm.tolist()
        }
        with open(results_path, 'w') as f:
            json.dump(resultados, f)
    
    return modelo


# ## 7.8 Criação da função auxiliar para remover acentos de nomes

# In[186]:


def remove_acentos(string):
    nfkd = unicodedata.normalize('NFKD', string)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ## 7.9 Criação da função Auxiliar para buscar nome no dataframe

# In[187]:


def busca_no_dataframe(nome):
    nome = nome.upper()
    if nome in data['nome'].values:        
        copyDT = data.loc[data['nome'] == nome, [
            'nome', 'classificacao', 'frequencia_feminina', 'nome_grupo',
            'frequencia_masculina', 'frequencia_total',
            'proporcao', 'porcentagem_feminina',
            'porcentagem_masculina', 'nomes_alternativos'  # Incluindo a nova coluna
        ]].copy()

        
        retornoDataFrame = copyDT.to_dict(orient='records')[0] 
        

        if 'nomes_alternativos' in retornoDataFrame and pd.notna(retornoDataFrame['nomes_alternativos']):
            retornoDataFrame['nomes_alternativos'] = retornoDataFrame['nomes_alternativos'].split('|')
        else:
            retornoDataFrame['nomes_alternativos'] = []  
        

        retornoDataFrame['status'] = '200'
        retornoDataFrame['classificacao'] = le.inverse_transform([retornoDataFrame['classificacao']])[0]

        return retornoDataFrame 
    else:
        return {"status": "400"}


resultado = busca_no_dataframe('augusto')
print(resultado)


# In[188]:


data['nomes_alternativos']


# ## 7.10  Criando função auxiliar para desconverter de classeEncode para classeOriginal

# In[189]:


def desconverteEncoding(resultado, le=le):
    return le.inverse_transform([resultado])[0]
    


# - - - 

# # 8. Regressão Logistica

# In[190]:


log_reg = LogisticRegression(max_iter=1000)
log_reg = treinar_e_avaliar_modelo(log_reg, X, y, "regressao_logistica")
modelos["regressao_logistica"] = log_reg
log_reg


# Com o modelo de regressao logistica foi obtido o resultado medio de F1-score de ~0.85, que significa que temos um um desempenho constante mesmo com diferentes conjuntos de dados . 
# 
# Já o classification_report nos da a informação que a precisão para ambas as classes é de 0.87, o que é um otimo sinal que não há um overfit pois é um valor bem proximo do F1 score
# 
# a matriz de confusão também revela que há valores muito baixo de Falso Negativos e Falso Positivos comparados ao Verdadeiro Positivo e ao Verdadeiro Negativo

# # 9. KNN

# In[191]:


knn = KNeighborsClassifier(n_neighbors=5)
knn = treinar_e_avaliar_modelo(knn, X, y, "knn")
modelos['knn'] = knn
knn


# No resultado do knn vemos que o f1 score dele foi um pouco maior que o anterior (regressao logistica ) porem os falsos positivos e falsos negativos foram bem maiores 

# # 10. Naive bayes | GaussianNB

# In[192]:


naive_bayes = GaussianNB()
naive_bayes = treinar_e_avaliar_modelo(naive_bayes, X, y, "naive_bayes")
modelos['naive_bayes'] = naive_bayes
naive_bayes


# o Modelo de naive bayes foi bem inferior em questão do F1-Score em relação aos dois ultimos modelos  e tivemos um crescente muito grande em questão de falsos negativos

# # 11. Random forest

# In[193]:


random_forest = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=42)
random_forest = treinar_e_avaliar_modelo(random_forest, X, y, "random_forest")
modelos['random_forest'] = random_forest
random_forest


# o modelo de random forest foi o melhor de de F1-score até agora, e obteve um falso negativo baixo , e obtemos valores medianos bem baixo tambem de falso postivo

# # 12. rede neural

# ## 12.1 Criando função especifica para o tensorflow/keras

# In[194]:


def criar_modelo_neural(input_dim):
    modelo = Sequential([
        Dense(160, input_dim=input_dim, activation='relu'), 
        Dropout(0.2),
        Dense(80, activation='relu'), 
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


# In[195]:


def treinar_e_avaliar_modelo_keras(modelo, X, y, model_name, classes=['F', 'M'], epochs=50, batch_size=32, n_splits=5):

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    X = np.array(X)
    y = np.array(y)


    base_path = f'./modelos_e_resultados/{model_name}/'
    model_path = os.path.join(base_path, 'modelo.h5')
    results_path = os.path.join(base_path, 'resultados.json')

  
    os.makedirs(base_path, exist_ok=True)

    if os.path.exists(model_path) and os.path.exists(results_path):
      
        modelo = load_model(model_path)
        with open(results_path, 'r') as f:
            resultados = json.load(f)

        print("Resultados carregados:")
        print(f"Acurácias de cada fold: {resultados['accuracies']}")
        print(f"Média da Acurácia: {resultados['media_accuracia']}")
        print("Classification Report:")
        print(resultados['classification_report'])

      
        cm = np.array(resultados['confusion_matrix'])
        plotar_matriz_confusao(cm, classes=classes)
    else:
    
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

  
        historicos = []
        accuracies = []
        melhor_acuracia = 0
        melhor_modelo = None

     
        for fold, (train_index, val_index) in enumerate(kf.split(X, y), 1):
            print(f"\nTreinando fold {fold}")

            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

           
            model_clone = clone_model(modelo)
            model_clone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


            history = model_clone.fit(X_train, y_train,
                                      validation_data=(X_val, y_val),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      verbose=1)

       
            _, accuracy = model_clone.evaluate(X_val, y_val, verbose=0)
            accuracies.append(accuracy)
            historicos.append(history.history)

            print(f"Acurácia do fold {fold}: {accuracy}")

      
            if accuracy > melhor_acuracia:
                melhor_acuracia = accuracy
                melhor_modelo = model_clone


        melhor_modelo.save(model_path)


        y_pred = melhor_modelo.predict(X)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()


        class_report = classification_report(y, y_pred_classes, target_names=classes)
        cm = confusion_matrix(y, y_pred_classes)


        plotar_matriz_confusao(cm, classes=classes)


        resultados = {
            "accuracies": accuracies,
            "media_accuracia": np.mean(accuracies),
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "historicos": historicos
        }

        with open(results_path, 'w') as f:
            json.dump(resultados, f)

        print("\nResultados finais:")
        print(f"Acurácias de cada fold: {accuracies}")
        print(f"Média da Acurácia: {np.mean(accuracies)}")
        print("\nClassification Report:")
        print(class_report)

        modelos[model_name] = modelo

    return modelo



# ## 12.2 Modelo de rede neural

# In[196]:


rede = criar_modelo_neural(X.shape[1])
rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rede = treinar_e_avaliar_modelo_keras(rede,X,y,'rede_neural')
modelos['rede_neural'] = rede


# nesse modelo utilizando rede neurais , é possivel ver que a acuracia media de cada fold é bme alta (95%) e que no melhor modelo chega a bater um f1-score de 0.99% , o numero de falsos positivos é maior que alguns modelos e tem o  menor nuemro de falso negativo até agora

# # 13. Algoritmos de conjunto

# ## 13.1  Bagging com rede neural

# In[197]:


def previsao_bagging(modelos, X):

    if len(modelos) == 0:
        raise ValueError("A lista de modelos está vazia. Certifique-se de que os modelos estão treinados e adicionados à lista.")
    

    previsoes_agregadas = np.zeros(X.shape[0])
    

    for modelo in modelos:
        previsoes = modelo.predict(X).flatten()
        previsoes_agregadas += previsoes

    previsoes_agregadas /= len(modelos)
    

    previsoes_finais = (previsoes_agregadas > 0.5).astype(int)
    
    return previsoes_finais



def criar_modelo_neural(input_dim):
    modelo = Sequential([
        Dense(160, input_dim=input_dim, activation='relu'), 
        Dropout(0.2),
        Dense(80, activation='relu'), 
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


def bagging_neural_network(X, y, model_name, classes=['F', 'M'], n_estimators=10, epochs=50, batch_size=32):
    base_path = f'./modelos_e_resultados/{model_name}/'
    model_paths = [os.path.join(base_path, f'modelo_{i}.keras') for i in range(n_estimators)]
    results_path = os.path.join(base_path, 'resultados.json')

    criar_diretorio(base_path)

    missing_models = [path for path in model_paths if not os.path.exists(path)]

    if not missing_models and os.path.exists(results_path):
        # Carregar modelos e resultados
        modelos = [load_model(path) for path in model_paths]
        with open(results_path, 'r') as f:
            resultados = json.load(f)

        # Exibir os resultados carregados
        print(f"Resultados carregados:")
        print(f"F1-scores de cada fold: {resultados['f1_scores']}")
        print(f"Média do F1-score: {resultados['media_f1_score']}")
        print("Classification Report:")
        print(resultados['classification_report'])

        # Exibir a matriz de confusão carregada
        cm = np.array(resultados['confusion_matrix'])
        plotar_matriz_confusao(cm, classes=classes)
    else:
        if missing_models:
            print("Os seguintes modelos estão faltando e serão treinados novamente:")
            for path in missing_models:
                print(path)
        
        # Garantir que X e y são arrays NumPy
        X = np.array(X)
        y = np.array(y)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
        
        modelos = []
        previsoes_agregadas = np.zeros((X_teste.shape[0],))

        # Criando múltiplos modelos
        for i in range(n_estimators):
            print(f"Treinando modelo {i+1}/{n_estimators}")

            # Amostragem com reposição
            indices = np.random.choice(X_treino.shape[0], X_treino.shape[0], replace=True)
            X_bootstrap = X_treino[indices]
            y_bootstrap = y_treino[indices]

            # Criar e treinar o modelo
            modelo = criar_modelo(X.shape[1])
            checkpoint = ModelCheckpoint(model_paths[i], save_best_only=True, monitor='val_loss', mode='min')
            modelo.fit(X_bootstrap, y_bootstrap, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint])
            modelos.append(modelo)

            # Previsão
            previsoes = modelo.predict(X_teste).flatten()
            previsoes_agregadas += previsoes

        # Média das previsões
        previsoes_agregadas /= n_estimators

        # Transformar previsões em classes
        previsoes_finais = (previsoes_agregadas > 0.5).astype(int)

        # Calcular métricas
        acc = accuracy_score(y_teste, previsoes_finais)
        f1 = f1_score(y_teste, previsoes_finais, average='macro')
        print(f'Acurácia: {acc:.2f}')
        print(f'F1-score: {f1:.2f}')

        class_report, Y_teste_decodificado, previsoes_decodificadas = gerar_relatorio_classificacao(y_teste, previsoes_finais, le)

        cm = confusion_matrix(Y_teste_decodificado, previsoes_decodificadas)
        plotar_matriz_confusao(cm, classes=classes)

        resultados = {
            "f1_scores": [f1],
            "media_f1_score": f1,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist()
        }
        with open(results_path, 'w') as f:
            json.dump(resultados, f)

    return modelos



modelos_bagging = bagging_neural_network(X, y, model_name='bagging', classes=['F', 'M'])


# o modelo de baggins de rede neurais demonstra ter mais falsos positivos e negativos do que a propria rede neural isolada, porém apresenta um f1 score bom

# In[198]:


nome_exemplo = 'augusto'
dados_input = preparar_input_para_modelo(nome_exemplo)
if len(dados_input.shape) == 1:
    dados_input = dados_input.reshape(1, -1)
try:
    previsoes_finais = previsao_bagging(modelos_bagging, dados_input)
    previsoes_finais = le.inverse_transform(previsoes_finais)
    print(f"Previsão final para {nome_exemplo}: {previsoes_finais[0]}")

except Exception as e:
    print(f"Erro: {e}")


# ## 13.2 Boosting

# In[199]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# In[200]:


def treinar_e_avaliar_adaboost_logreg(X, y, model_name, classes=['F', 'M']):
    base_path = f'./modelos_e_resultados/{model_name}/'
    model_path = os.path.join(base_path, 'modelo.joblib')
    results_path = os.path.join(base_path, 'resultados.json')

    criar_diretorio(base_path)

    if os.path.exists(model_path) and os.path.exists(results_path):
        adaboost_custom = load(model_path)
        with open(results_path, 'r') as f:
            resultados = json.load(f)


        print(f"Resultados carregados:")
        print(f"Acurácia: {resultados['accuracy']:.2f}")
        print(resultados['classification_report'])


        cm = np.array(resultados['confusion_matrix'])
        plotar_matriz_confusao(cm, classes=classes)
    else:

        X = np.array(X)
        y = np.array(y)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        estimator = LogisticRegression(max_iter=1000)


        adaboost_custom = AdaBoostClassifier(estimator=estimator, n_estimators=50, random_state=42)


        adaboost_custom.fit(X_train, y_train)


        y_pred = adaboost_custom.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {accuracy:.2f}")

        class_report = gerar_relatorio_classificacao(y_test, y_pred)


        cm = confusion_matrix(y_test, y_pred)
        plotar_matriz_confusao(cm, classes=classes)


        dump(adaboost_custom, model_path)
        print(f'class report type {type(class_report[1])}')
    
        resultados = {
            "accuracy": accuracy,
            "classification_report": class_report[0],
            "confusion_matrix": cm.tolist()
        }
        with open(results_path, 'w') as f:
            json.dump(resultados, f)

    return adaboost_custom


# In[201]:


adaboost_model = treinar_e_avaliar_adaboost_logreg(X, y, 'boosting_log_reg')
adaboost_model


# o modelo de adaboost com regressao logistica apresentou uma acuracia relativamente baixa comparado aos outros modelos , valores altos de falso positivo e negativos , e um f1 score baixo também

# ## 13.4 Stacking

# In[202]:


def treinar_e_avaliar_stacking(X, y, model_name, classes=['Class 0', 'Class 1']):
    base_path = f'./modelos_e_resultados/{model_name}/'
    model_path = os.path.join(base_path, 'modelo.joblib')
    results_path = os.path.join(base_path, 'resultados.json')

    criar_diretorio(base_path)

    if os.path.exists(model_path) and os.path.exists(results_path):

        stacking_model = load(model_path)
        with open(results_path, 'r') as f:
            resultados = json.load(f)


        print(f"Resultados carregados:")
        print(f"Acurácia: {resultados['accuracy']:.2f}")
        print(resultados['classification_report'])


        cm = np.array(resultados['confusion_matrix'])
        plotar_matriz_confusao(cm, classes=classes)
    else:

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

 
        estimators = [
            ('dt', DecisionTreeClassifier(max_depth=3)),
            ('rf', RandomForestClassifier(n_estimators=100))
        ]


        meta_model = LogisticRegression(max_iter=1000)


        stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model)


        stacking_model.fit(X_train, y_train)


        y_pred = stacking_model.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {accuracy:.2f}")

        class_report = gerar_relatorio_classificacao(y_test, y_pred)


        cm = confusion_matrix(y_test, y_pred)
        plotar_matriz_confusao(cm, classes=classes)

        dump(stacking_model, model_path)

        resultados = {
            "accuracy": accuracy,
            "classification_report": class_report[0],
            "confusion_matrix": cm.tolist()
        }
        print(resultados)
        with open(results_path, 'w') as f:
            json.dump(resultados, f)

    return stacking_model


# In[203]:


stacking_model = treinar_e_avaliar_stacking(X, y, 'stacking_model')
stacking_model 


# o algoritmo de staciking de apresentou resultado semelhante ao de boosting, também com valores de acuracia e f1 baixos e classificações erradas 

# In[204]:


def preveEmTodosModelos(nome):
    retorno = {}
    
    nome = remove_acentos(nome)
    inputDados = preparar_input_para_modelo(nome)
    print("Convertendo input de dados")
    print(inputDados)
    print(inputDados.shape)
    print("-------------------")
    
    if not ('log_reg' in globals()):
        raise ValueError("o modelo de regressão logistica não foi carregado.")
    else:
        resultadoLogReg = desconverteEncoding(log_reg.predict(inputDados))
        retorno['LogReg'] = resultadoLogReg
        print(f'Resultado LogReg : {resultadoLogReg}')
        print("--------------------------")

    if not ('knn' in globals()):
        raise ValueError("o modelo de KNN não foi carregado.")
    else:
        resultadoKnn = desconverteEncoding(knn.predict(inputDados))
        print(f'Resultado KNN : {resultadoKnn}')
        print("--------------------------")



    if not ('naive_bayes' in globals()):
        raise ValueError("o modelo de naive_bayes não foi carregado.")
    else:
        resultadoNaive_bayes = desconverteEncoding(naive_bayes.predict(inputDados))
        retorno['naive_bayes'] = resultadoNaive_bayes
        print(f'Resultado naive_bayes : {resultadoNaive_bayes}')
        print("--------------------------")


    
    if not ('random_forest' in globals()):
        raise ValueError("o modelo de random_forest não foi carregado.")
    else:
        resultadoRandom_forest = desconverteEncoding(random_forest.predict(inputDados))
        retorno['random_forest'] = resultadoRandom_forest
        print(f'Resultado random_forest : {resultadoRandom_forest}')
        print("--------------------------")


    if not ('rede' in globals()):
        raise ValueError("o modelo de redeNeural não foi carregado.")
    else:
        resultadoRede = rede.predict(inputDados)
        resultadoRede = desconverteEncoding(int(resultadoRede[0]))
        print(resultadoRede)
        
        print(f'Resultado redeNeural {resultadoRede}')
        print("--------------------------")




    if not ('modelos_bagging' in globals()):
        raise ValueError("o modelo de baggin não foi carregado.")
    else:
        resultadoBagging = desconverteEncoding(previsao_bagging(modelos_bagging, dados_input))
        retorno['modelos_bagging'] = resultadoBagging
        print(f'Resultado bagging {resultadoBagging}')
        print("--------------------------")


    if not ('adaboost_model' in globals()):
        raise ValueError("o modelo de adaboost_model não foi carregado.")
    else:
        resultadoBoosting = desconverteEncoding(adaboost_model.predict(inputDados))
        retorno['resultadoBoosting'] = resultadoBoosting
        print(f'Resultado BoostingAda {resultadoBoosting}')
        print("--------------------------")

    if not ('stacking_model' in globals()):
        raise ValueError("o modelo de stacking_model não foi carregado.")
    else:
        resultadoStacking = desconverteEncoding(stacking_model.predict(inputDados))
        retorno['stacking_model'] = resultadoStacking
        print(f'Resultado Stacking {resultadoStacking}')
        print("--------------------------")

    if not ('data' in globals()):
        raise ValueError("o dataframe nao foi carregado.")
    else:
        resultadoDataFrame = busca_no_dataframe(nome)
        retorno['data'] = resultadoDataFrame
        print(f'Resultado Stacking {resultadoDataFrame}')
        print("--------------------------")

    return retorno


# In[205]:


def preveEmTodosModelos(nome):
    retorno = {}
    
    nome = remove_acentos(nome)
    inputDados = preparar_input_para_modelo(nome)

    
    if not ('log_reg' in globals()):
        raise ValueError("o modelo de regressão logistica não foi carregado.")
    else:
        resultadoLogReg = desconverteEncoding(log_reg.predict(inputDados))
        retorno['LogReg'] = resultadoLogReg

    if not ('knn' in globals()):
        raise ValueError("o modelo de KNN não foi carregado.")
    else:
        resultadoKnn = desconverteEncoding(knn.predict(inputDados))


    if not ('naive_bayes' in globals()):
        raise ValueError("o modelo de naive_bayes não foi carregado.")
    else:
        resultadoNaive_bayes = desconverteEncoding(naive_bayes.predict(inputDados))
        retorno['naive_bayes'] = resultadoNaive_bayes



    
    if not ('random_forest' in globals()):
        raise ValueError("o modelo de random_forest não foi carregado.")
    else:
        resultadoRandom_forest = desconverteEncoding(random_forest.predict(inputDados))
        retorno['random_forest'] = resultadoRandom_forest



    if not ('rede' in globals()):
        raise ValueError("o modelo de redeNeural não foi carregado.")
    else:
        resultadoRede = rede.predict(inputDados)
        resultadoRede = desconverteEncoding(int(resultadoRede[0]))



    if not ('modelos_bagging' in globals()):
        raise ValueError("o modelo de baggin não foi carregado.")
    else:
        resultadoBagging = desconverteEncoding(previsao_bagging(modelos_bagging, dados_input))
        retorno['modelos_bagging'] = resultadoBagging



    if not ('adaboost_model' in globals()):
        raise ValueError("o modelo de adaboost_model não foi carregado.")
    else:
        resultadoBoosting = desconverteEncoding(adaboost_model.predict(inputDados))
        retorno['resultadoBoosting'] = resultadoBoosting


    if not ('stacking_model' in globals()):
        raise ValueError("o modelo de stacking_model não foi carregado.")
    else:
        resultadoStacking = desconverteEncoding(stacking_model.predict(inputDados))
        retorno['stacking_model'] = resultadoStacking


    if not ('data' in globals()):
        raise ValueError("o dataframe nao foi carregado.")
    else:
        resultadoDataFrame = busca_no_dataframe(nome)
        retorno['data'] = resultadoDataFrame

    return retorno


# In[206]:


preveEmTodosModelos('ariel')


# In[ ]:





# In[ ]:





# In[ ]:





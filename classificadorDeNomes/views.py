# classificadorDeNomes/views.py
from django.shortcuts import render, redirect
from .helpers import preveEmTodosModelos
from django.apps import apps

def home(request):
    result = None

    if request.method == 'POST':
        nome = request.POST.get('nome')
        if nome:
            # Processa a entrada
            result = preveEmTodosModelos(nome)
            # Armazena o resultado na sessão para ser acessado após o redirecionamento
            request.session['result'] = result
            return redirect('home')  # Redireciona para evitar reenvio

    # Obtém o resultado da sessão após o redirecionamento
    result = request.session.pop('result', None)

    # Obter as métricas dos algoritmos da configuração
    config = apps.get_app_config('classificadorDeNomes')
    metrics = config.metrics

    # Informações dos algoritmos
    algorithms = [
        {
            'name': 'Regressão Logística',
            'class': 'logReg',
            'image': 'image/logistic-regression.png',
            'description': 'Estima a probabilidade de uma classificação binária, utilizando a função logística (sigmoide) para transformar a saída de uma regressão linear em um valor de probabilidade entre 0 e 1.',
            'metrics': metrics.get('regressao_logistica', {})
        },
        {
            'name': 'Random Forest',
            'class': 'randForest',
            'image': 'image/random_forest.png',
            'description': 'Cria um conjunto de árvores de decisão aleatórias, cada uma delas fornecendo uma classificação. A classificação final é determinada pela maioria das classificações.',
            'metrics': metrics.get('random_forest', {})
        },
        {
            'name': 'Naive Bayes',
            'class': 'naiveBayes',
            'image': 'image/bayes-theorem.png',
            'description': 'É um classificador probabilístico baseado no teorema de Bayes, que assume que a presença de uma característica particular em uma classe não está relacionada com a presença de qualquer outra característica.',
            'metrics': metrics.get('naive_bayes', {})
        },
        {
            'name': 'Rede Neural',
            'class': 'redeNeural',
            'image': 'image/deep-learning.png',
            'description': 'É um modelo de aprendizado de máquina inspirado na estrutura do cérebro humano. Ele consiste em várias camadas de neurônios artificiais, que são responsáveis por aprender e transformar os dados de entrada em uma saída.',
            'metrics': metrics.get('rede_neural', {})
        },
        {
            'name': 'Boosting',
            'class': 'boosting',
            'image': 'image/startup.png',
            'description': 'É uma técnica de aprendizado de máquina que combina vários modelos de aprendizado de máquina fracos para criar um modelo forte.',
            'metrics': metrics.get('boosting_log_reg', {})
        },
        {
            'name': 'Stacking',
            'class': 'stacking',
            'image': 'image/stacking.png',
            'description': 'É uma técnica de aprendizado de máquina que combina vários modelos de aprendizado de máquina para melhorar a precisão do modelo.',
            'metrics': metrics.get('stacking_model', {})
        },
        {
            'name': 'Bagging',
            'class': 'bagging',
            'image': 'image/bagging.png',
            'description': 'É uma técnica de aprendizado de máquina que combina vários modelos de aprendizado de máquina para melhorar a precisão do modelo.',
            'metrics': metrics.get('bagging_kfold', {})
        }
    ]

    return render(request, 'home.html', {'result': result, 'algorithms': algorithms})
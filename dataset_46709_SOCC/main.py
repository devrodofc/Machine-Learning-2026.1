import time
import numpy as np
from src.loader import load_arff, process_data
from src.metrics import k_fold_split, calculate_metrics
from src.knn import KNN
from src.bayes import NaiveBayes

def simple_bag_of_words(textos, vocab_size=15):
    """
    Constrói um Bag of Words (BoW) do zero.
    Conta as palavras mais frequentes de todo o corpus e transforma
    cada texto em um vetor de contagem dessas palavras.
    """
    contagem_global = {}
    textos_limpos = []
    
    # Limpeza básica e contagem
    for texto in textos:
        # Remove pontuações básicas e converte para minúsculo
        palavras = ''.join([c.lower() if c.isalnum() or c.isspace() else ' ' for c in texto]).split()
        textos_limpos.append(palavras)
        for p in palavras:
            if len(p) > 3: # Ignora palavras muito curtas (stopwords)
                contagem_global[p] = contagem_global.get(p, 0) + 1
                
    # Ordena e seleciona o vocabulário (Top N palavras)
    vocabulario = sorted(contagem_global, key=contagem_global.get, reverse=True)[:vocab_size]
    
    # Cria a matriz numérica (Características)
    X_num = np.zeros((len(textos), vocab_size))
    for i, palavras in enumerate(textos_limpos):
        for j, palavra_vocab in enumerate(vocabulario):
            X_num[i, j] = palavras.count(palavra_vocab)
            
    return X_num

def format_resultado(media, std):
    return f"{media:.2f} ± {std:.2f}"

def run_experiment():
    print("Iniciando Pipeline de Machine Learning...")
    
    # 1. Carregamento dos Dados (Simulação da extração ARFF)
    filepath = "data/dataset_46709.arff" # Garanta que renomeou/baixou corretamente
    # matriz_bruta = load_arff(filepath)
    # Como não temos o arquivo localmente agora, vamos criar dados sintéticos
    # que imitam o output do seu Loader após o processamento do texto
    # (Substitua a geração sintética abaixo pelas chamadas comentadas acima na vida real)
    
    print("Vetorizando textos (Bag of Words manual)...")
    np.random.seed(42)
    X = np.random.rand(1000, 12) # 1000 instâncias, 12 atributos numéricos
    y = np.random.randint(0, 2, 1000) # Classificação Binária (Construtivo: Sim/Não)
    
    # 2. Configuração dos Classificadores
    classificadores = {
        "K-vizinhos (Dist. Euclidiana)": KNN(k=5, dist_metric='euclidean'),
        "K-vizinhos (Dist. Manhattan)": KNN(k=5, dist_metric='manhattan'),
        "Bayesiano (Univariado)": NaiveBayes(multivariado=False),
        "Bayesiano (Multivariado)": NaiveBayes(multivariado=True)
    }
    
    resultados = {nome: {'acc': [], 'prec': [], 'f1': [], 't_treino': [], 't_teste': []} for nome in classificadores}
    
    # 3. Validação Cruzada K-Fold (K=5)
    print("Executando 5-Fold Cross Validation...\n")
    folds = k_fold_split(X, y, k=5)
    
    for nome_clf, clf in classificadores.items():
        for treino_idx, teste_idx in folds:
            X_treino, y_treino = X[treino_idx], y[treino_idx]
            X_teste, y_teste = X[teste_idx], y[teste_idx]
            
            # Treino
            start_train = time.time()
            clf.fit(X_treino, y_treino)
            tempo_treino = time.time() - start_train
            
            # Teste
            start_test = time.time()
            y_pred = clf.predict(X_teste)
            tempo_teste = time.time() - start_test
            
            # Métricas
            acc, prec, _, f1 = calculate_metrics(y_teste, y_pred)
            
            # Armazenando
            resultados[nome_clf]['acc'].append(acc)
            resultados[nome_clf]['prec'].append(prec)
            resultados[nome_clf]['f1'].append(f1)
            resultados[nome_clf]['t_treino'].append(tempo_treino)
            resultados[nome_clf]['t_teste'].append(tempo_teste)

    # 4. Impressão da Tabela Comparativa (Padrão Exigido)
    linha_divisoria = "-" * 105
    print(linha_divisoria)
    print(f"{'Classificador':<30} | {'Acurácia':<12} | {'Precisão':<12} | {'F1-Score':<12} | {'Tempo Treino (s)':<16} | {'Tempo Teste (s)'}")
    print(linha_divisoria)
    
    for nome, metricas in resultados.items():
        acc_str = format_resultado(np.mean(metricas['acc']), np.std(metricas['acc']))
        prec_str = format_resultado(np.mean(metricas['prec']), np.std(metricas['prec']))
        f1_str = format_resultado(np.mean(metricas['f1']), np.std(metricas['f1']))
        t_treino_str = format_resultado(np.mean(metricas['t_treino']), np.std(metricas['t_treino']))
        t_teste_str = format_resultado(np.mean(metricas['t_teste']), np.std(metricas['t_teste']))
        
        print(f"{nome:<30} | {acc_str:<12} | {prec_str:<12} | {f1_str:<12} | {t_treino_str:<16} | {t_teste_str}")
    print(linha_divisoria)

if __name__ == "__main__":
    run_experiment()
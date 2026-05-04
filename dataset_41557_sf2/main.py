import time
import numpy as np
from src.loader import load_solar_flare_regression
from src.metrics import k_fold_split, r2_score_manual, r2_ajustado_manual
from src.regressors import KNNRegressor, LinearRegressionMultipla

def format_resultado(media, std):
    """Gera a string formatada de Média ± Desvio Padrão."""
    return f"{media:.4f} ± {std:.4f}"

def run_regression_experiment():
    print("Iniciando Pipeline de Machine Learning (REGRESSÃO)...")
    
    # 1. Ingestão e Processamento dos Dados
    print("Carregando e aplicando One-Hot Encoding no dataset de Erupções Solares...")
    filepath = "data/file20381d055179.arff"
    
    # X contém a matriz com One-Hot Encoding, y contém os valores contínuos de explosões
    X, y = load_solar_flare_regression(filepath)
    
    n_amostras, p_atributos = X.shape
    print(f"Dataset carregado com sucesso: {n_amostras} amostras e {p_atributos} atributos independentes.\n")
    
    # 2. Instanciação dos Modelos Matemáticos
    regressores = {
        "K-vizinhos (Dist. Euclidiana)": KNNRegressor(k=5, dist_metric='euclidean'),
        "K-vizinhos (Dist. Manhattan)": KNNRegressor(k=5, dist_metric='manhattan'),
        "Regressão Linear Múltipla": LinearRegressionMultipla()
    }
    
    # Estrutura para armazenar o histórico do K-Fold
    resultados = {nome: {'r2': [], 'r2_adj': [], 't_treino': [], 't_teste': []} for nome in regressores}
    
    # 3. Validação Cruzada (K-Fold = 5)
    print("Executando 5-Fold Cross Validation...\n")
    folds = k_fold_split(X, y, k=5)
    
    for nome_reg, reg in regressores.items():
        for treino_idx, teste_idx in folds:
            # Fatiamento (Slicing) dos dados usando os índices gerados
            X_treino, y_treino = X[treino_idx], y[treino_idx]
            X_teste, y_teste = X[teste_idx], y[teste_idx]
            
            # --- Fase de Treinamento ---
            start_train = time.time()
            reg.fit(X_treino, y_treino)
            tempo_treino = time.time() - start_train
            
            # --- Fase de Teste (Predição) ---
            start_test = time.time()
            y_pred = reg.predict(X_teste)
            tempo_teste = time.time() - start_test
            
            # --- Fase de Avaliação (Métricas Matemáticas) ---
            r2 = r2_score_manual(y_teste, y_pred)
            # O R2 Ajustado precisa do número de amostras no teste (n) e do número de atributos (p)
            r2_adj = r2_ajustado_manual(r2, n=len(y_teste), p=p_atributos)
            
            # Armazenando os resultados deste Fold
            resultados[nome_reg]['r2'].append(r2)
            resultados[nome_reg]['r2_adj'].append(r2_adj)
            resultados[nome_reg]['t_treino'].append(tempo_treino)
            resultados[nome_reg]['t_teste'].append(tempo_teste)

    # 4. Construção e Impressão da Tabela Comparativa Exigida
    linha_divisoria = "-" * 110
    print(linha_divisoria)
    print(f"{'Regressor':<30} | {'R2-Score':<15} | {'R2-Score Ajustado':<18} | {'Tempo Treino (s)':<16} | {'Tempo Teste (s)'}")
    print(linha_divisoria)
    
    for nome, metricas in resultados.items():
        r2_str = format_resultado(np.mean(metricas['r2']), np.std(metricas['r2']))
        r2_adj_str = format_resultado(np.mean(metricas['r2_adj']), np.std(metricas['r2_adj']))
        t_treino_str = format_resultado(np.mean(metricas['t_treino']), np.std(metricas['t_treino']))
        t_teste_str = format_resultado(np.mean(metricas['t_teste']), np.std(metricas['t_teste']))
        
        print(f"{nome:<30} | {r2_str:<15} | {r2_adj_str:<18} | {t_treino_str:<16} | {t_teste_str}")
    print(linha_divisoria)

if __name__ == "__main__":
    run_regression_experiment()
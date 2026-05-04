import csv
import numpy as np

def load_solar_flare_regression(filepath):
    linhas_dados = []
    lendo_dados = False
    
    # 1. Leitura do arquivo ARFF ignorando o cabeçalho
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha_limpa = linha.strip()
            if lendo_dados:
                if linha_limpa and not linha_limpa.startswith('%'):
                    linhas_dados.append(linha_limpa)
            elif linha_limpa.upper().startswith('@DATA'):
                lendo_dados = True
                
    leitor = csv.reader(linhas_dados)
    matriz_bruta = list(leitor)
    
    # As 10 primeiras colunas são atributos categóricos
    X_raw = [linha[:10] for linha in matriz_bruta]
    
    # 2. A variável alvo agora é contínua (ex: quantidade de explosões classe C)
    # Convertendo para float e criando um array Numpy 1D
    y = np.array([float(linha[10]) for linha in matriz_bruta])
    
    # 3. One-Hot Encoding Manual (Expande as 10 colunas em múltiplas colunas binárias)
    num_linhas = len(X_raw)
    num_colunas_originais = len(X_raw[0])
    
    listas_encoded = [[] for _ in range(num_linhas)]
    
    for col_idx in range(num_colunas_originais):
        # Limpa aspas e encontra valores únicos da coluna
        valores_coluna = [linha[col_idx].strip("\"'") for linha in X_raw]
        valores_unicos = list(set(valores_coluna))
        
        # Para cada linha, cria um vetor de zeros e marca 1 na posição da categoria
        for row_idx in range(num_linhas):
            vetor_zeros = [0] * len(valores_unicos)
            valor_original = X_raw[row_idx][col_idx].strip("\"'")
            idx_categoria = valores_unicos.index(valor_original)
            vetor_zeros[idx_categoria] = 1
            listas_encoded[row_idx].extend(vetor_zeros)
            
    # X_encoded agora terá muito mais do que 10 colunas (aprox. 30 colunas binárias)
    X_encoded = np.array(listas_encoded, dtype=float)
    
    return X_encoded, y
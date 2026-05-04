import csv
import numpy as np

def load_arff(filepath):
    """
    Lê o arquivo ARFF, ignora os metadados e extrai a matriz bruta.
    """
    linhas_dados = []
    lendo_dados = False
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha_limpa = linha.strip()
            
            # Se já encontramos a tag @DATA, coletamos as linhas
            if lendo_dados:
                if linha_limpa and not linha_limpa.startswith('%'):
                    linhas_dados.append(linha_limpa)
            
            # Detecta a transição de metadados para dados reais
            elif linha_limpa.upper().startswith('@DATA'):
                lendo_dados = True
                
    # O csv.reader é vital aqui para lidar com vírgulas dentro de textos entre aspas
    leitor = csv.reader(linhas_dados)
    matriz_bruta = list(leitor)
    
    return matriz_bruta

def process_data(matriz_bruta, colunas_numericas, indice_target):
    """
    Filtra colunas numéricas e converte o target para numérico.
    (Implementação genérica aguardando sua decisão de tratamento)
    """
    X_raw = [[linha[i] for i in colunas_numericas] for linha in matriz_bruta]
    y_raw = [linha[indice_target] for linha in matriz_bruta]
    
    # Conversão de features para float (falhará se houver texto puro)
    X = np.array(X_raw, dtype=float)
    
    # Label Encoding do Target
    classes_unicas = list(set(y_raw))
    mapa_classes = {val: idx for idx, val in enumerate(classes_unicas)}
    y = np.array([mapa_classes[val] for val in y_raw], dtype=int)
    
    return X, y, mapa_classes
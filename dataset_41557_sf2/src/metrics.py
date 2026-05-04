import numpy as np

def k_fold_split(X, y, k=5, seed=42):
    """Gera os índices para K-Fold Cross Validation. (Permanece inalterado)"""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    tamanho_fold = len(X) // k
    folds = []
    for i in range(k):
        inicio = i * tamanho_fold
        fim = (i + 1) * tamanho_fold if i < k - 1 else len(X)
        indices_teste = indices[inicio:fim]
        indices_treino = np.concatenate([indices[:inicio], indices[fim:]])
        folds.append((indices_treino, indices_teste))
    return folds

def r2_score_manual(y_true, y_pred):
    """
    Calcula o Coeficiente de Determinação R².
    R² = 1 - (Soma dos Quadrados dos Resíduos / Soma dos Quadrados Totais)
    """
    # SS_res: Soma do quadrado da diferença entre o valor real e a predição
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # SS_tot: Soma do quadrado da diferença entre o valor real e a média de Y
    media_y = np.mean(y_true)
    ss_tot = np.sum((y_true - media_y) ** 2)
    
    # Evita divisão por zero se ss_tot for muito pequeno
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

def r2_ajustado_manual(r2, n, p):
    """
    Calcula o R² Ajustado, penalizando o modelo pela quantidade de atributos (p).
    R²_ajustado = 1 - [((1 - R²) * (n - 1)) / (n - p - 1)]
    """
    # Evita divisão por zero se a quantidade de amostras for insuficiente
    if (n - p - 1) == 0:
        return 0.0
        
    numerador = (1 - r2) * (n - 1)
    denominador = (n - p - 1)
    return 1 - (numerador / denominador)
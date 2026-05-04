import numpy as np

def k_fold_split(X, y, k=5, seed=42):
    """Gera os índices para K-Fold Cross Validation."""
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

def calculate_metrics(y_true, y_pred):
    """Calcula Acurácia, Precisão, Recall e F1-Score (Macro)."""
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []
    
    corretos = np.sum(y_true == y_pred)
    acuracia = corretos / len(y_true)
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return acuracia, np.mean(precisions), np.mean(recalls), np.mean(f1s)
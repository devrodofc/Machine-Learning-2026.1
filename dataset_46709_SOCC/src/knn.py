import numpy as np

class KNN:
    def __init__(self, k=5, dist_metric='euclidean'):
        self.k = k
        self.dist_metric = dist_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distancia(self, x1, x2):
        if self.dist_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.dist_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))

    def predict(self, X):
        predictions = []
        for x in X:
            # Calcula a distância entre x e todos os pontos de treino
            distancias = [self._distancia(x, x_treino) for x_treino in self.X_train]
            # Obtém os índices dos K vizinhos mais próximos
            k_indices = np.argsort(distancias)[:self.k]
            # Extrai as classes desses vizinhos
            k_vizinhos_labels = [self.y_train[i] for i in k_indices]
            # Votação majoritária
            classe_predita = max(set(k_vizinhos_labels), key=k_vizinhos_labels.count)
            predictions.append(classe_predita)
        return np.array(predictions)
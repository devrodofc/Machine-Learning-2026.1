import numpy as np

class KNN:
    def __init__(self, k=5, dist_metric='euclidean'):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            if self.dist_metric == 'euclidean':
                distancias = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            else: # manhattan
                distancias = np.sum(np.abs(self.X_train - x), axis=1)
                
            k_indices = np.argsort(distancias)[:self.k]
            k_labels = self.y_train[k_indices]
            # Votação majoritária otimizada
            valores, contagens = np.unique(k_labels, return_counts=True)
            predictions.append(valores[np.argmax(contagens)])
        return np.array(predictions)
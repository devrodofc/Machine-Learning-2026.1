import numpy as np

class KNNRegressor:
    def __init__(self, k=5, dist_metric='euclidean'):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Cálculo de distância vetorizado
            if self.dist_metric == 'euclidean':
                distancias = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            else: # manhattan
                distancias = np.sum(np.abs(self.X_train - x), axis=1)
                
            # Seleciona os K vizinhos
            k_indices = np.argsort(distancias)[:self.k]
            k_valores = self.y_train[k_indices]
            
            # Em vez de votação, calculamos a média aritmética dos K valores contínuos
            predictions.append(np.mean(k_valores))
            
        return np.array(predictions)

class LinearRegressionMultipla:
    def __init__(self):
        self.betas = None

    def fit(self, X, y):
        """
        Implementa a Equação Normal: Beta = (X^T * X)^-1 * X^T * Y
        Para que a regressão encontre o intercepto (beta_0), precisamos
        adicionar uma coluna de '1's à matriz X (o "termo de viés").
        """
        num_linhas = X.shape[0]
        coluna_bias = np.ones((num_linhas, 1))
        
        # Concatena a coluna de '1's no início da matriz X
        X_design = np.hstack((coluna_bias, X))
        
        # Vetorização da Equação Normal
        # 1. X_T: Matriz Transposta de X
        X_T = X_design.T
        
        # 2. Produto escalar de X_T e X
        X_T_X = np.dot(X_T, X_design)
        
        # Lida com matrizes singulares (não-inversíveis) usando a Pseudo-Inversa
        # Se os dados (one-hot encoding) tiverem colunas altamente correlacionadas, 
        # a matriz normal não tem inversa matemática.
        X_T_X_inv = np.linalg.pinv(X_T_X) 
        
        # 3. Produto escalar com X_T novamente e depois com Y
        X_T_Y = np.dot(X_T, y)
        
        # 4. Encontrando o vetor de coeficientes Betas
        self.betas = np.dot(X_T_X_inv, X_T_Y)

    def predict(self, X):
        """
        Calcula as predições usando: Y_pred = X * Beta
        """
        num_linhas = X.shape[0]
        coluna_bias = np.ones((num_linhas, 1))
        X_design = np.hstack((coluna_bias, X))
        
        # O produto escalar entre a matriz de design e os coeficientes gera o Y
        return np.dot(X_design, self.betas)
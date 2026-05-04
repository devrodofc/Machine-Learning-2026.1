import numpy as np

class NaiveBayes:
    def __init__(self, multivariado=False):
        self.multivariado = multivariado
        self.classes = None
        self.params = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'prior': len(X_c) / len(X),
                'mean': np.mean(X_c, axis=0)
            }
            if self.multivariado:
                # np.cov espera variáveis nas linhas, então transpomos X_c
                cov_matrix = np.cov(X_c.T)
                # Adiciona epsilon à diagonal para evitar matriz singular
                epsilon = 1e-6
                np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + epsilon)
                self.params[c]['cov'] = cov_matrix
                self.params[c]['cov_inv'] = np.linalg.inv(cov_matrix)
                self.params[c]['cov_det'] = np.linalg.det(cov_matrix)
            else:
                self.params[c]['var'] = np.var(X_c, axis=0) + 1e-6

    def _pdf_univariada(self, x, mean, var):
        num = np.exp(-((x - mean)**2) / (2 * var))
        den = np.sqrt(2 * np.pi * var)
        return num / den

    def _pdf_multivariada(self, x, mean, cov_inv, cov_det):
        d = len(mean)
        diff = x - mean
        expoente = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        denominador = ((2 * np.pi) ** (d / 2)) * (cov_det ** 0.5)
        return np.exp(expoente) / denominador

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriores = []
            for c in self.classes:
                p = self.params[c]
                prior = np.log(p['prior']) # Usando log para evitar underflow numérico
                
                if self.multivariado:
                    likelihood = np.log(self._pdf_multivariada(x, p['mean'], p['cov_inv'], p['cov_det']))
                else:
                    likelihood = np.sum(np.log(self._pdf_univariada(x, p['mean'], p['var'])))
                
                posteriores.append(prior + likelihood)
            y_pred.append(self.classes[np.argmax(posteriores)])
        return np.array(y_pred)
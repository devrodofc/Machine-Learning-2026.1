import numpy as np

class NaiveBayes:
    def __init__(self, multivariado=False):
        self.multivariado = multivariado
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
                cov_matrix = np.cov(X_c.T)
                np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 1e-6)
                self.params[c]['cov_inv'] = np.linalg.inv(cov_matrix)
                self.params[c]['cov_det'] = np.linalg.det(cov_matrix)
            else:
                self.params[c]['var'] = np.var(X_c, axis=0) + 1e-6

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriores = []
            for c in self.classes:
                p = self.params[c]
                prior = np.log(p['prior'])
                if self.multivariado:
                    d = len(p['mean'])
                    diff = x - p['mean']
                    expoente = -0.5 * np.dot(np.dot(diff.T, p['cov_inv']), diff)
                    denominador = ((2 * np.pi) ** (d / 2)) * (p['cov_det'] ** 0.5)
                    likelihood = np.log(np.exp(expoente) / denominador)
                else:
                    num = np.exp(-((x - p['mean'])**2) / (2 * p['var']))
                    den = np.sqrt(2 * np.pi * p['var'])
                    likelihood = np.sum(np.log(num / den))
                
                posteriores.append(prior + likelihood)
            y_pred.append(self.classes[np.argmax(posteriores)])
        return np.array(y_pred)
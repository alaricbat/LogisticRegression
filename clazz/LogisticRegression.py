import numpy as np

class LogisticRegression:

    def __init__(self,
                 learning_rate = 0.05,
                 epoches = 1000,
                 lamda = 0.1
                 ):
        self.lr = learning_rate
        self.epoches = epoches
        self.lamda = lamda
        self.w = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)

        for _ in range(self.epoches):
            random_ids = np.random.permutation(n_samples)
            for i in random_ids:
                x_i = X[i, :]
                y_i = y[i]
                z_i = self.__sigmoid(x_i @ self.w) 
                # Gradient update: scale L2 regularization by n_samples for SGD
                grad = (z_i - y_i) * x_i + (self.lamda / n_samples) * self.w  
                # Clip gradient to prevent extreme weight updates   
                grad = np.clip(grad, -5.0, 5.0)
                self.w -= self.lr * grad

    def predict(self, X, threshold = 0.5):
        return [1 if i > threshold else 0 for i in self.__sigmoid(X @ self.w)]
        

    def __sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip range to avoid floating-point overflow
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    

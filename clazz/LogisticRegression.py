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
                self.w -= self.lr * ((z_i - y_i) * x_i + self.lamda * self.w)

    def predict(self, X, threshold = 0.5):
        return [1 if i > threshold else 0 for i in self.__sigmoid(X @ self.w)]
        

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    

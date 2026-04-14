import numpy as np

class LogRegressionL2:
    def __init__(self, lr=0.01, lam=1.0, n_iter=1000):
        self.lr = lr
        self.lambda_ = lam
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def set_model(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for i in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (y_pred - y)) + self.lambda_ * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db
        
    def predict_prob(self, X):
            return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
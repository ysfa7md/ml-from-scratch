import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.n, self.d = X.shape
        self._w = np.zeros(self.d)
        self._b = 0

        for epoch in range(self.epochs):
            z = X @ self._w + self._b
            y_pred = self.sigmoid(z)

            loss = self.loss(y, y_pred)

            dw = (1 / self.n) * (X.T @ (y_pred - y))
            db = (1 / self.n) * np.sum(y_pred - y)

            self.update_parm(dw, db)
            if epoch % 50==0:
                print(f"epoch={epoch}, loss={loss:.4f}")

    def predict_proba(self, X):
        X = np.array(X)
        z = X @ self._w + self._b
        return self.sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def update_parm(self, dw, db):
        self._w -= self.lr * dw
        self._b -= self.lr * db

    def loss(self, y, y_pred):
        eps = 1e-9
        return (-1 / self.n) * np.sum(
            y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
        )


    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

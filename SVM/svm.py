import numpy as np


class SVM:
    def __init__(self, c=1.0, lr=0.01, epochs=500):
        self.c = c
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.n, self.d = X.shape
        self._w = np.zeros(self.d)
        self._b = 0

        for epoch in range(self.epochs):
            for sampel in range(self.n):
                x_i, y_i = X[sampel], y[sampel]
                z = x_i @ self._w + self._b

                margin = y_i * z

                if margin >= 1:
                    dw = self._w
                    db = 0
                else:
                    dw = self._w - self.c * y_i * x_i
                    db = -self.c * y_i

                self.update_parm(dw, db)

                if sampel == self.n - 1 and epoch % 50==0:
                    loss = self.loss(X,y)
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

                # margins = y * (X @ self._w + self._b)
                # mask = margins < 1
                # dw = self._w - self.c * np.sum((y[mask][:, np.newaxis] * X[mask]), axis=0)
                # db = -self.c * np.sum(y[mask])

    def predict(self, X):
        X = np.array(X)
        return np.sign(X @ self._w + self._b)

    def update_parm(self, dw, db):
        self._w -= self.lr * dw
        self._b -= self.lr * db

    def accuracy(self, X, y):
        y = np.where(y <= 0, -1, 1)
        preds = self.predict(X)
        return np.mean(preds == y)

    def loss(self,X,y):
        return 0.5 * np.sum(self._w**2) + self.c * np.sum(np.maximum(0, 1 - y * (X @ self._w + self._b)))

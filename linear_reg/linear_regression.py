import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, epochs=100):
        self._w = 0
        self._b = 0
        self.n = 0
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        self._w = 0.0
        self._b = 0.0
        X = np.array(X).flatten()
        y = np.array(y)
        self.n = len(X)
        for epoch in range(self.epochs):
            y_prd = self._w * X + self._b
            loss = self.loss(y, y_prd)
            self.losses.append(loss)
            self.compute_grad(X, y, y_prd)
            # print(f"epoch={epoch}, w={self._w:.2f}, b={self._b:.2f} => loss={loss:.2f}")

    def loss(self, y, y_prd):
        return (1 / self.n) * np.sum((y - y_prd) ** 2)

    def compute_grad(self, X, y, y_prd):
        dw = (-2 / self.n) * np.sum(X * (y - y_prd))
        db = (-2 / self.n) * np.sum((y - y_prd))

        self.update_parm(dw, db)

    def update_parm(self, dw, db):
        self._w -= self.lr * dw
        self._b -= self.lr * db

    def predict(self, X_test):
        X_test = np.array(X_test).flatten()
        return self._w * X_test + self._b

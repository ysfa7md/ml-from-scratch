import numpy as np


class LinearRegression:
    def __init__(self,lr=0.01,epoch=100):
        self._w = 0
        self._b = 0
        self.n = 0
        self.lr = lr
        self.epoch = epoch

    def fit(self, X_train, y_train):
        self.n = len(X_train)
        for i in range(self.epoch):
            y_prd = self._w * X_train + self._b
            loss = self.loss(y_train, y_prd)
            self.compute_grad(X_train, y_train, y_prd)

            print(f'{i=:}, {self._w=: .2f}, {self._b=: .2f}=> {loss=: .2f}')



    def loss(self, y_train, y_prd):
        return (1 / self.n) * np.sum((y_train - y_prd) ** 2)

    def compute_grad(self, X_train, y_train, y_prd):
        dw = (-2 / self.n) * np.sum(X_train * (y_train - y_prd))
        db = (-2 / self.n) * np.sum((y_train - y_prd))

        self.update_parm(dw, db)

    def update_parm(self, dw, db):
        self._w -= self.lr * dw
        self._b -= self.lr * db

    def pred(self, X_test):
        return self._w * X_test + self._b

def main():
    # dataset
    X = np.array([1,2,3,4,5])
    y = np.array([5,4,3,2,1])
    model=LinearRegression(lr=0.01,epoch=2000)
    model.fit(X,y)
    yprd=model.pred(X)
    print(yprd)
    print(y)

if __name__=='__main__':
    main()

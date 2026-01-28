import numpy as np

def distance(X1,X2):
    ss=(X1-X2)**2
    distance = np.sqrt(np.sum(ss))
    return distance

class KNN():
    def __init__(self,k=3):
        self.k=k

    def fit(self,X, y):
        self.X=np.array(X)
        self.y=np.array(y)

    def predict(self,X):
        return [self.predict_one(x) for x in X]

    def predict_one(self,X):
        distances = [distance(X,x) for x in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        return np.argmax(np.bincount(k_nearest_labels))



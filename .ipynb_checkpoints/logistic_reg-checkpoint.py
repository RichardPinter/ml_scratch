import numpy as np
import pandas as pd

### Binary classification, linearly separable
class LogisticRegression:

    def __init__(self,lr = 0.00001, epoch = 10000):
        self.lr = lr
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        ### Gradient descent
        for _ in range(self.epoch):
            linear_model =  np.dot(X,self.weights)+self.bias
            y_predicted =  self._sigmoid(linear_model)


            ### Update the weights
            dw =  2/n_samples *np.dot(X.T,(y-y_predicted))
            db = 1/n_samples *np.sum(y-y_predicted)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X,cutoff):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [True if prob >cutoff else False for prob in y_predicted]

        return y_predicted_cls

    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


# Dataset is available at https://www.kaggle.com/oddrationale/mnist-in-csv
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from future.utils import iteritems

class Data(object):
    def get_data(self, limit=None):
        df = pd.read_csv('mnist_train.csv')
        data = df.values
        np.random.shuffle(data)
        X = data[:, 1:] / 255
        y = data[:, 0]
        if limit is not None:
            X, y = X[:limit], y[:limit]
        return X, y

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing
            }
            self.priors[c] = float(len(Y[Y == c]))/len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
    data = Data()
    X, y = data.get_data(30000)

    Ntrain = 24000
    Xtrain, Ytrain = X[:Ntrain], y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], y[Ntrain:]

    model = Bayes()
    model.fit(Xtrain, Ytrain)

    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))

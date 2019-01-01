# Dataset is available at https://www.kaggle.com/oddrationale/mnist-in-csv
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
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

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            sl = SortedList() # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
            # print "input:", x
            # print "sl:", sl

            # vote
            votes = {}
            for _, v in sl:
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    data = Data()
    X, y = data.get_data(2000)

    Ntrain = 1600
    Xtrain, Ytrain = X[:Ntrain], y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], y[Ntrain:]
    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5)
    for k in ks:
        knn = KNN(k)
        knn.fit(Xtrain, Ytrain)

        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)

        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)

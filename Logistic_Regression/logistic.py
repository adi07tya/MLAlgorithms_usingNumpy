import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Data(object):
    def get_data(self):
        df = pd.read_csv('data.csv')
        data = df.values
        np.random.shuffle(data)
        X = data[:,:-1]
        Y = data[:,-1].astype(np.int32)

        N, D = X.shape
        X2 = np.zeros((N, D+3))
        X2[:,0:(D-1)] = X[:,0:(D-1)]

        for n in range(N):
            t = int(X[n,D-1])
            X2[n,t+D-1] = 1

        X = X2
        for i in (1,2):
            m = X[:,i].mean()
            s = X[:,i].std()
            X[:,i] = (X[:,i] - m) / s

        Xtrain = X[:-100]
        Ytrain = Y[:-100]
        Xtest = X[-100:]
        Ytest = Y[-100:]

        return Xtrain, Ytrain, Xtest, Ytest

class LogisticRegression(object):
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def forward(self, X, W , b):
        return self.sigmoid(X.dot(W) + b)

    def cross_entropy(self, T, Y):
        return -(np.mean(T*np.log(Y) + (1 - Y)*np.log(1 - Y)))

    def fit(self, X, Y):
        D = X.shape[1]
        self.W = np.random.randn(D)
        self.b = 0

        costs = []
        learning_rate = 0.01
        for i in range(1000):
            pY = self.forward(X, self.W, self.b)
            c = self.cross_entropy(Y,pY)
            costs.append(c)

            self.W -= learning_rate*Xtrain.T.dot(pY - Y)
            self.b -= learning_rate*(pY - Y).sum()
            if i%10 == 0:
                print(i, c)

        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.round(self.sigmoid(X.dot(self.W) + self.b))

    def score(self, T, Y):
        return np.mean(T == Y)

if __name__ == '__main__':
    data = Data()
    Xtrain, Ytrain, Xtest, Ytest = data.get_data()
    Xtrain = Xtrain[Ytrain <= 1]
    Ytrain = Ytrain[Ytrain <= 1]
    lr = LogisticRegression()
    lr.fit(Xtrain, Ytrain)
    T = lr.predict(Xtrain)
    print("Training Score: ",lr.score(T,Ytrain))

    T = lr.predict(Xtest)
    print("Test Score: ",lr.score(T,Ytest))

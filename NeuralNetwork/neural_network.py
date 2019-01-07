import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

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

        Xtrain = X[:-100]
        Ytrain = Y[:-100]
        Xtest = X[-100:]
        Ytest = Y[-100:]

        for i in (1, 2):
            m = Xtrain[:,i].mean()
            s = Xtrain[:,i].std()
            Xtrain[:,i] = (Xtrain[:,i] - m) / s
            Xtest[:,i] = (Xtest[:,i] - m) / s

        return Xtrain, Ytrain, Xtest, Ytest

class NeuralNetwork(object):
    def softmax(self, a):
        expA = np.exp(a)
        return expA / expA.sum(axis=1, keepdims=True)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def forward(self, X, W1, b1, W2, b2):
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return self.softmax(Z.dot(self.W2) + self.b2), Z

    def cross_entropy(self, T, Y):
        return -np.mean(T*np.log(Y))

    def fit(self, X, Y):
        M = 5
        D = X.shape[1]
        K = Y.shape[1]
        self.W1 = np.random.randn(D,M)
        self.b1 = np.random.randn(M)
        self.W2 = np.random.randn(M,K)
        self.b2 = np.random.randn(K)

        costs = []
        learning_rate = 0.01
        for i in range(10000):
            pY, Z = self.forward(X, self.W1, self.b1, self.W2, self.b2)
            c = self.cross_entropy(Y, pY)
            costs.append(c)

            self.W2 -= learning_rate*Z.T.dot(pY- Y)
            self.b2 -= learning_rate*(pY - Y).sum(axis=0)
            self.dZ = (pY - Y).dot(self.W2.T) * (1 - Z*Z)
            self.W1 -= learning_rate*X.T.dot(self.dZ)
            self.b1 -= learning_rate*self.dZ.sum(axis=0)
            if i%1000 == 0:
                print(i, c)

    def predict(self, X):
        Y, Z = self.forward(X, self.W1, self.b1, self.W2, self.b2)
        return np.argmax(Y, axis=1)

    def score(self, T, Y):
        return np.mean(T == Y)

if __name__ == '__main__':
    data = Data()
    Xtrain, Ytrain, Xtest, Ytest = data.get_data()
    K = len(set(Ytrain))
    Ytrain_ind = y2indicator(Ytrain, K)
    Ytest_ind = y2indicator(Ytest, K)

    model = NeuralNetwork()
    model.fit(Xtrain, Ytrain_ind)
    print("Training Score: ", model.score(model.predict(Xtrain), Ytrain))

    print("Testing Score: ", model.score(model.predict(Xtest), Ytest))

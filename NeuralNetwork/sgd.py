import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime

def forward(X, W, b):
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y

def predict(p_y):
    return np.argmax(p_y, axis=1)

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()

def gradW(t, y, X):
    return X.T.dot(t - y)

def gradb(t, y):
    return (t - y).sum(axis=0)

def one_hot(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i,y[i]] = 1
    return ind

def get_data():
    df = pd.read_csv('mnist_train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:,1:]
    Y = data[:,0].astype(np.int32)

    Xtrain = X[:2500]
    Ytrain = Y[:2500]
    Xtest = X[2500:3000]
    Ytest = Y[2500:3000]

    mean = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)

    Xtrain = (Xtrain - mean)
    Xtest = (Xtest - mean)

    return Xtrain, Ytrain, Xtest, Ytest

def sgd():
    Xtrain, Xtest, Ytrain, Ytest = get_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = one_hot(Ytrain)
    Ytest_ind = one_hot(Ytest)

    # 1. full
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in range(50):
        p_y = forward(Xtrain, W, b)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)


        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)


    # 2. stochastic
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(50): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_stochastic.append(ll)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)


    # 3. batch
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_batch.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for batch GD:", datetime.now() - t0)



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sgd()

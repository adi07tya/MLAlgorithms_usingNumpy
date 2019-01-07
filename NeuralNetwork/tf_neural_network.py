import numpy as np
import pandas as pd
import tensorflow as tf
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
    def forward(self, X, W1, b1, W2, b2):
        Z = tf.nn.softmax(tf.matmul(X, W1) + b1)
        return tf.matmul(Z, W2) + b2

    def fit(self, X, Y):
        D = X.shape[1]
        M = 3
        K = 4

        N = len(Y)
        T = np.zeros((N, K))
        for i in range(N):
            T[i, Y[i]] = 1

        self.W1 = tf.Variable(tf.random_normal([D, M], stddev=0.01))
        self.b1 = tf.Variable(tf.random_normal([M], stddev=0.01))
        self.W2 = tf.Variable(tf.random_normal([M, K], stddev=0.01))
        self.b2 = tf.Variable(tf.random_normal([K], stddev=0.01))

        tfX = tf.placeholder(tf.float32, [None, D])
        tfY = tf.placeholder(tf.float32, [None, K])

        logits = self.forward(tfX, self.W1, self.b1, self.W2, self.b2)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfY,logits=logits))
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
        predict_op = tf.argmax(logits, 1)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(10000):
            sess.run(train_op, feed_dict={tfX: X, tfY: T})
            pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
            if i % 1000 == 0:
                print("Accuracy:", np.mean(Y == pred))

if __name__ == '__main__':
    data = Data()
    Xtrain, Ytrain, Xtest, Ytest = data.get_data()

    nn = NeuralNetwork()
    nn.fit(Xtrain, Ytrain)

import numpy as np
import pandas as pd
import tensorflow as tf

def score(T, Y):
    return np.mean(T == Y)

def get_data():
    df = pd.read_csv('mnist_train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:2500,1:] / 255
    y = data[:2500,0].astype(np.int32)

    Xtrain = X[:2000]
    Ytrain = y[:2000]
    Xtest = X[2000:]
    Ytest = y[2000:]

    return Xtrain, Ytrain, Xtest, Ytest

def neural_net():
    Xtrain, Ytrain, Xtest, Ytest = get_data()

    Ytrain_one_hot = np.zeros((len(Ytrain), 10))
    for i in range(len(Ytrain)):
        Ytrain_one_hot[i, Ytrain[i]] = 1

    Ytest_one_hot = np.zeros((len(Ytest), 10))
    for i in range(len(Ytest)):
        Ytest_one_hot[i, Ytest[i]] = 1

    lr = 0.00004
    reg = 0.01

    N, D = Xtrain.shape
    b_size = 500
    n_batches = N // b_size

    M1 = 300
    M2 = 100
    K = 10
    W1 = tf.Variable((np.random.randn(D,M1)/np.sqrt(D)).astype(np.float32))
    b1 = tf.Variable(np.zeros(M1).astype(np.float32))
    W2 = tf.Variable((np.random.randn(M1,M2)/np.sqrt(D)).astype(np.float32))
    b2 = tf.Variable(np.zeros(M2).astype(np.float32))
    W3 = tf.Variable((np.random.randn(M2,K)/np.sqrt(D)).astype(np.float32))
    b3 = tf.Variable(np.zeros(K).astype(np.float32))

    X = tf.placeholder(tf.float32, shape=(None,D), name='X')
    T = tf.placeholder(tf.float32, shape=(None,K), name='T')

    Z1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1,W2) + b2)
    Yish = tf.matmul(Z2,W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish,
           labels=T))
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
    predict_op = tf.argmax(Yish, 1)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(100):
            for j in range(n_batches):
                Xbatch = Xtrain[j*b_size:(j*b_size + b_size),]
                Ybatch = Ytrain_one_hot[j*b_size:(j*b_size + b_size),]

                sess.run(train_op, feed_dict={X:Xbatch, T:Ybatch})
                if j == 0:
                    test_cost = sess.run(cost, feed_dict={X:Xtest, T:Ytest_one_hot})
                    prediction = sess.run(predict_op, feed_dict={X:Xtest})
                    sc = score(prediction, Ytest)
                    print("test_cost=%.3f,test_score=%.3f"%(test_cost,sc))

if __name__ == '__main__':
    neural_net()

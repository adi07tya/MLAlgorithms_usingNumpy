import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2
#Input
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
#True Output
T = np.array([0, 1, 1, 0])

#Bias
ones = np.ones((N, 1))

#Feature Engineering
xy = (X[:, 0] * X[:, 1]).reshape(N, 1)

#Final Input
Xb = np.concatenate((ones, xy, X), axis=1)

w = np.random.randn(D + 2)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

#Gradient Descent
learning_rate = 0.01
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 1000 == 0:
        print(e)

    w += learning_rate * Xb.T.dot(T - Y)

    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

X = []
y = []
for line in open('data_2d.csv'):
    x1, x2, Y = line.split(',')
    X.append([1, float(x1), float(x2)])
    y.append(float(Y))


X = np.array(X)
y = np.array(y)

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
Yhat = np.dot(X, w)

d1 = y - Yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared:", r2)

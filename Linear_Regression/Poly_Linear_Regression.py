import numpy as np
import matplotlib.pyplot as plt

X = []
y = []
for line in open('data_poly.csv'):
    a, b = line.split(',')
    a = float(a)
    X.append([1, a, a*a])
    y.append(float(b))

X = np.array(X)
y = np.array(y)

plt.scatter(X[:,1], y)
plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
Yhat = np.dot(X, w)

plt.scatter(X[:, 1], y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

d1 = y - Yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared:", r2)

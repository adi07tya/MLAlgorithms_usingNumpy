import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
y = []
for line in open('data_1d.csv'):
    a, b = line.split(',')
    X.append(float(a))
    y.append(float(b))

X = np.array(X)
y = np.array(y)

plt.scatter(X, y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(y) - y.mean()*X.sum() ) / denominator
b = ( y.mean() * X.dot(X) - X.mean() * X.dot(y)) /denominator

Yhat = a*X + b

plt.scatter(X, y)
plt.plot(X, Yhat)
plt.show()

d1 = y - Yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)

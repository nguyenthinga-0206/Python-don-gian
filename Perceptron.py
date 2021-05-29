import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# means = [[2, 2], [2, 4]]
# cov = [[.2, .2], [.2, .2]]
# N = 1000
# X0 = np.random.multivariate_normal(means[0], cov, N).T
# X1 = np.random.multivariate_normal(means[1], cov, N).T

data = pd.read_csv('data.csv').values
N, d = data.shape
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

a = x[y[:, 0] == 1]
b = x[y[:, 0] == -1]

plt.scatter(a[:, 0], a[:, 1], c='red', edgecolors='none', s=30, label='C = 1')
plt.scatter(b[:, 0], b[:, 1], c='blue', edgecolors='none', s=30, label='C = -1')
plt.legend(loc=1)
plt.xlabel('A')
plt.ylabel('B')

# X = np.concatenate((X0, X1), axis = 1)
# y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar
# X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def h(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(x, y, w):
    return np.array_equal(h(w, x), y)

def perceptron (x, y, w_init):
    w = [w_init]
    N = x.shape[1]
    d = x.shape[0]
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = x.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(x, y, w_init)

print(w[-1].T)
plt.show()
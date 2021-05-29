#from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 3, 5, 6, 7, 8, 9, 10, 11]]).T
Y = np.array([[4, 5, 9, 14, 14, 15, 20, 22, 23]]).T
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

w0 = w[0][0]
w1 = w[1][0]
x0 = np.linspace(0, 12, 2)
y0 = w0 + w1 * x0

plt.plot(X.T, Y.T, 'ro')
plt.plot(x0, y0)
plt.show()

y1 = w1*9 + w0
y2 = w1*11 + w0

print( u'Predict weight of person with height 9 cm: %.2f (kg), real number: 20 (kg)'  %(y1) )
print( u'Predict weight of person with height 11 cm: %.2f (kg), real number: 23 (kg)'  %(y2) )
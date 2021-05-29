import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Load data từ file csv
data = pd.read_csv('../data.csv').values
N, d = data.shape
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

# Vẽ data bằng scatter
x_cho_vay = x[y[:, 0] == 1]
x_tu_choi = x[y[:, 0] == 0]

plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('X')
plt.ylabel('Y')

# Thêm cột 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.01

for i in range(1, numOfIteration):
    # Tính giá trị dự đoán
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
    # Gradient descent
    w = w - learning_rate * np.dot(x.T, y_predict - y)
    print(cost[i])

# Vẽ đường phân cách.
x = np.arange(4, 10, 1)
y = (-w[0] - w[1] * x) / w[2]
plt.plot(x,y)
plt.show()


import numpy as np

A = np.array([[0.3, 0.2],
              [0.4, 0.6]])

S = np.array([[1, 1],
              [1, 0]])

temp = np.matmul(S.transpose(), A)
out = np.matmul(temp, S)

print(out)
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(6)
nodes = np.random.random([10, 2])
plt.xlim(0, 1)
plt.ylim(0, 1)
tour1 = np.concatenate((np.array([[0.5, 0.5]]), nodes[1:5], np.array([[0.5, 0.5]])), axis=0)
# plt.plot(tour1[:, 0], tour1[:, 1], 'o')
plt.plot(tour1[:, 0], tour1[:, 1], 'o', linestyle='-')
plt.plot(nodes[0, 0], nodes[0, 1], 'o', color='black')
tour2 = np.concatenate((np.array([[0.5, 0.5]]), nodes[6:10], np.array([[0.5, 0.5]])), axis=0)
# plt.plot(tour2[:, 0], tour2[:, 1], 'o', color='red')
plt.plot(tour2[:, 0], tour2[:, 1], 'o', color='red', linestyle='-')
plt.plot(nodes[5, 0], nodes[5, 1], 'o', color='black')

# plt.plot(nodes[:, 0], nodes[:, 1], 'o')
plt.show()
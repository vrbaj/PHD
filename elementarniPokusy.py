import numpy as np

theta = np.zeros((1, 3))
theta[0, 0] = 1
theta[0, 1] = 2
theta[0, 2] = 3
print(theta.shape)
x = np.transpose(theta)
print(x.shape)
print(np.dot(theta,x))
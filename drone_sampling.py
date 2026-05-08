import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

#observation matrix
#load the matrix
observation = np.loadtxt('res.txt', skiprows=1, delimiter=',')

dims = (500, 200) #meters
obs = np.hstack([observation[:, 1:4], observation[:, 7:]])
print(obs.shape)

n_samples = 100
sample_points = np.vstack([np.random.random(n_samples) * dims[0], 
                           np.random.random(n_samples) * dims[1] - dims[1]/2]).T
print(sample_points.shape)

tree = KDTree(obs[:,:2])
_, index = tree.query(sample_points)
# print(index)

y_predict = obs[index, :]
# print(y_predict)


plt.scatter(obs[:,0], obs[:,1], c=obs[:,3], cmap='RdBu')
# plt.scatter(sample_points[:,0], sample_points[:,1], c='red')
plt.scatter(y_predict[:,0], y_predict[:,1], c=y_predict[:,3])
plt.colorbar()
plt.show()
#sample the complete matrix with a bunch of points in space
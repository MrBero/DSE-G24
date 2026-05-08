import numpy as np
import matplotlib.pyplot as plt

#observation matrix
#load the matrix
observation = np.loadtxt('res.txt', skiprows=1, delimiter=',')
sample_points = [[200, 100], [241.123, 248]]

dims = (500, 200) #meters
obs = np.hstack([observation[:, 1:4], observation[:, 7:]])
print(obs)
plt.scatter(obs[:,0], obs[:,1], c=obs[:,3], cmap='RdBu')
plt.scatter(sample_points[:,0], sample_points[:,1
, c='red')
plt.colorbar()
plt.show()
#sample the complete matrix with a bunch of points in space
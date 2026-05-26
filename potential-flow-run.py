import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

x, y, z = np.meshgrid(np.linspace(-10,10,100),
                      np.linspace(-10,10,100),
                      np.linspace(-10,10,100))
u = np.linspace(-10,10,100)
v = x**2 
w = x**3

ax.quiver(x,y,z,u,v,w)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

x, y, z = np.meshgrid(np.linspace(-10,10,10),
                      np.linspace(-10,10,10),
                      np.linspace(-10,10,10))

def source(x,y,z, x0,y0,z0,m):
    return m/(2*np.pi)*np.log( np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) )

def vel_potental(x,y,z):
    return source(x,y,z, 0, 1, 1, 100) + source(x,y,z, 0, 5, 1, -10)

def derivative_x(vel_potential, axis):
    h = (vel_potential[1,:] - vel_potential[0,:])
    return (vel_potential[1:,:] - vel_potential)/h #fwd difference


print(vel_potental(x,y,z).shape)
# ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
# plt.plot(x,y)
# plt.show()
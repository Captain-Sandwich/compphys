#!/usr/bin/python2

import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

np.random.seed(9905)

Np = 10000
N = 1000

# prepare lattice
walks = np.zeros((N, 2*N+1))
# some initialization
walks[0, N] = Np # initialize all particles in the middle
for i in xrange(1,N):
    for j in walks[i-1, :].nonzero()[0]: # Iterate over sites with particles
    # and move them to the left or right
        n = walks[i-1, j]
        r = (np.random.random(n) < 0.5) # make boolean variables
        walks[i, j-1] += sum(r) # walk left
        walks[i, j+1] += n - sum(r) # walk right

# calculate the averages
squared_mean_of_x = (np.sum(np.arange(-N,N+1) * walks, axis=1) / Np)**2
mean_of_x_squared = (np.sum(np.arange(-N,N+1)**2 * walks, axis=1) / Np)


#%% Plotting
plt.plot( mean_of_x_squared - squared_mean_of_x, 'r,', label='Simulation')  # Plot results
plt.plot( np.arange(N), 'k-', linewidth=1, alpha=0.8, label='Analytical prediction ')  # Plot expectations
plt.legend(loc=4)
plt.xlabel('$N$')
plt.ylabel(r'$\left< x^2 \right> - \left< x \right>^2$')
tikz_save('latex/images/random_walk.tikz', figureheight = '\\figureheight', figurewidth = '\\figurewidth')
plt.show()



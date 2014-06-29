#-*- coding: utf-8 -*-
"""
simple schrödinger
@author: patrick
"""

from schrodinger import Schrodinger1
from pylab import *
import matplotlib.animation as animation
#from util import set_plot_options
#set_plot_options()

#Prepare lattice, potential and wavefunction
delta = 0.1
tau = 0.001
m = 50000
L = 1001

#x for plotting
x = np.arange(L)*delta

#wave packet
x0 = 20
q = 1
sigma = 3
psi = np.arange(L,dtype='complex')*delta - x0
psi = (2*np.pi*sigma**2)**(-0.25) * np.exp(1j*-q*psi) * np.exp(-psi**2 / (4*sigma)**2) # added factor -1 in the first exponential

#potential
V = np.zeros(L)
V[500:505] = 2

#Solver instance
s = Schrodinger1(psi, V, tau, delta)

#some simple plotting

vscale = 1.0/np.max(V)
xscale = np.ones(L)*3
#xscale[551:] *= 100
times = []
probabilities = []
probabilities.append(s.probability())
times.append(s.T)
for i in range(100):
    s.evolve(500)
    probabilities.append(s.probability())
    times.append(s.T)

f = figure()
images = []
plot(x, vscale*V, lw=2, color='black')
for t, psi in zip(times, probabilities):
    line, = plot(x, xscale*psi, color='b')
    images.append([line])
im_ani = animation.ArtistAnimation(f, images, interval=100, blit=True)

show()


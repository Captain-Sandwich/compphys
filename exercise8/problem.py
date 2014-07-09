#-*- coding: utf-8 -*-
"""
simple schroedinger
@author: patrick
"""

from bloch import Bloch
from pylab import *
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from util import set_plot_options
set_plot_options()

# higher order function, that returns a B(t) function
# this makes the solver independent of the used B-field
def makeBfunc(omega, h, B0, phi=0):
    return lambda t: np.array([h*cos(omega*t + phi), -h*sin(omega*t + phi), np.ones_like(t)*B0])

f0 = 4.
f1 = 0.25
gamma = 1
tau = 1./(max(f1,f0)*100)
b0 = 2*np.pi*f0
h = 2*np.pi*f1
omega = b0
bfunc = makeBfunc(omega, h, b0)

T = 4

# parameters that change:
Ts = [[0,1], [1,0], [1,1]] # pairs of invt1, invt2
#Ts = [[0,0]]
Mi = [0,1,0]
Mi = [1,0,1]

for invt1, invt2 in Ts:
    # Mi = np.array([1,0,1])

    b = Bloch(Mi, invt1, invt2, tau, gamma, T, bfunc)
    M = b.run()
    t = arange(M.shape[0])*tau

    figure()
    lines = plot(t, M)
    xlabel('Time')
    ylabel('$M_i$')
    legend(lines, ('$M_x$', '$M_y$', '$M_z$'))
    savefig('latex/images/%.1f-%.1f-(%d,%d,%d).pdf' % (invt1, invt2, Mi[0], Mi[1], Mi[2])) 

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(M[:,0], M[:,1], M[:,2])
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    savefig('latex/images/3D %.1f-%.1f-(%d,%d,%d).pdf' % (invt1, invt2, Mi[0], Mi[1], Mi[2])) 


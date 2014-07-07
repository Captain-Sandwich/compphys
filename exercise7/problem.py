#-*- coding: utf-8 -*-
"""
simple schroedinger
@author: patrick
"""

from schrodinger import Schrodinger1
from pylab import *
import numpy as np
import matplotlib.animation as animation
from util import set_plot_options
set_plot_options()



def get_k_space(psi, delta):
    fourier = np.fft.fft(psi)
    L = psi.size
    freq = np.fft.fftfreq(L, delta)
    freq = np.fft.fftshift(freq)
    fourier = np.fft.fftshift(fourier)
    return freq, fourier

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
sigma = 3.
#psi = np.arange(L,dtype='complex')*delta - x0
#psi = (2*np.pi*sigma**2)**(-0.25) * np.exp(1j*-q*psi) * np.exp(-psi**2 / (4*sigma)**2) # added factor -1 in the first exponential
x = x-x0
psi = (2*np.pi*sigma**2)**(-0.25) * np.exp(1j*x*q) * np.exp(-x**2 / (4*sigma**2))

#potential
V = np.zeros(L)
V[500:505] = 2.


#Solver instances
s = Schrodinger1(psi, V, tau, delta) # With tunnel barrier
s2 = Schrodinger1(psi, np.zeros(L), tau, delta) # without tunnel barrier
# This works, because psi is not altered in-place

times = np.arange(6)*10.
steps = diff(times)/tau

#some simple plotting
vscale = 1.
xscale = np.ones(L)
#xscale[551:] *= 100

f = figure()
ax = f.add_subplot(111)
freqs = []
fts = []
ax.plot(x, s2.probability(), label='$t = %.1f$' % s2.T)
for n in steps:
    s2.evolve(n)
    ax.plot(x, s2.probability(), label='$t=%.1f$' % s2.T)

ax.legend()
ax.set_ylim(0, 0.2)
ax.set_xlabel('$x-x_0$')
ax.set_ylabel(r'$\left| \Psi(x) \right|^2$')
ax.set_title('Free Propagation')
f.savefig('latex/images/free.pdf')


f2 = figure()
ax1 = f2.add_subplot(111)
ax2 = ax1.twinx()
ax2.plot(x, vscale*V, lw=2, color='black')
ax1.plot(x, s.probability()*xscale, label='$t=%d$' % s.T)

fr,ft = get_k_space(s.get_psi(), delta)
freqs.append(fr)
fts.append(ft)
for n in steps:
    s.evolve(n)
    ax1.plot(x, s.probability()*xscale, label='$t=%.1f$' % s.T)

fr,ft = get_k_space(s.get_psi(), delta)
freqs.append(fr)
fts.append(ft)

ax1.set_xlabel('$x - x_0$')
ax1.set_ylim(0, 0.2)
ax1.legend()
ax2.set_ylabel('Potential $V$')
ax1.set_title('Tunneling Barrier')
ax1.set_ylabel(r'$\left| \Psi(x) \right|^2$')
f2.savefig('latex/images/tunneling.pdf')

f3 = figure()
ax3 = f3.add_subplot(111)
freq = freqs[0] * 2*np.pi # go from frequency to angular frequency
for i,ft in enumerate(fts):
    plot(freq, np.abs(ft), label='$t=%.1f$' % times[i])
ax3.set_xlabel('$q$')
ax3.legend()
ax3.vlines(1,0,40)
ax3.set_ylabel('Magnitude')
ax3.set_title('Momentum Space')
ax3.set_xlim(-2,2)
ax3.vli
f3.savefig('latex/images/fft.pdf')
show()
# #vscale = 1.0/np.max(V)
# times = []
# probabilities = []
# probabilities.append(s.probability())
# times.append(s.T)
# for i in range(100):
#     s.evolve(500)
#     probabilities.append(s.probability())
#     times.append(s.T)

 #f = figure()
 #images = []
 #plot(x, vscale*V, lw=2, color='black')
 #for t, psi in zip(times, probabilities):
 #    line, = plot(x, xscale*psi, color='b')
 #    images.append([line])
 #ylim(0,0.14)
 #im_ani = animation.ArtistAnimation(f, images, interval=100, blit=True)


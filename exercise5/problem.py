# -*- coding: utf-8 -*-
"""
Yee Algorithm Maxwell Equation Solver in 1D.

@author: bethke
"""

import numpy as np
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
from yee import Yee
from util import set_plot_options

set_plot_options()

# First construct the grid and all
la = 1 # lambda
pperla = 50
tau = 0.9 * la/pperla
#tau = 1.2 * la/pperla # instable case
m = 10000

#%% Build grid and boundary conditions
L = 100 * la * pperla
n = 1.46
# Build boundary layers
sigma = np.zeros(L)
sigma[:6*la*pperla] = 1
sigma[-6*la*pperla:] = 1
sigma_star = sigma[:-1]

mu = np.ones(L-1)
# build glass
epsilon = np.ones(L)*1.05
epsilon[L/2: L/2 + 2 * la * pperla] = n*n # Thin plate
#epsilon[L/2:] = n*n # Thick glass plate

# include source
f = 3
i_s = 20 * la * pperla


# Instantiate the Yee class
y = Yee(epsilon, sigma, mu, sigma_star, tau, i_s,  0.5*f)
# Then I call y.evolve(n) repeatedly to evolve the system for some time and then
# generate plots with "snapshots" of the E-field.


T = [40,10,30]
x = np.arange(0,L,dtype=np.float)/pperla
for i, t in enumerate(T):
    y.evolve(t/tau)
    plt.figure(figsize=(5.80555556/3.0*0.9*0.9, 1.75208862348))
    plt.axvspan(L/2./pperla, L/2./pperla + 2*la, color='#5E9DC8', label='Glass')
    plt.axvspan(0, 6*la, color='grey', alpha=0.5, label='Absorber')
    plt.axvspan(1.0*L/pperla - 6* la, L/pperla, color='grey', alpha=0.5)
    time_template = '$T$ = %d'
    plt.plot(x, y.E, 'k', lw=0.2)
    plt.text(10, 0.01, time_template % round(y.T), fontsize=11)
    plt.ylim(-0.015, 0.015)
    plt.xlabel(r'$x$/$\lambda$')
    plt.tick_params(top='off', right='off')
    if i==0:
        plt.ylabel('$E_z$')
    plt.savefig('latex/images/evolution1_%d.pdf' % (i+1))
    plt.close()

#%% make an animation
y = Yee(epsilon, sigma, mu, sigma_star, tau, i_s,  0.5*f)
fig = plt.figure()
plt.axvspan(L/2, L/2 + 2 * la * pperla, color='#5E9DC8', label='Glass')
plt.axvspan(0, 6*la*pperla, color='grey', alpha=0.5, label='Absorber')
plt.axvspan(L - 6* la * pperla, L, color='grey', alpha=0.5)
time_template = '$T$ = %d'
ims = []
frames = 500
for i in range(frames):
    y.evolve(m/frames) # Evolve the system
    line, = plt.plot(y.E, 'k', lw=1)
    time_text = plt.text(3000, 0.01, '')
    time_text.set_text(time_template % (i*m/frames*tau))
    ims.append([line, time_text])

ani = ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
#ani.save('glass plate.avi')
plt.show()

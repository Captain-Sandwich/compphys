# -*- coding=utf-8 -*-
#!/usr/bin/python2

from __future__ import division
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# First step: simulation of data records


np.random.seed(9905)

def genEPRBData(nsteps=32, nsamples=1e5, HWP2=0, T0=1000, W=1):
    '''This function generates EPRB count data. The data structure is
    analog to the count array in the FORTRAN sample program.'''
    cHWP2 = np.cos(HWP2 * pi / 180.0)
    sHWP2 = np.sin(HWP2 * pi / 180.0)
    count = np.zeros((2,2,2,nsteps), dtype=np.int_)
    ipsi0 = np.linspace(0, 2*pi, nsteps) # Here I prepare an array of angles
    # This unrolls the green FORTRAN loop
    cHWP1 = np.cos(ipsi0) #these functions are all vectorized
    sHWP1 = np.sin(ipsi0)

    for i in xrange(nsteps):
        r0 = np.random.random(nsamples)*2*pi # r0 is a vector of nsamples values between 0 and 2pi
        # This corresponds to the random polarisation angle of the first photon
        c1 = np.cos(r0)
        s1 = np.sin(r0)
        c2 = -s1 # The second photon is 90° polarised to the first one
        s2 = c1

        # The analyzers are called with nsamples values for each angle. this unrolls part of the blue loop
        j1, l1 = analyzer(c1, s1, cHWP1[i], sHWP1[i], T0) #call analyzer 1. the angle of the first analyzer is stepped
        j2, l2 = analyzer(c2, s2, cHWP2, sHWP2, T0) #call analyzer 2. this angle is fixed
        
        # Count
        for j in xrange(nsamples): # counting doesn't vectorize very nicely
            count[j1[j], j2[j], 0, i] +=1 # Malus law model
            if abs(l1[j]-l2[j]) < W: # Malus law model and time window
                count[j1[j], j2[j], 1, i] +=1
    return count, j1, l1, j2, l2
    

def analyzer(c, s, cHWP, sHWP, T0):
    # Plane rotation
    c2 = cHWP * c + sHWP * s
    s2 = cHWP * s - sHWP * c
    x = c2*c2 - s2*s2
    y = 2*c2*s2

    # Malus
    r0 = np.random.random(size=c.shape)
    j = np.int8(x > (2*r0 - 1))

    # time delay
    l = y*y*y*y*T0*np.random.random(size=c.shape)

    return j,l

nsteps = 32
nsamples = 100000

count = genEPRBData(nsteps=nsteps, nsamples=nsamples)
total = np.zeros((2, nsteps))
E12 = np.zeros_like(total)
E2 = np.zeros_like(total)
E1 = np.zeros_like(total)

#%%
for j in xrange(nsteps):
    for i in xrange(2):
        total[i, j] = np.sum(count[:,:,i,j])
        E12[i,j] = count[0,0,i,j] + count[1,1,i,j] - count[1,0,i,j] - count[0,1,i,j]
        E1[i,j] = count[0,0,i,j] + count[0,1,i,j] - count[1,1,i,j] - count[1,0,i,j]
        E2[i,j] = count[0,0,i,j] + count[1,0,i,j] - count[1,1,i,j] - count[0,1,i,j]
        if total[i,j]:
            E12[i,j] = E12[i,j]/total[i,j]
            E1[i,j] = E1[i,j]/total[i,j]
            E2[i,j] = E2[i,j]/total[i,j]

if __name__ == '__main__':
    # Simulate
    #E1, E2, E12, count = simulate(32, 100000)

    # Load results
    E1 = np.load('E1.npy')
    E12 = np.load('E12.npy')
    E2 = np.load('E2.npy')

    # Plot results
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    phi = np.linspace(0, 2, E1.shape[1])
    ax1.plot(phi, E1[0,:]*E2[0,:], label='no coincidence counting')
    ax1.plot(phi, E1[1,:]*E2[1,:], label='coincidence counting')
    ax2.plot(phi, E12[0,:], label='no coincidence counting')
    ax2.plot(phi, E12[1,:], label='coincidence counting')
    ax1.set_xlabel('$\\varphi / \\pi$')
    ax2.set_xlabel('$\\varphi / \\pi$')
    ax1.set_ylabel('E1*E2')
    ax2.set_ylabel('E12')
    ax1.legend()
    ax2.legend()
    fig.show()
    fig.tight_layout()


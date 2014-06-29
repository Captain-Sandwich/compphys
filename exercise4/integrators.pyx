# -*- coding: utf-8 -*-
"""
integrators

@author: bethke
"""
import numpy as np
cimport numpy as np
cimport cython

cdef double k=1.0
cdef double m=1.0


@cython.cdivision(True)
@cython.boundscheck(False)
def euler(np.ndarray[np.double_t, ndim=1] x,
          np.ndarray[np.double_t, ndim=1] v, double dt):
    '''Simple Euler integrator'''
    cdef int N = x.size
    cdef int i
    cdef double F
    for i in range(1, N):
        F = -k*x[i-1]
        v[i] = v[i-1] + F/m*dt
        x[i] = x[i-1] + v[i-1]*dt
    return x, v

@cython.cdivision(True)
@cython.boundscheck(False)
def euler_cromer_a(np.ndarray[np.double_t, ndim=1] x,
                   np.ndarray[np.double_t, ndim=1] v, double dt):
    '''Euler-Cromer integrator a)'''
    cdef int N = x.size
    cdef int i
    cdef double F
    for i in xrange(1, N):
        F = -k*x[i-1]
        v[i] = v[i-1] + F/m*dt
        x[i] = x[i-1] + v[i]*dt
    return x, v

@cython.cdivision(True)
@cython.boundscheck(False)
def euler_cromer_b(np.ndarray[np.double_t, ndim=1] x,
                   np.ndarray[np.double_t, ndim=1] v, double dt):
    '''Euler-Cromer integrator b)'''
    cdef int N = x.size
    cdef int i
    cdef double F
    for i in xrange(1, N):
        x[i] = x[i-1] + dt*v[i-1]
        F = -k*x[i]
        v[i] = v[i-1] + F/m*dt
    return x, v

@cython.cdivision(True)
@cython.boundscheck(False)
def verlet(np.ndarray[np.double_t, ndim=1] x,
           np.ndarray[np.double_t, ndim=1] v, double dt):
    '''Verlet integrator'''
    cdef int N = x.size
    cdef int i
    cdef double a
    # Euler for the first step
    a = -k*x[0]/m
    v[1] = v[0] + a*dt
    x[1] = x[0] + v[0]*dt

    for i in xrange(2, N):
        a = (-k*x[i-1])/m
        x[i] = 2*x[i-1] - x[i-2] + a*dt*dt
        v[i] = (x[i] - x[i-2])/2/dt
    return x, v

@cython.cdivision(True)
@cython.boundscheck(False)
def velocity_verlet(np.ndarray[np.double_t, ndim=1] x,
                    np.ndarray[np.double_t, ndim=1] v, double dt):
    '''Velocity Verlet integrator'''
    cdef int N = x.size
    cdef double dt2 = dt*dt
    cdef int i
    cdef double a
    cdef double a2
    # Self-starting algorithm
    a2 = (-k*x[0])/m
    for i in xrange(0, N-1):
        a = a2
        x[i+1] = x[i] + dt*v[i] + 0.5*a*dt2
        a2 = (-k*x[i+1])/m
        v[i+1] = v[i] + 0.5*(a + a2)*dt
    return x, v

@cython.cdivision(True)
@cython.boundscheck(False)
def mvverlet(np.ndarray[np.double_t, ndim=2] x,
             np.ndarray[np.double_t, ndim=2] v, double dt):
    ''' Velocity verlet integrator for a chain of coupled oscillators'''
    cdef int N = x.shape[0]
    cdef int Nosc = x.shape[1]
    cdef int i
    cdef double dt2 = dt*dt
    cdef np.ndarray a = np.zeros(Nosc, dtype=np.double)
    cdef np.ndarray a2 = np.zeros(Nosc, dtype=np.double)
    
    # First force calculation
    a2[0] = -(x[0,0] - x[0,1])*k/m
    a2[Nosc-1] = -(x[0,Nosc-1] - x[0, Nosc-2])*k/m
    a2[1:Nosc-1] = -(2*x[0, 1:Nosc-1] - x[0, 0:Nosc-2] - x[0, 2:Nosc])*k/m
    for i in xrange(0, N-1):
        a = a2.copy() # Only calculate the forces once per timestep        
        
        # Calculate new position
        x[i+1,:] = x[i,:] + dt*v[i,:] + 0.5*a*dt2

        # Force calculation        
        a2[0] = -(x[i+1,0] - x[i+1,1])*k/m
        a2[Nosc-1] = -(x[i+1,Nosc-1] - x[i+1, Nosc-2])*k/m
        a2[1:Nosc-1] = -(2*x[i+1, 1:Nosc-1] - x[i+1, 0:Nosc-2] - x[i+1, 2:Nosc])*k/m
        
        # Calculate new velocity
        v[i+1,:] = v[i,:] + 0.5*(a+a2)*dt
    return x,v
        

def multienergy(np.ndarray[np.double_t, ndim=2] x,
                np.ndarray[np.double_t, ndim=2] v):
    '''Calculates energy of multiple particles'''
    cdef int Nosc=x.shape[1]
    cdef int N = x.shape[0]
    cdef np.ndarray E = np.zeros(N, dtype=np.double)
    cdef np.ndarray dx = np.empty(Nosc, dtype=np.double)
    dx = x[:, 1:] - x[:, :-1]
    E = 0.5*(dx*dx).sum(axis=1) + 0.5*(v*v).sum(axis=1)
    return E


def energy(np.ndarray[np.double_t, ndim=1] x,
           np.ndarray[np.double_t, ndim=1] v):
    '''returns energy of one particle'''
    return 0.5*v*v + 0.5*x*x

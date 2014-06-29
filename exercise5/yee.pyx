 # -*- coding: utf-8 -*-
"""
integrators

@author: bethke
"""
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange

# import exp and sin function from C
cdef extern from "math.h":
    double exp(double x) nogil
    double sin(double x) nogil
    
cdef double pi = np.pi

cdef class Yee:
    '''1-dimensional Yee algorithm solver with a point-source that generates a
    wave package at specified position and frequency.'''
    cdef readonly np.ndarray E
    cdef readonly np.ndarray H
    cdef readonly np.ndarray A
    cdef readonly np.ndarray B
    cdef readonly np.ndarray C
    cdef readonly np.ndarray D
    cdef int N
    cdef readonly double T
    cdef readonly double tau
    cdef int pos
    cdef double freq
    
    def __init__(self, epsilon, sigma, mu, sigma_star, tau, source_pos, source_freq):
        self.tau = tau
        self.T = 0
        self.pos = source_pos
        self.freq = source_freq
        
        #Prepare vectors
        self.N = sigma.size
        self.E = np.zeros(self.N)
        self.H = np.zeros(self.N - 1)
        
        self._calc_matrices(epsilon, sigma, mu, sigma_star)
        
    def _calc_matrices(self, epsilon, sigma, mu, sigma_star):
        a = 0.5 * sigma_star * self.tau / mu
        self.A = (1 - a) / (1 + a)
        self.B = (self.tau / mu) / (1 + a)
        b = 0.5 * sigma * self.tau / epsilon
        self.C = (1 - b) / (1 + b)
        self.D = (self.tau / epsilon) / (1 + b)

    cdef double _source(self, double t) nogil:
        return sin(2*pi*t*self.freq) * exp(-((t-30)/10)**2)

    def source(self,t):
        return self._source(t)        
    
    cdef void step(self):
        '''Perform a single yee iteration'''
        self.E[1:-1] = self.D[1:-1] * (self.H[1:] - self.H[0:-1]) / self.tau + self.C[1:-1] * self.E[1:-1]
        # include source
        self.E[self.pos] = self.E[self.pos] - self.D[self.pos]*self._source(self.T)
        # second half-step
        self.H = self.B * (self.E[1:] - self.E[0:-1]) / self.tau + self.A * self.H
         #Increase time counter
        self.T += self.tau
    
    cpdef evolve(self, int nsteps):
        '''Evolve the system for a number of steps'''
        cdef int i
        for i in range(nsteps):
            self.step()
   

def yee(np.ndarray[np.double_t, ndim=1] E, np.ndarray[np.double_t, ndim=1] H,
        np.ndarray[np.double_t, ndim=1] A,
        np.ndarray[np.double_t, ndim=1] B,
        np.ndarray[np.double_t, ndim=1] C,
        np.ndarray[np.double_t, ndim=1] D,
        double tau,
        int N,
        int i_s,
        np.ndarray[np.double_t, ndim=1] J):
            
    cdef int i
    # Yee iteration
    for i in range(N):
        # first half-step
        #E[1:-2] = D[1:-2] * (H[1:-2] - H[0:-3]) / tau + C[1:-2] * E[1:-2]

        E[1:-1] = D[1:-1] * (H[1:-1] - H[0:-2]) / tau + C[1:-1] * E[1:-1]
        # include source
        E[i_s] = E[i_s] - D[i_s]*J[i]
        # second half-step
        H[0:-1] = B[0:-1] * (E[1:] - E[0:-1]) / tau + A[0:-1] * H[0:-1]
    return E, H
   
   
cdef int mandel(complex c) nogil:
    cdef double thresh = 100
    cdef int maxiter = 100
    cdef int i
    cdef complex z=0
    for i in range(maxiter):
        z = z*z + c
        if z.real*z.real + z.imag*z.imag > thresh:
            return i
    return maxiter

def python_mandel(c):
    thresh = 10
    maxiter = 100
    z = 0
    for i in range(maxiter):
        z = z*z + c
        if abs(z) > thresh:
            return i
    return maxiter

@cython.boundscheck(False)
def mandelbrot(np.ndarray[np.double_t, ndim=1] real, np.ndarray[np.double_t, ndim=1] imag):
    cdef complex z
    cdef int[:,:] outview
    cdef double[:] realview
    cdef double[:] imagview
    cdef np.ndarray out
    cdef int I = real.shape[0]
    cdef int J = imag.shape[0]
    cdef int i
    cdef int j
    
    out = np.zeros((I,J),dtype=np.int)
    outview = out
    realview = real
    imagview = imag
    
    for i in prange(I, nogil=True):
        for j in range(J):
            z = realview[i] + 1j*imagview[j]
            outview[i,j] = mandel(z)
    
    return out
    

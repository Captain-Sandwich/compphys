 # -*- coding: utf-8 -*-
"""
diffusion

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
    double cos(double x) nogil

@cython.boundscheck(False)
cdef inline void mydot(complex[:,:] mat, complex[:] vec) nogil:
    cdef complex park=vec[0]
    vec[0] = park * mat[0,0] + vec[1] * mat[0,1]
    vec[1] = park * mat[1,0] + vec[1] * mat[1,1]


@cython.boundscheck(False)
cdef class Schrodinger1:
    '''Schroedinger Solver in 1D for constant potential'''
    cdef complex[:] psi
    cdef complex[:] expV
    cdef double tau
    cdef double delta
    cdef readonly double T
    cdef int L
    cdef bint even
    cdef complex[:,:] k_matrix
    def __init__(self, np.ndarray[np.complex_t, ndim=1] psi_init, np.ndarray[np.double_t, ndim=1] V, tau, delta):

        #some initialization
        self.tau = tau
        self.psi = psi_init.copy()
        self.L = psi_init.size
        self.even = (self.L % 2 == 0)

        #set up matrix for the product formula
        cdef complex c = cos(-tau/(2*delta*delta))
        cdef complex s = 1j*sin(-tau/(2*delta*delta))
        self.k_matrix = np.array([[c, s], [s, c]])

        #set up potential evolution factors:
        #V = delta**-2 * (1 + delta**2 * V_i)
        self.expV = np.exp(-1j*tau*(1/delta/delta + V)) #TODO: check this

    def probability(self):
        return np.abs(np.asarray(self.psi, dtype=np.complex))**2

    def get_psi(self):
        return np.array(self.psi)

    cdef step(self):
        '''Evolve the wave function one time step'''
        cdef int j
        self.T += self.tau
        with nogil:
            #First K_1 half step
            for j in range(0, self.L-1, 2):
                mydot(self.k_matrix, self.psi[j:j+2])

            #First K_2 half step
            for j in range(1, self.L-1, 2):
                mydot(self.k_matrix, self.psi[j:j+2])

            #V step
            for j in range(self.L):
                self.psi[j] = self.psi[j] * self.expV[j]

            #Second K_2 half step
            for j in range(1, self.L-1, 2):
                mydot(self.k_matrix, self.psi[j:j+2])

            #Second K_2 half step
            for j in range(0, self.L-1, 2):
                mydot(self.k_matrix, self.psi[j:j+2])

    def evolve(self, int n):
        cdef int i
        for i in range(n):
            self.step()

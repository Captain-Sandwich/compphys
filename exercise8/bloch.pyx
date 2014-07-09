 # -*- coding: utf-8 -*-
"""
1D Schrödinger equation solver

@author: bethke
"""
import numpy as np
cimport numpy as np
cimport cython
from cython cimport view
from cython.parallel import parallel, prange

# import exp and sin function from C
cdef extern from "math.h":
    double exp(double x) nogil
    double sin(double x) nogil
    double cos(double x) nogil
    double sqrt(double x) nogil

@cython.boundscheck(False)
cdef inline void mydot(double[:,:] mat, double[:] vec) nogil:
    cdef double p0=vec[0]
    cdef double p1=vec[1]
    vec[0] = p0 * mat[0,0] + p1 * mat[0,1] + vec[2] * mat[0,2]
    vec[1] = p0 * mat[1,0] + p1 * mat[1,1] + vec[2] * mat[1,2]
    vec[2] = p0 * mat[2,0] + p1 * mat[2,1] + vec[2] * mat[2,2]


@cython.boundscheck(False)
cdef class Bloch:
    cdef double[:,:] M
    cdef double invt1
    cdef double invt2
    cdef readonly double T
    cdef object Bfunc
    cdef double tau
    cdef double gamma
    cdef int N
    cdef int i

    cdef double[:,:] b_mat
    cdef double[:] c_mat

    def __init__(self, M_init, invt1, invt2, tau, gamma, T, Bfunc):
        # copy everything into class members
        self.N = np.int(T/tau) # Number of time steps
        self.M = np.zeros((self.N,3))
        cdef int i
        for i in range(3):
            self.M[0,i] = M_init[i]
        self.Bfunc = Bfunc
        self.invt1 = invt1
        self.invt2 = invt2
        self.T = 0 # Time counter
        self.i = 0 # internal step counter
        self.tau = tau
        self.gamma = gamma

        self.b_mat = np.empty((3,3))

        self.c_mat =np.empty(3)
        self.c_mat[0] = exp(-tau*gamma*0.5*invt2)
        self.c_mat[1] = exp(-tau*gamma*0.5*invt2)
        self.c_mat[2] = exp(-tau*gamma*0.5*invt1)

    cdef inline build_b_matrix(self, double[:] b):
        cdef double omega2 = b[0]*b[0] + b[1]*b[1] + b[2]*b[2]
        cdef double omega = sqrt(omega2)
        cdef double bx2 = b[0]*b[0]
        cdef double by2 = b[1]*b[1]
        cdef double bz2 = b[2]*b[2]
        cdef double c = cos(omega * self.tau * self.gamma)
        cdef double s = sin(omega * self.tau * self.gamma)

        self.b_mat[0,0] = (bx2 + (by2 + bz2) * c) / omega2
        self.b_mat[0,1] = (b[0]*b[1]*(1-c) + omega*b[2]*s) / omega2
        self.b_mat[0,2] = (b[0]*b[2]*(1-c) - omega*b[1]*s) / omega2
        self.b_mat[1,0] = (b[0]*b[1]*(1-c) - omega*b[2]*s) / omega2
        self.b_mat[1,1] = (by2 + (bx2 + bz2) * c) / omega2
        self.b_mat[1,2] = (b[1]*b[2]*(1-c) + omega*b[0]*s) / omega2
        self.b_mat[2,0] = (b[0]*b[2]*(1-c) + omega*b[1]*s) / omega2
        self.b_mat[2,1] = (b[1]*b[2]*(1-c) - omega*b[0]*s) / omega2
        self.b_mat[2,2] = (bz2 + (bx2 + by2) * c) / omega2

    def step(self, int i):
        cdef int j
        # get the momentary B-field
        cdef double[:] b = self.Bfunc(self.T+self.tau/2)

        # build B evolution matrix
        self.build_b_matrix(b)

        # decay half-step
        for j in range(3):
            self.M[i, j] = self.c_mat[j] * self.M[i-1, j]

        # B-field step
        mydot(self.b_mat, self.M[i, :])

        # decay half-step
        for j in range(3):
            self.M[i, j] *= self.c_mat[j]

        self.T += self.tau

    def run(self):
        cdef int i
        for i in range(1, self.N):
            self.step(i)
        return np.asarray(self.M)


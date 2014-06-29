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

@cython.boundscheck(False)
cdef inline void mydot(double[:,:] mat, double[:] vec) nogil:
    cdef double park=vec[0]
    vec[0] = park * mat[0,0] + vec[1] * mat[0,1]
    vec[1] = park * mat[1,0] + vec[1] * mat[1,1]


@cython.boundscheck(False)
@cython.cdivision(True)
def cdiffuse(np.ndarray[np.double_t, ndim=1] phi, double tau, int m, double delta, double D, object num_states=0):
    cdef double[:] v
    cdef double expa
    cdef double expa2
    cdef double matval
    cdef double matval2
    cdef np.ndarray expmat
    cdef double[:,:] expmat_v
    cdef np.ndarray expmat2
    cdef double[:,:] expmat2_v
    cdef np.ndarray results
    cdef np.ndarray states
    cdef np.ndarray times
    cdef int i
    cdef int j
    cdef int L=phi.size
    cdef int frame=0
    cdef bint even
    cdef bint full_output
    even = (L % 2 == 0)
    full_output = (num_states > 0)


    # Prepare results matrix
    results = np.empty(m)
    times = np.empty(num_states)
    if num_states:
        states = np.empty((L, num_states))
        states[:,0] = phi.copy()
        times[0] = 0.0

    v = phi
    # first calculate matrices
    matval = np.exp(-2*tau*D/delta/delta)
    matval2 = np.exp(-tau*D/delta/delta)
    expa = np.exp(-tau*D/delta/delta)
    expa2 = np.exp(-tau*D/delta/delta * 0.5)

    expmat = 0.5 * np.array([[1+matval, 1-matval], [1-matval, 1+matval]])
    expmat2 = 0.5 * np.array([[1+matval2, 1-matval2], [1-matval2, 1+matval2]])

    expmat_v = expmat
    expmat2_v = expmat2

    # Perform time evolution
    for i in range(m):
        with nogil:
            # First A half-step
            for j in range(0,L-1,2):
                mydot(expmat2_v, v[j:j+2])

            if not even:
                v[L-1] = v[L-1] * expa2

            # B step
            v[0] = v[0] * expa
            for j in range(1,L-1,2):
                mydot(expmat_v, v[j:j+2])
            if even:
                v[-1] = v[-1] * expa
            else:
                # This is already part of the second A half-step
                v[L-1] = v[L-1] * expa2
            # Second A half-step
            for j in range(0,L-1,2):
                mydot(expmat2_v, v[j:j+2])

        # now sample variance
        results[i] = _cvariance(v, delta)
        if full_output and i > 0 and i % (m//(num_states)) == 0:
            frame +=1
            states[:, frame] = phi.copy()
            times[frame] = (i+1)*tau

    if full_output:
        return results, states, times
    else:
        return results

@cython.boundscheck(False)
def random_walkers(np.ndarray[np.int_t, ndim=1] x, int steps, double delta):
    cdef int N = x.sum()
    cdef int L = x.size
    cdef np.ndarray results
    cdef int[:] v
    cdef int Np=0
    cdef int i
    cdef int j
    cdef int n
    cdef int rS
    results = np.empty(steps)
    for i in range(steps):
        v = np.zeros_like(x)
        for j in x.nonzero()[0]:
            n = x[j]
            r = np.random.randint(0,2,size=n).sum()
            if j != 0 and j != L-1:
                v[j-1] += r
                v[j+1] += n - r
            elif j == 0:
                v[1] += r
            elif j == L-1:
                v[L-2] += n - r
        x = np.copy(v)
        results[i] = _cvariance_b(v, delta)
    return results

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cvariance_b(int[:] x, double delta):
    cdef int N=x.shape[0]
    cdef int i
    cdef int i2
    cdef int i0 = int((N+1)/2)
    cdef double squaresum = 0
    cdef double simplesum = 0
    cdef double total= 0.0


    for i in range(N):
        i2 = (i-i0)*(i-i0)
        simplesum += (i-i0)*x[i]
        squaresum += i2*x[i]
        total += x[i]

    squaresum /= total
    simplesum /= total

    return squaresum - simplesum*simplesum

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cvariance(double[:] x, double delta):
    cdef int N=x.shape[0]
    cdef int i
    cdef int i2
    cdef int i0 = int((N+1)/2)
    cdef double squaresum = 0
    cdef double simplesum = 0
    cdef double total= 0.0


    for i in range(N):
        i2 = (i-i0)*(i-i0)
        simplesum += delta*(i-i0)*x[i]
        squaresum += delta*delta*i2*x[i]
        total += x[i]

    squaresum /= total
    simplesum /= total

    return squaresum - simplesum*simplesum

cpdef cvariance(np.ndarray[np.double_t, ndim=1] x, double delta):
    cdef double[:] v
    v = x
    return _cvariance(v, delta)

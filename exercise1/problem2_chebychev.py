# -*- coding: utf-8 -*-
"""
Program for Chebychev polynomials
"""

import numpy as np
from matplotlib import pyplot as plt


def cheby(x, N):
    """Returns an length(x) by N+1 matrix with values of the chebychev
    polynomials up to the Nth order at position x"""
    # prepare result matrix
    res = np.empty((len(x), N + 1), order='F')

    # fill in  0th order
    res[:, 0] = 1
    if N > 0:
        #fill in 1st order
        res[:, 1] = x
    if N > 1:
        # now the recursion starts
        for i in xrange(2, N+1):
            # recursion formula: $T_{n+1}(x) = 2xT_n(x)-T_{n-1}(x)$
            res[:, i] = 2 * x * res[:, i - 1] - res[:, i - 2]
    return res


if __name__=='__main__':
    # plot chebychev polynomials up to the 4th order
    x = np.linspace(-1, 1, 400)
    data = cheby(x, 4)
    fig = plt.figure()
    for i in xrange(5):
        plt.plot(x, data[:, i], label='n=%d' % i)
    plt.legend(loc=4)
    plt.title('Chebychev Polynomials')
    plt.ylabel('$T_n(x)$')
    plt.xlabel('$x$')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig('latex/images/chebychev.pdf')

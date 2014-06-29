# -*- coding: utf-8 -*-
"""
program for problem 1
"""

import numpy as np

#problem 1a: create a 6x6 matrix with random numbers.
np.random.seed(9905)  # seed the RNG
A = (np.random.rand(6,6) - 0.5) * 6 # 6x6 matrix of random doubles

#problem 1b: find maximum of A and its indices
max_value = np.max(A)
indices = np.unravel_index(np.argmax(A), A.shape)
print 'The maximum of A is in row %d, column %d. Its value is %d.' % \
        (indices[0], indices[1], max_value)

#problem 1c: row and column vectors
row_max = np.max(A, axis=0)
col_max = np.max(A, axis=1)
max_product = np.dot(row_max, col_max)
print 'col_max:', col_max
print 'row_max:', row_max
print 'max_product:', max_product

#problem 1d: multiply A and B
B = (np.random.rand(6,6) - 0.5) * 6 # 6x6 matrix of random doubles
C = np.dot(A, B)
D = np.dot(B, A)

print 'C:', C
print 'D:', D
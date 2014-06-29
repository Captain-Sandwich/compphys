# -*- coding: utf-8 -*-
"""
problem 3: integer overflow
"""

import numpy as np
import matplotlib.pyplot as plt

#python does not know 8-bit integers, so I use a 1x1 numpy array
x = np.zeros(1, dtype='uint8')

result = np.zeros(300)
for i in xrange(300):
    x[0] = x[0] + 1
    result[i] = x[0]

# plot results
fig = plt.figure()
plt.plot(result,marker='+',ls='')
plt.xlabel('iteration step $i$')
plt.ylabel('$x$')
plt.title('integer overflow')
plt.tight_layout()
plt.savefig('latex/images/overflow.pdf')
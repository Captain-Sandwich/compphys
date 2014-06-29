# -*- coding: utf-8 -*-
"""
utility functions for computational physics

@author: bethke
"""

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def array2latex(A):
    """Print latex markup for matrix A"""
    shape = A.shape
    result = []
    for i in xrange(shape[0]):
        result.append(' & '.join([str(x) for x in A[:,i]]))
    return ' \\\\\n'.join(result)
 
def cm2colors(N, cmap='autumn'):
    """Takes N evenly spaced colors out of cmap and returns a
    list of rgb values"""
    values = range(N)
    cNorm  = Normalize(vmin=0, vmax=N-1)
    scalarMap = ScalarMappable(norm=cNorm, cmap=cmap)
    colors = []
    for i in xrange(N):
        colors.append(scalarMap.to_rgba(values[i]))
    return colors

def show_colors(colors):
    N = len(colors)
    colors = np.array(colors)
    colors = colors.reshape((N,1,4))
    plt.imshow(colors,interpolation='nearest')

def set_plot_options():
    rc('legend', fancybox=True, handlelength=1.2, borderaxespad=0.25, borderpad=0.2,
       fontsize=7, columnspacing=1.0)
    rc('font', size=11, family='Palatino Linotype')
    rc('axes', color_cycle=['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33', 'a65628'],
       labelsize=7, linewidth=0.4)
    rc('figure', figsize=(2.83509486, 1.75208862348)) # one column size
    rc('xtick', direction='out', labelsize=7)
    rc('xtick.major', width=0.4, size=4)
    rc('ytick', direction='out', labelsize=7)
    rc('ytick.major', width=0.4, size=4)
    rc('grid', alpha=0.5, linestyle='-', linewidth=0.2)
    rc('lines', markeredgewidth=0.2, markersize=2, linewidth=0.5)
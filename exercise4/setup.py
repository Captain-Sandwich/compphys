# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:48:05 2014

@author: bethke
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension('integrators',
                  ['integrators.pyx'],
                  include_dirs=[numpy.get_include()],
        ),
])
#!/usr/bin/env python
"""pyci setup
"""
import os
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext


__version__ = ''
exec(open('pyci/_version.py').read())

EXT_MODULES = []

# Package setup commands 
setup(name = 'pyci',
      version = __version__,
      python_requires = '>=3.7',
      requires = ['numpy (>=1.22)',  # need to check other requirements like matplotlib, itertools
                  'scipy (>=1.0)',
                  'psi4 (>=1.4)', 
                  'cython (>=0.21)',
                  'opt_einsum (>=3.3.0)'],  
      packages = find_packages(),
      ext_modules = cythonize(EXT_MODULES),
      cmdclass = {'build_ext': build_ext},
      author = 'Sai Vijay Mocherla',
      author_email = 'vijaysai.mocherla@gmail.com',
      license = 'MIT',
      description = 'A python module that demonstrates Configuration interaction Calculations',
      long_description  = '',
      keywords = 'configuration-interaction, electron dynamics, electronic-structure',
      url = 'https://github.com/vijaymocherla/pyci'
)

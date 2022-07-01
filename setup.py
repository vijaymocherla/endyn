#!/usr/bin/env python
"""pyci setup
"""
import os
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy 
import scipy

__version__ = ''
exec(open('pyci/__init__.py').read())

EXT_MODULES = [
            Extension("pyci.configint.cy.rcisd_core",
                  sources=["pyci/configint/cy/rcisd_core.pyx"],
                  include_dirs = [numpy.get_include(),scipy.get_include()],
                  extra_compile_args = ["-O3", "-funroll-loops"],), 

            Extension("pyci.lib.configint.rcisd",
                  sources=["pyci/lib/configint/rcisd.c"],
                  language="C",
                  extra_compile_args = ["-O3", "-funroll-loops", "-Wunused-but-set-variable", "-lm", "-std=c11"],)
]

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

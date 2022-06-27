from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy

ext_modules = [
    Extension(
        "rcisd_core",
        ["rcisd_core.pyx"],
        extra_compile_args = ["-O3", "-funroll-loops"],
    ),
]

setup(
    name='cyext',
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs = [numpy.get_include(), scipy.get_include()]
)
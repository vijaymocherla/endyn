from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



EXT_MODULES=[
    Extension("rungekutta", ["rungekutta.pyx"]),
]

setup(
  name = 'intergrators',
  cmdclass = {'build_ext': build_ext},
  ext_modules = EXT_MODULES,
)
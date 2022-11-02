from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



EXT_MODULES=[
    Extension("rungekutta", ["rungekutta.pyx"]),
    Extension("splitoperator", ["splitoperator.pyx"]),
    Extension("cranknicholson", ["cranknicholson.pyx"])]

for e in EXT_MODULES:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
  name = 'intergrators',
  cmdclass = {'build_ext': build_ext},
  ext_modules = EXT_MODULES,
)

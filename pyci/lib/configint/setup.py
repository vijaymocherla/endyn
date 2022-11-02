from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



EXT_MODULES=[
    Extension("rcisd", ["rcisd.pyx"]),
    Extension("cis_d", ["fast_cis_d.pyx"]),
]

for e in EXT_MODULES:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
  name = 'configint',
  python_requires = '>=3.7',
  cmdclass = {'build_ext': build_ext},
  ext_modules = EXT_MODULES,
)

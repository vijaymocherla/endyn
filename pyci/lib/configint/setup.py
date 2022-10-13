from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



EXT_MODULES=[
    Extension("rcisd", ["rcisd.pyx"]),
    Extension("cis_d", ["fast_cis_d.pyx"])
]

setup(
  name = 'configint',
  cmdclass = {'build_ext': build_ext},
  ext_modules = EXT_MODULES,
)

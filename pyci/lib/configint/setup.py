from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
        Extension("rcisd",
        sources=["rcisd.c"],
        language="C",
        extra_compile_args = ["-O3", "-funroll-loops", 
                              "-Wunused-but-set-variable", "-lm", "-std=c11"])
]

setup(
    name="rcisd",
    ext_modules=cythonize(ext_modules, language_level="3")
)
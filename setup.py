from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["experiment.pyx", "dynaq.pyx", "dynaqplus.pyx"])
)
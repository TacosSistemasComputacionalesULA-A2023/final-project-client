from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["./src/experiment.pyx", "./src/agents/dynaq.pyx", "./src/agents/dynaqplus.pyx"])
)
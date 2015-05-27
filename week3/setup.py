from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'pimc',
  ext_modules = cythonize("pimc.pyx"),
  include_dirs=[numpy.get_include()]
)
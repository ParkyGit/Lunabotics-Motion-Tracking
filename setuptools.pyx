from setuptools import setup
from Cython.Build import cythonize

class File:
  pccw = "pccw_video.py"
  pccRed = "pcc(withRedFilter).py"
  pccNoRed = "pcc(w_oRedFilter).py"
  Centroid = "centroid_finder.py"
  Statics = "Static_tests.py"

setup(
    name='Lunabotics Cython',
    ext_modules=cythonize(File.pccw),
    zip_safe=False,
)

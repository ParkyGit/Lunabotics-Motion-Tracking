from setuptools import setup, find_packages
from Cython.Build import cythonize

class File:
  pccw = "pccw_video.py"
  pccRed = "pcc(withRedFilter).py"
  pccNoRed = "pcc(w_oRedFilter).py"
  Centroid = "centroid_finder.py"
  Statics = "Static_tests.py"
  Dft = "DFT_examination.py"

members = vars(File).items()
members = [x[1] for x in members if "__" not in x[0]]

setup(
    name='Lunabotics Cython',
    ext_modules=cythonize("*.pyx"),
    zip_safe=False,
)

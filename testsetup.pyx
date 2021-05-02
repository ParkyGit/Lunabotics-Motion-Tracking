from setuptools import setup, find_packages
from Cython.Build import cythonize

pccw = "pccw_video.py"
pccRed = "pcc(withRedFilter).py"
pccNoRed = "pcc(w_oRedFilter).py"
Centroid = "centroid_finder.py"
Statics = "Static_tests.py"
Dft = "DFT_examination.py"


setup(
    name='Lunabotics Cython',
    ext_modules=cythonize(pccNoRed),
    zip_safe=False,
)

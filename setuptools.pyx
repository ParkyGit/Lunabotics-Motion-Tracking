from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Lunabotics Cython',
    ext_modules=cythonize("pcc(w_oRedFilter).py"),
    zip_safe=False,
)

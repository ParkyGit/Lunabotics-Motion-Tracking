from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Lunabotics Cython',
    ext_modules=cythonize("pccw_video.py"),
    zip_safe=False,
)

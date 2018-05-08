from platform import system
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy import get_include as get_numpy_include

extensions = [
    Extension("datasets.kitti.box_overlaps", ["datasets/kitti/box_overlaps.pyx"],
        include_dirs=[get_numpy_include()])
]

setup(
    name="VoxelDetection",
    ext_modules=cythonize(extensions),
)

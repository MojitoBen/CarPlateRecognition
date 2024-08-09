from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [
    Extension(
        "bbox_iou",
        sources=["bbox_iou.pyx"],
        include_dirs=[np.get_include()],
        # other options...
    ),
    # other extensions...
]

setup(
    ext_modules = cythonize(ext_modules)
)
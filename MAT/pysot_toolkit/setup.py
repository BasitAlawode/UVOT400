from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='utils.region',
        sources=[
            'utils/region.pyx',
            'utils/src/region.c',
        ],
        include_dirs=[
            'utils/src'
        ]
    )
]

setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(ext_modules)
)

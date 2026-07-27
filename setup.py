from setuptools import setup
from nimporter_plus import *

setup(
    name = 'neat',
    py_modules = ['neat'],
    ext_modules = build_nim_extensions(exclude_dirs = ['tests'])
)

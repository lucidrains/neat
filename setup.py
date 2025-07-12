from setuptools import setup
from nimporter import *

setup(
    name = 'neat',
    py_modules = ['neat'],
    ext_modules = build_nim_extensions()
)

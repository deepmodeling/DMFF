import os
import re
import sys
import platform
import subprocess
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        subprocess.check_call(['cmake', '-B', 'build','-DCUDA_NVCC_FLAGS=61'] + cmake_args)
        subprocess.check_call(['cmake', '--build', 'build'])

setuptools.setup(
    name='dpnblist',
    version='1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A test project using pybind11 and CMake',
    long_description='',
    ext_modules=[CMakeExtension('dpnblist')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

#!/usr/bin/env python3
"""
Setup script for Factorama Python bindings

This setup.py allows installing the Python bindings using pip after building
with CMake. It assumes the compiled extension module is already built.
"""

import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
from pathlib import Path

__version__ = "1.0.0"

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "factorama._factorama",
        ["python/factorama/_factorama.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to factorama headers
            "../src",
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="factorama",
    version=__version__,
    author="Factorama Contributors",
    author_email="",
    url="https://github.com/your-repo/factorama",
    description="Python bindings for the Factorama factor graph optimization library",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=["factorama"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
#!/usr/bin/env python3
"""
Setup script for Factorama Python bindings

This setup.py integrates with the CMake build system and automatically
generates type stubs for VSCode intellisense.
"""

import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from pathlib import Path

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        generate_stubs()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        generate_stubs()

def generate_stubs():
    """Generate type stubs using pybind11-stubgen"""
    try:
        print("Generating type stubs...")
        result = subprocess.run([
            sys.executable, "-m", "pybind11_stubgen",
            "factorama",
            "-o", "python",
            "--root-suffix", "",
            "--ignore-invalid-expressions", ".*"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Type stubs generated successfully!")
        else:
            print(f"Warning: Could not generate stubs: {result.stderr}")
    except Exception as e:
        print(f"Warning: Could not generate stubs: {e}")

__version__ = "1.0.0"

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
    package_data={"factorama": ["*.so", "*.pyd", "*.pyi"]},  # Include compiled extensions and stubs
    extras_require={"test": "pytest", "stubs": "pybind11-stubgen>=0.10.0"},
    cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand},
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
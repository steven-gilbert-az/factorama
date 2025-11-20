#!/bin/bash

# install_python.sh - Build and install Factorama Python package from scratch

set -e  # Exit on any error

echo "Building and installing Factorama Python package..."

# Clean any existing build directories
echo "Cleaning previous builds..."
rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf _skbuild

# Update pip and install/upgrade build dependencies to avoid version conflicts
echo "Updating build dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade "scikit-build-core>=0.10" "pybind11[global]>=2.11" pybind11-stubgen

# Build and install the package using pip (which will use scikit-build-core)
echo "Building and installing Python package..."
python3 -m pip install -e . --no-build-isolation

# Generate type stubs for the C++ bindings
echo "Generating type stubs..."
pybind11-stubgen factorama._factorama -o python_bindings/python/

echo "Installation complete! You can now 'import factorama' in Python."

![Factorama Banner](media/factorama_banner.jpg)

# Factorama
A factor graph library with emphasis on simple usage. Suitable for small-medium SLAM, calibration, and SFM problems.

Available for both **Python** and **C++**.

## Features

- **Factor Graph Framework**: Flexible factor graph implementation for bundle adjustment
- **Variable Types**: Support for 6-DOF poses, 3D landmarks, inverse depth parameterization, and extrinsic calibration
- **Sparse Matrix Support**: Uses Eigen sparse matrices to speed up computation
- **Solvers**: Levenberg-Marquardt and Gauss-Newton algorithms
- **Multiple Factor Types**: Bearing observations, pose priors, relative constraints, and camera-IMU alignment

## Quick Start

### Python

Install via pip:
```bash
pip install factorama
```

Basic usage:
```python
import factorama
from factorama import FactorGraph, PoseVariable, LandmarkVariable

# Create factor graph
graph = FactorGraph()

# Add variables and factors
# ... (see Python guide for details)

# Optimize
graph.finalize_structure()
optimizer = factorama.SparseOptimizer()
optimizer.setup(graph, settings)
optimizer.optimize()
```

**[→ Full Python Documentation](README_PYPI.md)**

### C++

Build from source:
```bash
git clone https://github.com/steven-gilbert-az/factorama.git
cd factorama
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

**[→ Full C++ Documentation](README_CPP.md)**

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, coding standards, and contribution workflow.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use this library in academic work, please cite:

```
[Citation information to be added]
```
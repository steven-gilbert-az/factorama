![Factorama Banner](media/factorama_banner.jpg)

# Factorama
A factor graph library with emphasis on simple usage. Suitable for small-medium SLAM, calibration, and SFM problems.

A C++17 library for factor graph-based optimization using non-linear least squares. Built for simplicity and ease of use.

## Features

- **Factor Graph Framework**: Flexible factor graph implementation for bundle adjustment
- **Variable Types**: Support for 6-DOF poses, 3D landmarks, inverse depth parameterization, and extrinsic calibration
- **Sparse Matrix Support**: uses Eigen sparse matrices to speed up computation
- **Solvers**: Levenberg-Marquardt and Gauss-Newton algorithms
- **Multiple Factor Types**: Bearing observations, pose priors, relative constraints, and camera-IMU alignment
- **Testing**: Unit tests and integration tests using Catch2

## Quick Start

### Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Eigen3 3.4 or higher
- Catch2 3.x (for testing, optional)

### Build Instructions

```bash
# Clone and build
git clone <repository-url>
cd factorama
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest
```

### Installation

```bash
# Install to system (after building)
sudo make install
```

For installation without tests:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j$(nproc)
sudo make install
```

### Basic Usage

```cpp
#include "factorama/factor_graph.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/bearing_observation_factor.hpp"
#include "factorama/sparse_optimizer.hpp"

using namespace factorama;

// Create variables
Eigen::Matrix<double, 6, 1> pose_vec = Eigen::Matrix<double, 6, 1>::Zero();
auto camera_pose = std::make_shared<PoseVariable>(1, pose_vec);

Eigen::Vector3d landmark_pos(0.0, 0.0, 5.0);
auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

// Create factor
Eigen::Vector3d bearing_vector(0.0, 0.0, 1.0);
auto factor = std::make_shared<BearingObservationFactor>(
    0, camera_pose, landmark, bearing_vector, 1.0);

// Build factor graph
FactorGraph graph;
graph.add_variable(camera_pose);
graph.add_variable(landmark);
graph.add_factor(factor);
graph.finalize_structure();

// Configure optimizer
OptimizerSettings settings;
settings.method = OptimizerMethod::GaussNewton;
settings.max_num_iterations = 100;
settings.step_tolerance = 1e-6;
settings.verbose = true;

// Run optimization
auto graph_ptr = std::make_shared<FactorGraph>(graph);
SparseOptimizer optimizer;
optimizer.setup(graph_ptr, settings);
optimizer.optimize();
```

## Project Structure

```
factorama/
├── src/factorama/          # Core library implementation
│   ├── types.hpp            # Base variable and factor interfaces
│   ├── factor_graph.hpp     # Factor graph container and operations
│   ├── sparse_optimizer.hpp # Sparse optimization algorithms
│   ├── *_variable.hpp       # Variable implementations
│   └── *_factor.hpp         # Factor implementations
├── unit_test.cpp           # Unit tests
├── integration_test.cpp    # Integration tests
└── CMakeLists.txt          # Build configuration
```

## Core Components

### Variables

- **PoseVariable**: 6-DOF camera poses with SE(3) parameterization
- **LandmarkVariable**: 3D world landmarks in Euclidean space  
- **InverseRangeVariable**: Inverse depth parameterization (1 degree of freedom)
- **RotationVariable**: Rotation-only variables for calibration
- **GenericVariable**: Generic N-dimensional linear variable (uses Eigen::VectorXd and MatrixXd to achieve flexibility without over-templatization)

### Factors

- **BearingObservationFactor**: Camera bearing measurements to 3D landmarks
- **InverseRangeBearingFactor**: Bearing constraints with inverse depth parameterization
- **PosePositionPriorFactor**: Position prior constraints
- **PoseOrientationPriorFactor**: Orientation prior constraints  
- **GenericPriorFactor**: Generic prior constraints for any variable type
- **GenericBetweenFactor**: Relative constraints between variables
- **PosePositionBetweenFactor**: Position-only relative constraints
- **PoseOrientationBetweenFactor**: Orientation-only relative constraints

### Optimization

The library provides optimization via the `SparseOptimizer` class (or just use the Jacobian/residual/update functions and bring your own optimizer):

- **Sparse Linear Algebra**: Uses Eigen's sparse matrix operations for efficient computation
- **Algorithms**: Gauss-Newton and Levenberg-Marquardt
- **Configurable Settings**: Comprehensive optimization parameters

```cpp
OptimizerSettings settings;

// Algorithm selection
settings.method = OptimizerMethod::GaussNewton;        // or LevenbergMarquardt

// Convergence criteria
settings.max_num_iterations = 100;                     // Maximum iterations
settings.step_tolerance = 1e-6;                        // ||dx|| threshold
settings.residual_tolerance = 1e-6;                    // Residual improvement threshold

// Levenberg-Marquardt damping parameters
settings.initial_lambda = 1e-3;                        // Initial damping
settings.max_lambda = 1e5;                             // Maximum damping (prevents runaway)
settings.lambda_up_factor = 10.0;                      // Damping increase factor
settings.lambda_down_factor = 0.1;                     // Damping decrease factor

// Gauss-Newton step control
settings.learning_rate = 1.0;                          // Step size (1.0 = full step)

// Debugging and diagnostics
settings.verbose = false;                              // Enable iteration logging
settings.check_rank_deficiency = false;                // Enable rank analysis (slower)
```

## Build Options

Configure the build with these CMake options:

```bash
# Enable GTSAM benchmark comparison (requires GTSAM)
cmake .. -DGTSAM=ON

# Disable test building
cmake .. -DBUILD_TESTS=OFF

```

## Dependencies

### Required
- **Eigen3 3.4+**: Linear algebra and matrix operations
- **C++17**: Modern C++ features and standard library

### Optional
- **Catch2 3.x**: Unit testing framework
- **GTSAM**: For benchmark comparisons

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install libeigen3-dev libcatch2-dev
```

## Examples

### Bundle Adjustment Examples

See `integration_test.cpp` for examples including:
1. Basic stereo camera bundle adjustment
2. Inverse depth parameterization scenarios
3. Camera-IMU calibration with relative orientation factors
4. Prior factor constraints


## Testing


```bash
# Run all tests
cd build && ctest

# Run specific test suites
./unit_test                # Basic functionality tests
./integration_test         # End-to-end scenario tests  
./optimizer_test          # Optimization algorithm tests
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, coding standards, and contribution workflow.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use this library in academic work, please cite:

```
[Citation information to be added]
```
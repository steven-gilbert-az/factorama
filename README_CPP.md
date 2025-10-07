# Factorama C++

A C++17 library for factor graph-based optimization using non-linear least squares. Built for simplicity and ease of use.

## Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Eigen3 3.4 or higher
- Catch2 3.x (for testing, optional)

## Build Instructions

```bash
# Clone and build
git clone https://github.com/steven-gilbert-az/factorama.git
cd factorama
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest
```

## Installation

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

## Basic Usage

See [EXAMPLES.md](EXAMPLES.md) for usage examples.

## Project Structure

```
factorama/
├── src/factorama/          # Core library implementation
│   ├── base_types.hpp            # Base variable and factor interfaces
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

# Contributing to factorama

Thank you for your interest in contributing to factorama! This document provides guidelines for contributing to the project.

## Development Environment

See [README.md](README.md) for prerequisites and basic build instructions.

### Setting Up Development Environment

```bash
# Development build with tests enabled
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
ctest  # Verify setup
```

## Code Style and Standards

### C++ Standards

- **C++ Version**: Use C++17 features and standard library
- **Compiler Warnings**: Code must compile without warnings using `-Wall -Wextra -Wpedantic`
- **Header Guards**: Use `#pragma once` for header files
- **Namespaces**: All code should be in the `factorama` namespace

### Naming Conventions

- **Variables/Functions**: `snake_case`
  ```cpp
  int variable_name;
  void function_name();
  double constant_value = 1e-9;  // Constants also use snake_case
  ```

- **Classes/Types**: `PascalCase`
  ```cpp
  class PoseVariable;
  enum VariableType;
  ```

- **Private Members**: Trailing underscore
  ```cpp
  class MyClass {
  private:
      int member_variable_;
  };
  ```

### Braces and Control Structures

Always use braces for control structures, even single statements:

```cpp
// Good
if (condition) {
    do_something();
}

// Bad - no unguarded statements
if (condition)
    do_something();
```

### Rotation Matrix Naming Convention

**Important**: Any rotation matrices should be named `dcm_XY` where X and Y are the names of two coordinate frames:

- `dcm_XY` represents the direction cosine matrix that will rotate a matrix from frame Y to frame X
- This can be read as "dcm from Y to X" or "dcm to X from Y"
- e.g. `dcm_CW` is the rotation from World to Camera
- This notation is convenient for chaining many rotations together
  - e.g. `dcm_CW * dcm_WB * dcm_BA * dcm_AX = dcm_CX`
- It also helps checking consistency (all inner letters must match when chaining rotations together)

```cpp
// Good
Eigen::Matrix3d dcm_CW;  // Camera from World
Eigen::Matrix3d dcm_WC = dcm_CW.transpose();  // World from Camera

// Bad
Eigen::Matrix3d R_camera_world;  // Ambiguous direction
```

## Testing

### Test Organization

- **Unit Tests**: `unit_test.cpp` - Test individual components
- **Integration Tests**: `integration_test.cpp` - Test end-to-end scenarios
- **Optimizer Tests**: `optimizer_test.cpp` - Test optimization algorithms

### Writing Tests

Use Catch2 framework with descriptive test names:

```cpp
TEST_CASE("PoseVariable correctly applies SE3 increments", "[pose][variable]")
{
    SECTION("Identity pose with small rotation increment")
    {
        // Test implementation
        REQUIRE(some_condition);
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-9));
    }
}
```

### Test Requirements

- New features must include unit tests
- Use appropriate numerical tolerances (typically 1e-9 to 1e-12)
- Test both normal operation and edge cases

## Adding New Components

### Adding a New Variable Type

1. Create header in `src/factorama/my_variable.hpp`
2. Inherit from `Variable` and implement all virtual methods
3. Add to `VariableType` enum in `types.hpp`
4. Add comprehensive unit tests

### Adding a New Factor Type

1. Create header/source in `src/factorama/my_factor.hpp/.cpp`
2. Inherit from `Factor` and implement all virtual methods
3. Provide analytical Jacobians (no numerical differentiation)
4. Add to `FactorType` enum in `types.hpp`
5. Test residual and Jacobian computation
6. Verify Jacobians with finite differences in tests

## Development Build Commands

```bash
# Development build with all tests
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest
./unit_test
./integration_test
./optimizer_test
```

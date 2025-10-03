# Factorama Python

A Python interface to the Factorama C++ factor graph optimization library. Factorama provides a simple and efficient framework for factor graph-based optimization, perfect for small to medium-scale SLAM, calibration, and structure-from-motion problems.

## Purpose

Factor graphs are a powerful framework for non-linear optimization problems commonly found in robotics and computer vision. Factorama's Python bindings make it easy to:

- **Build factor graphs** with poses, landmarks, and custom variables
- **Add constraints** through various factor types (bearing observations, priors, relative constraints)
- **Optimize** using Gauss-Newton or Levenberg-Marquardt algorithms
- **Visualize results** with built-in plotting capabilities

Perfect for prototyping SLAM algorithms, camera calibration, bundle adjustment, and sensor fusion applications.

## Installation

```bash
pip install factorama
```

### Dependencies

Factorama requires:
- **numpy** - For matrix operations and numerical arrays
- **matplotlib** - For visualization and plotting (optional, but recommended)

## Examples

### Basic Robot Localization

```python
from factorama import FactorGraph, PoseVariable, LandmarkVariable, PosePositionPriorFactor, PoseOrientationPriorFactor, BearingObservationFactor

# Create factor graph and add variables
graph = FactorGraph()
robot_pose = PoseVariable(1, initial_pos, initial_dcm)
landmark = LandmarkVariable(2, landmark_pos)
graph.add_variable(robot_pose)
graph.add_variable(landmark)

# Add factors
position_prior = PosePositionPriorFactor(100, robot_pose, prior_pos, 0.1)
orientation_prior = PoseOrientationPriorFactor(101, robot_pose, prior_dcm, 0.1)
bearing_factor = BearingObservationFactor(200, robot_pose, landmark, bearing_obs, 0.01)
graph.add_factor(position_prior)
graph.add_factor(orientation_prior)
graph.add_factor(bearing_factor)

# Optimize
graph.finalize_structure()
optimizer = SparseOptimizer()
settings = OptimizerSettings()
settings.method = OptimizerMethod.LevenbergMarquardt
optimizer.setup(graph, settings)
optimizer.optimize()
```

**[→ Complete runnable example: examples/basic_localization.py](examples/basic_localization.py)**

### Bundle Adjustment with Multiple Views

```python
from factorama import FactorGraph, PoseVariable, LandmarkVariable, PosePositionPriorFactor, PoseOrientationPriorFactor, BearingObservationFactor, PlotFactorGraph

# Create factor graph with multiple poses and landmarks
graph = FactorGraph()

# Add camera poses
poses = []
for i in range(3):
    pose = PoseVariable(i + 1, pose_pos, pose_dcm)
    poses.append(pose)
    graph.add_variable(pose)

# Add landmarks and priors
landmarks = []
for i, pos in enumerate(landmark_positions):
    landmark = LandmarkVariable(10 + i, pos)
    landmarks.append(landmark)
    graph.add_variable(landmark)

# Add bearing observations between all pose-landmark pairs
for pose in poses:
    for landmark in landmarks:
        factor = BearingObservationFactor(factor_id, pose, landmark, bearing, 0.01)
        graph.add_factor(factor)

# Optimize and visualize
graph.finalize_structure()
optimizer.optimize()
PlotFactorGraph(graph)
```

**[→ Complete runnable example: examples/bundle_adjustment.py](examples/bundle_adjustment.py)**

### Inverse Depth Parameterization

```python
import numpy as np
from factorama import FactorGraph, PoseVariable, InverseRangeVariable, InverseRangeBearingFactor, SparseOptimizer, OptimizerSettings

graph = FactorGraph()

# Camera pose
camera_pose = PoseVariable(1, camera_pos, camera_dcm)
graph.add_variable(camera_pose)

# Inverse depth landmark (origin, bearing direction, initial range)
origin_pos = np.array([0.0, 0.0, 0.0])
bearing_direction = np.array([1.0, 0.0, 0.0])
initial_range = 10.0
inv_depth_landmark = InverseRangeVariable(2, origin_pos, bearing_direction, initial_range)
graph.add_variable(inv_depth_landmark)

# Bearing observation factor
bearing_obs = np.array([1.0, 0.0, 0.0])
bearing_factor = InverseRangeBearingFactor(100, camera_pose, inv_depth_landmark, bearing_obs, 0.01)
graph.add_factor(bearing_factor)

# Optimize
graph.finalize_structure()
optimizer = SparseOptimizer()
settings = OptimizerSettings()
optimizer.setup(graph, settings)
optimizer.optimize()

print(f"Final landmark position: {inv_depth_landmark.pos_W()}")
```

## Variables

### PoseVariable
Represents SE(3) poses with 6 DOF (position + orientation)
```python
from factorama import PoseVariable

# From position and rotation matrix
pose = PoseVariable(id, position_3d, rotation_matrix_3x3)

# Alternative: From SE(3) vector [tx, ty, tz, rx, ry, rz]
pose = PoseVariable(id, pose_vector)

# Access position and rotation
position = pose.pos_W()
rotation_matrix = pose.dcm_CW()
```

### LandmarkVariable
Represents 3D landmarks with 3 DOF
```python
from factorama import LandmarkVariable

landmark = LandmarkVariable(id, position_3d)
position = landmark.pos_W()
```

### GenericVariable
Represents arbitrary N-dimensional variables
```python
from factorama import GenericVariable

generic = GenericVariable(id, initial_vector)
```

### RotationVariable
Represents SO(3) rotations with 3 DOF
```python
from factorama import RotationVariable

rotation = RotationVariable(id, rotation_matrix_3x3)
dcm = rotation.dcm_AB()
```

### InverseRangeVariable
Represents landmarks using inverse depth parameterization (1 DOF)
```python
from factorama import InverseRangeVariable

inv_range = InverseRangeVariable(id, origin_pos, bearing_direction, initial_range)
position = inv_range.pos_W()
inverse_depth = inv_range.inverse_range()
```

## Factors

### Prior Factors
- **GenericPriorFactor**: Prior constraints on any variable type
- **PosePositionPriorFactor**: Position-only prior for poses
- **PoseOrientationPriorFactor**: Orientation-only prior for poses
- **RotationPriorFactor**: Prior constraints on rotation variables

### Observation Factors
- **BearingObservationFactor**: 3D bearing measurements from poses to landmarks
- **InverseRangeBearingFactor**: Bearing constraints with inverse depth parameterization
- **BearingProjectionFactor2D**: 2D bearing projections

### Relative Constraint Factors
- **GenericBetweenFactor**: Relative constraints between any variable types
- **PosePositionBetweenFactor**: Position-only relative constraints between poses
- **PoseOrientationBetweenFactor**: Orientation-only relative constraints between poses

## Optimization

```python
from factorama import SparseOptimizer, OptimizerSettings, OptimizerMethod

# Create optimizer
optimizer = SparseOptimizer()

# Configure settings
settings = OptimizerSettings()
settings.method = OptimizerMethod.LevenbergMarquardt  # or GaussNewton
settings.max_num_iterations = 100
settings.step_tolerance = 1e-6
settings.residual_tolerance = 1e-6
settings.verbose = True

# Setup and optimize
optimizer.setup(factor_graph, settings)
optimizer.optimize()

# Access results
print(f"Iterations: {optimizer.current_stats.current_iteration}")
print(f"Final chi2: {optimizer.current_stats.chi2}")
```

## Utility Functions

```python
from factorama import ExpMapSO3, LogMapSO3, PlotFactorGraph

# SO(3) exponential and logarithm maps
rotation_matrix = ExpMapSO3(rotation_vector)
rotation_vector = LogMapSO3(rotation_matrix)

# Factor graph visualization
PlotFactorGraph(graph, plot_3d=True)
```
import numpy as np
from factorama import FactorGraph, PoseVariable, LandmarkVariable, PosePositionPriorFactor, PoseOrientationPriorFactor, BearingObservationFactor, SparseOptimizer, OptimizerSettings, OptimizerMethod, PlotFactorGraph
import matplotlib.pyplot as plt


# Create factor graph
graph = FactorGraph()

# Add multiple camera poses
poses = []
for i in range(3):
    # Initial poses with small perturbation
    pose_pos = np.array([i * 2.0, 0.0, 0.0])
    pose_dcm = np.eye(3)  # Identity rotation
    pose = PoseVariable(i + 1, pose_pos, pose_dcm)
    poses.append(pose)
    graph.add_variable(pose)

# Add landmarks
landmarks = []
landmark_positions = [
    np.array([5.0, 5.0, 10.0]),
    np.array([5.0, -5.0, 10.0]),
    np.array([0.0, 0.0, 15.0])
]

for i, pos in enumerate(landmark_positions):
    landmark = LandmarkVariable(10 + i, pos)
    landmarks.append(landmark)
    graph.add_variable(landmark)

# Add pose priors
for i, pose in enumerate(poses):
    prior_pos = np.array([i * 2.0, 0.0, 0.0])
    prior_dcm = np.eye(3)
    position_prior = PosePositionPriorFactor(50 + i*2, pose, prior_pos, 0.1)
    orientation_prior = PoseOrientationPriorFactor(50 + i*2 + 1, pose, prior_dcm, 0.1)
    graph.add_factor(position_prior)
    graph.add_factor(orientation_prior)

# Add bearing observations (each camera sees all landmarks)
factor_id = 100
for pose_idx, pose in enumerate(poses):
    for landmark_idx, landmark in enumerate(landmarks):
        # Compute expected bearing from pose to landmark
        pose_pos = pose.pos_W()
        landmark_pos = landmark.pos_W()
        bearing = landmark_pos - pose_pos
        bearing = bearing / np.linalg.norm(bearing)

        # Add small noise
        bearing += np.random.normal(0, 0.01, 3)
        bearing = bearing / np.linalg.norm(bearing)

        factor = BearingObservationFactor(
            factor_id, pose, landmark, bearing, 0.01)
        graph.add_factor(factor)
        factor_id += 1

# Optimize
graph.finalize_structure()
optimizer = SparseOptimizer()
settings = OptimizerSettings()
settings.method = OptimizerMethod.GaussNewton
optimizer.setup(graph, settings)
optimizer.optimize()

# Visualize results
PlotFactorGraph(graph)

plt.show()
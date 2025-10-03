import numpy as np
from factorama import FactorGraph, PoseVariable, LandmarkVariable, PosePositionPriorFactor, PoseOrientationPriorFactor, BearingObservationFactor, SparseOptimizer, OptimizerSettings, OptimizerMethod

# Create factor graph
graph = FactorGraph()

# Add pose variable
initial_pos = np.array([0.1, 0.1, 0.0])
initial_dcm = np.eye(3)  # Identity rotation
pose_var = PoseVariable(1, initial_pos, initial_dcm)
graph.add_variable(pose_var)

# Add landmark variables
landmark_pos = np.array([10.0, 0.0, 0.0])
landmark = LandmarkVariable(2, landmark_pos)
graph.add_variable(landmark)

# Add pose priors
prior_pos = np.array([0.0, 0.0, 0.0])
prior_dcm = np.eye(3)
pose_position_prior = PosePositionPriorFactor(100, pose_var, prior_pos, 0.1)
pose_orientation_prior = PoseOrientationPriorFactor(101, pose_var, prior_dcm, 0.1)
graph.add_factor(pose_position_prior)
graph.add_factor(pose_orientation_prior)

# Add bearing observation factor
bearing_obs = np.array([1.0, 0.0, 0.0])  # Unit vector to landmark
bearing_factor = BearingObservationFactor(200, pose_var, landmark, bearing_obs, 0.01)
graph.add_factor(bearing_factor)

# Finalize and optimize
graph.finalize_structure()

optimizer = SparseOptimizer()
settings = OptimizerSettings()
settings.method = OptimizerMethod.LevenbergMarquardt
settings.max_num_iterations = 50
settings.verbose = True

optimizer.setup(graph, settings)
optimizer.optimize()

print(f"Final pose: {pose_var.value()}")
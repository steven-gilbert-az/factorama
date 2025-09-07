#!/usr/bin/env python3
"""
Simple localization example using Factorama Python bindings

This example demonstrates basic robot localization using bearing observations
to landmarks. The robot has prior information about its initial pose and
observes several landmarks with known positions.
"""

import numpy as np
import factorama


def create_simple_localization_problem():
    """Create a simple localization problem with one pose and multiple landmarks"""
    
    # Create factor graph
    graph = factorama.FactorGraph()
    
    # Create robot pose variable (SE(3): [tx, ty, tz, rx, ry, rz])
    # Initial guess: robot at origin with no rotation
    initial_pose = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.1])  # Small noise from true pose
    robot_pose = factorama.PoseVariable(1, initial_pose)
    robot_pose.set_is_constant(True)
    graph.add_variable(robot_pose)
    
    # Create landmark variables
    landmark_positions = [
        np.array([10.0, 0.0, 0.0]),   # Landmark 1
        np.array([0.0, 10.0, 0.0]),   # Landmark 2 
        np.array([-10.0, 0.0, 0.0]),  # Landmark 3
        np.array([0.0, -10.0, 0.0])   # Landmark 4
    ]
    
    landmarks = []
    for i, pos in enumerate(landmark_positions):
        landmark = factorama.LandmarkVariable(100 + i, pos)
        landmarks.append(landmark)
        graph.add_variable(landmark)

        landmark_prior_factor = factorama.GenericPriorFactor(150 + i, landmark, pos, 0.1)
        graph.add_factor(landmark_prior_factor)
    
    # Add prior on robot pose (we have some initial estimate)
    prior_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # True robot pose
    pose_prior = factorama.GenericPriorFactor(200, robot_pose, prior_mean, 0.1)  # sigma=0.1
    graph.add_factor(pose_prior)
    
    # Add bearing observation factors
    # Simulate 3D bearing observations from robot at origin (unit vectors in camera frame)
    bearing_observations = [
        np.array([1.0, 0.0, 0.0]),   # Bearing to landmark 1 (along +X axis)
        np.array([0.0, 1.0, 0.0]),   # Bearing to landmark 2 (along +Y axis) 
        np.array([-1.0, 0.0, 0.0]),  # Bearing to landmark 3 (along -X axis)
        np.array([0.0, -1.0, 0.0])   # Bearing to landmark 4 (along -Y axis)
    ]
    
    angle_sigma = 0.01  # High confidence in observations (1 degree)
    
    for i, (landmark, bearing) in enumerate(zip(landmarks, bearing_observations)):
        bearing_factor = factorama.BearingObservationFactor(
            300 + i, robot_pose, landmark, bearing, angle_sigma
        )
        graph.add_factor(bearing_factor)
    
    return graph


def optimize_graph(graph):
    """Optimize the factor graph"""
    
    print("Initial state:")
    print(f"  Number of variables: {graph.num_variables()}")
    print(f"  Number of factors: {len(graph.get_all_factors())}")
    
    # Finalize structure
    graph.finalize_structure()
    print(f"  Number of values: {graph.num_values()}")
    print(f"  Number of residuals: {graph.num_residuals()}")
    
    # Print initial robot pose
    robot_pose = graph.get_variable(1)
    print(f"  Initial robot pose: {robot_pose.value()}")
    
    # Create optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.GaussNewton
    settings.max_num_iterations = 20
    settings.step_tolerance = 1e-8
    settings.verbose = True
    
    # Setup and optimize
    optimizer.setup(graph, settings)
    
    print("\nOptimizing...")
    optimizer.optimize()
    
    # Print results
    print("\nOptimization complete!")
    print(f"  Final robot pose: {robot_pose.value()}")
    print(f"  Final iterations: {optimizer.current_stats.current_iteration}")
    print(f"  Final chi2: {optimizer.current_stats.chi2}")
    
    return optimizer


def main():
    """Main example function"""
    print("=== Factorama Python Bindings Example ===")
    print("Simple robot localization using bearing observations\n")
    
    # Create problem
    graph = create_simple_localization_problem()
    
    # Optimize
    optimizer = optimize_graph(graph)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
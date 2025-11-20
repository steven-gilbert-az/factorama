#!/usr/bin/env python3
"""
Test the enhanced PlotFactorGraph function with 2D factor graphs
"""

import numpy as np
import factorama
import matplotlib.pyplot as plt


def create_2d_slam_example():
    """Create a simple 2D SLAM example with poses and landmarks"""
    # Ground truth: 3 poses forming a path
    gt_poses = [
        np.array([0.0, 0.0, 0.0]),          # x, y, theta (origin, facing east)
        np.array([2.0, 0.0, np.pi / 4]),    # moved east, turned 45°
        np.array([3.0, 1.0, np.pi / 2])     # moved northeast, facing north
    ]

    # 4 landmarks forming a square
    gt_landmarks = [
        np.array([1.0, 1.5]),   # Landmark 0
        np.array([2.5, 1.5]),   # Landmark 1
        np.array([2.5, 2.5]),   # Landmark 2
        np.array([1.0, 2.5])    # Landmark 3
    ]

    # Create factor graph
    graph = factorama.FactorGraph()
    var_id = 0
    factor_id = 0

    # Create pose variables with noisy initial guesses
    poses = []
    for i, gt_pose in enumerate(gt_poses):
        noisy_pose = gt_pose.copy()
        noisy_pose[0] += 0.3 * np.random.randn()  # x noise
        noisy_pose[1] += 0.3 * np.random.randn()  # y noise
        noisy_pose[2] += 0.15 * np.random.randn() # theta noise

        pose = factorama.Pose2DVariable(var_id, noisy_pose)
        var_id += 1
        poses.append(pose)
        graph.add_variable(pose)

    # Create landmark variables with noisy initial guesses
    landmarks = []
    for gt_landmark in gt_landmarks:
        noisy_landmark = gt_landmark.copy()
        noisy_landmark[0] += 0.2 * np.random.randn()
        noisy_landmark[1] += 0.2 * np.random.randn()

        landmark = factorama.GenericVariable(var_id, noisy_landmark)
        var_id += 1
        landmarks.append(landmark)
        graph.add_variable(landmark)

    # Add strong prior on first pose (anchor)
    pose_prior = factorama.Pose2DPriorFactor(
        factor_id, poses[0], gt_poses[0], 0.01, 0.01)
    factor_id += 1
    graph.add_factor(pose_prior)

    # Add weaker priors on other poses
    for i in range(1, len(poses)):
        pose_prior = factorama.Pose2DPriorFactor(
            factor_id, poses[i], gt_poses[i], 0.5, 0.2)
        factor_id += 1
        graph.add_factor(pose_prior)

    # Add landmark priors (weak)
    for landmark, gt_landmark in zip(landmarks, gt_landmarks):
        landmark_prior = factorama.GenericPriorFactor(
            factor_id, landmark, gt_landmark, 1.0)
        factor_id += 1
        graph.add_factor(landmark_prior)

    # Helper function to compute bearing and range
    def compute_observation(pose_gt, landmark_gt):
        pose_pos = pose_gt[:2]
        pose_theta = pose_gt[2]
        delta_world = landmark_gt - pose_pos
        c = np.cos(pose_theta)
        s = np.sin(pose_theta)
        R_T = np.array([[c, s], [-s, c]])
        delta_local = R_T @ delta_world
        range_obs = np.linalg.norm(delta_local)
        bearing_obs = np.arctan2(delta_local[1], delta_local[0])
        return range_obs, bearing_obs

    # Add observations from each pose to nearby landmarks
    # Mix of bearing-only and range-bearing measurements
    bearing_sigma = 0.05
    range_sigma = 0.1

    # Pose 0 observes landmarks 0, 1
    # Landmark 0: bearing-only
    _, bearing = compute_observation(gt_poses[0], gt_landmarks[0])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[0], landmarks[0], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Landmark 1: range-bearing
    range_obs, bearing = compute_observation(gt_poses[0], gt_landmarks[1])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[0], landmarks[1], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 1 observes landmarks 0, 1, 2
    # Landmark 0: range-bearing
    range_obs, bearing = compute_observation(gt_poses[1], gt_landmarks[0])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[1], landmarks[0], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Landmark 1: bearing-only
    _, bearing = compute_observation(gt_poses[1], gt_landmarks[1])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[1], landmarks[1], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Landmark 2: range-bearing
    range_obs, bearing = compute_observation(gt_poses[1], gt_landmarks[2])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[1], landmarks[2], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 2 observes landmarks 1, 2, 3
    # Landmark 1: range-bearing
    range_obs, bearing = compute_observation(gt_poses[2], gt_landmarks[1])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[2], landmarks[1], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Landmark 2: bearing-only
    _, bearing = compute_observation(gt_poses[2], gt_landmarks[2])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[2], landmarks[2], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Landmark 3: bearing-only
    _, bearing = compute_observation(gt_poses[2], gt_landmarks[3])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[2], landmarks[3], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Add between factors connecting consecutive poses
    for i in range(len(poses) - 1):
        dx = gt_poses[i+1][0] - gt_poses[i][0]
        dy = gt_poses[i+1][1] - gt_poses[i][1]
        dtheta = gt_poses[i+1][2] - gt_poses[i][2]
        relative_pose = np.array([dx, dy, dtheta])

        measured_between = factorama.GenericVariable(var_id, relative_pose)
        var_id += 1
        measured_between.set_constant(True)
        graph.add_variable(measured_between)

        between_factor = factorama.Pose2DBetweenFactor(
            factor_id, poses[i], poses[i+1], measured_between, 0.1, 0.05)
        factor_id += 1
        graph.add_factor(between_factor)

    return graph


def test_2d_plotting_without_covariance():
    """Test plotting 2D factor graph without covariance"""
    print("\nTest 1: Plotting 2D factor graph WITHOUT covariance")

    # Create factor graph
    graph = create_2d_slam_example()
    graph.finalize_structure()

    print(f"  Graph has {graph.num_variables()} variables")
    print(f"  Graph has {graph.num_residuals()} residuals")

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = False
    optimizer.setup(graph, settings)

    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)
    print(f"  Initial residual norm: {initial_norm:.6f}")

    optimizer.optimize()

    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)
    print(f"  Final residual norm: {final_norm:.6f}")

    # Plot without covariance
    fig, ax = plt.subplots(figsize=(10, 8))
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=False, plot_3d=False)
    plt.title('2D SLAM - Without Covariance')
    plt.savefig('/tmp/test_2d_plot_no_cov.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_2d_plot_no_cov.png")
    plt.close()

    print("  ✓ Test passed!")


def test_2d_plotting_with_covariance():
    """Test plotting 2D factor graph with covariance ellipses"""
    print("\nTest 2: Plotting 2D factor graph WITH covariance")

    # Create factor graph
    graph = create_2d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = False
    optimizer.setup(graph, settings)

    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)
    print(f"  Initial residual norm: {initial_norm:.6f}")

    optimizer.optimize()

    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)
    print(f"  Final residual norm: {final_norm:.6f}")

    # Plot with covariance
    fig, ax = plt.subplots(figsize=(10, 8))
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=True,
                             optimizer=optimizer, plot_3d=False)
    plt.title('2D SLAM - With 1-Sigma Covariance Ellipses')
    plt.savefig('/tmp/test_2d_plot_with_cov.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_2d_plot_with_cov.png")
    plt.close()

    print("  ✓ Test passed!")


def test_2d_plotting_without_measurements():
    """Test plotting 2D factor graph without measurement lines"""
    print("\nTest 3: Plotting 2D factor graph WITHOUT measurement lines")

    # Create factor graph
    graph = create_2d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = False
    optimizer.setup(graph, settings)
    optimizer.optimize()

    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Plot without measurement lines (cleaner view)
    fig, ax = plt.subplots(figsize=(10, 8))
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=True,
                             optimizer=optimizer, plot_3d=False, plot_measurements=False)
    plt.title('2D SLAM - Clean View (No Measurement Lines)')
    plt.savefig('/tmp/test_2d_plot_no_measurements.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_2d_plot_no_measurements.png")
    plt.close()

    print("  ✓ Test passed!")


def test_2d_plotting_3d_view():
    """Test plotting 2D factor graph in 3D view (at z=0 plane)"""
    print("\nTest 4: Plotting 2D factor graph in 3D view")

    # Create factor graph
    graph = create_2d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = False
    optimizer.setup(graph, settings)
    optimizer.optimize()

    # Plot in 3D view
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=False, plot_3d=True)
    plt.title('2D SLAM - 3D View (z=0 plane)')
    plt.savefig('/tmp/test_2d_plot_3d_view.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_2d_plot_3d_view.png")
    plt.close()

    print("  ✓ Test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced PlotFactorGraph with 2D Factor Graphs")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    test_2d_plotting_without_covariance()
    test_2d_plotting_with_covariance()
    test_2d_plotting_without_measurements()
    test_2d_plotting_3d_view()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - /tmp/test_2d_plot_no_cov.png (with measurement lines)")
    print("  - /tmp/test_2d_plot_with_cov.png (with measurement lines)")
    print("  - /tmp/test_2d_plot_no_measurements.png (clean view)")
    print("  - /tmp/test_2d_plot_3d_view.png (3D view with measurements)")
    print("\nMeasurement types:")
    print("  - Orange dashed lines: Bearing-only measurements")
    print("  - Purple solid lines: Range-bearing measurements")

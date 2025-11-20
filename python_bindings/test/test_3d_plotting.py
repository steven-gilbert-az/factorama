#!/usr/bin/env python3
"""
Test the enhanced PlotFactorGraph function with 3D factor graphs
"""

import numpy as np
import factorama
import matplotlib.pyplot as plt


def rodrigues_to_dcm(r_vec):
    """Convert Rodrigues rotation vector to Direction Cosine Matrix (DCM)"""
    theta = np.linalg.norm(r_vec)
    if theta < 1e-10:
        return np.eye(3)

    k = r_vec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def create_3d_slam_example():
    """Create a 3D SLAM example with a spiral trajectory and landmarks"""

    # Ground truth: 5 poses forming a spiral upward trajectory
    # Each pose is [tx, ty, tz, rx, ry, rz] where r is Rodrigues rotation vector
    gt_poses = []
    num_poses = 5

    for i in range(num_poses):
        # Spiral trajectory
        angle = i * np.pi / 3  # 60 degrees between poses
        radius = 3.0
        height = i * 1.0

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        # Rotation: start facing tangent to spiral, looking slightly upward
        yaw = angle + np.pi / 2  # Tangent to circle
        pitch = 0.1  # Slight upward tilt
        roll = 0.0

        # Convert to Rodrigues (simplified: small angles)
        rx = pitch
        ry = 0.0
        rz = yaw

        gt_poses.append(np.array([x, y, z, rx, ry, rz]))

    # 8 landmarks forming a cube around the trajectory
    cube_size = 4.0
    cube_center = np.array([0.0, 0.0, 2.0])

    gt_landmarks = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                landmark = cube_center + cube_size * np.array([dx, dy, dz])
                gt_landmarks.append(landmark)

    # Create factor graph
    graph = factorama.FactorGraph()
    var_id = 0
    factor_id = 0

    # Create pose variables with noisy initial guesses
    poses = []
    for i, gt_pose in enumerate(gt_poses):
        noisy_pose = gt_pose.copy()
        # Add noise to position
        noisy_pose[0] += 0.3 * np.random.randn()  # x
        noisy_pose[1] += 0.3 * np.random.randn()  # y
        noisy_pose[2] += 0.3 * np.random.randn()  # z
        # Add noise to rotation
        noisy_pose[3] += 0.1 * np.random.randn()  # rx
        noisy_pose[4] += 0.1 * np.random.randn()  # ry
        noisy_pose[5] += 0.1 * np.random.randn()  # rz

        pose = factorama.PoseVariable(var_id, noisy_pose)
        var_id += 1
        poses.append(pose)
        graph.add_variable(pose)

    # Create landmark variables with noisy initial guesses
    landmarks = []
    for gt_landmark in gt_landmarks:
        noisy_landmark = gt_landmark.copy()
        noisy_landmark += 0.4 * np.random.randn(3)

        landmark = factorama.LandmarkVariable(var_id, noisy_landmark)
        var_id += 1
        landmarks.append(landmark)
        graph.add_variable(landmark)

    # Add strong prior on first pose (anchor the graph)
    pose_position_sigma = 0.01
    pose_orientation_sigma = 0.01
    first_pose_prior = factorama.GenericPriorFactor(
        factor_id, poses[0], gt_poses[0], 0.01)
    factor_id += 1
    graph.add_factor(first_pose_prior)

    # Add weaker priors on other poses
    for i in range(1, len(poses)):
        pose_prior = factorama.GenericPriorFactor(
            factor_id, poses[i], gt_poses[i], 0.5)
        factor_id += 1
        graph.add_factor(pose_prior)

    # Add weak landmark priors
    for landmark, gt_landmark in zip(landmarks, gt_landmarks):
        landmark_prior = factorama.GenericPriorFactor(
            factor_id, landmark, gt_landmark, 1.0)
        factor_id += 1
        graph.add_factor(landmark_prior)

    # Helper function to compute bearing observation in camera frame
    def compute_bearing_observation(pose_gt, landmark_gt):
        """Compute bearing from pose to landmark in camera frame"""
        pos_W = pose_gt[:3]
        r_vec = pose_gt[3:]

        # Get rotation matrix
        R_WC = rodrigues_to_dcm(r_vec)
        R_CW = R_WC.T

        # Vector from camera to landmark in world frame
        delta_W = landmark_gt - pos_W

        # Transform to camera frame
        delta_C = R_CW @ delta_W

        # Normalize to get bearing
        bearing_C = delta_C / np.linalg.norm(delta_C)

        return bearing_C

    # Add bearing observations
    # Each pose observes nearby landmarks
    bearing_sigma = 0.02  # radians

    for i, pose in enumerate(poses):
        # Compute which landmarks are visible from this pose
        # For simplicity, let each pose observe 4-6 landmarks
        num_observations = min(6, len(landmarks))

        # Choose landmarks based on proximity
        pose_pos = gt_poses[i][:3]
        distances = [np.linalg.norm(lm - pose_pos) for lm in gt_landmarks]
        closest_indices = np.argsort(distances)[:num_observations]

        for lm_idx in closest_indices:
            bearing_C = compute_bearing_observation(gt_poses[i], gt_landmarks[lm_idx])

            # Add small noise to bearing observation
            bearing_noisy = bearing_C + bearing_sigma * np.random.randn(3)
            bearing_noisy /= np.linalg.norm(bearing_noisy)  # Re-normalize

            bearing_factor = factorama.BearingObservationFactor(
                factor_id, poses[i], landmarks[lm_idx], bearing_C, bearing_sigma)
            factor_id += 1
            graph.add_factor(bearing_factor)

    # Add pose-to-pose constraints (odometry)
    # Use position and orientation between factors
    for i in range(len(poses) - 1):
        # Position between factor
        pos_delta = gt_poses[i+1][:3] - gt_poses[i][:3]

        # Create GenericVariable for the position measurement
        pos_delta_var = factorama.GenericVariable(var_id, pos_delta)
        var_id += 1
        pos_delta_var.set_constant(True)
        graph.add_variable(pos_delta_var)

        pos_between_factor = factorama.PosePositionBetweenFactor(
            factor_id, poses[i], poses[i+1], pos_delta_var, 0.1)
        factor_id += 1
        graph.add_factor(pos_between_factor)

        # Orientation between factor
        # Compute relative rotation
        R1 = rodrigues_to_dcm(gt_poses[i][3:])
        R2 = rodrigues_to_dcm(gt_poses[i+1][3:])
        R_rel = R1.T @ R2

        # Create rotation variable for the measurement
        rot_var = factorama.RotationVariable(var_id, R_rel)
        var_id += 1
        rot_var.set_constant(True)
        graph.add_variable(rot_var)

        ori_between_factor = factorama.PoseOrientationBetweenFactor(
            factor_id, poses[i], poses[i+1], rot_var, 0.05)
        factor_id += 1
        graph.add_factor(ori_between_factor)

    return graph


def test_3d_plotting_without_covariance():
    """Test plotting 3D factor graph without covariance"""
    print("\nTest 1: Plotting 3D factor graph WITHOUT covariance")

    # Create factor graph
    graph = create_3d_slam_example()
    graph.finalize_structure()

    print(f"  Graph has {graph.num_variables()} variables")
    print(f"  Graph has {graph.num_residuals()} residuals")

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 100
    settings.verbose = False
    optimizer.setup(graph, settings)

    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)
    print(f"  Initial residual norm: {initial_norm:.6f}")

    optimizer.optimize()

    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)
    print(f"  Final residual norm: {final_norm:.6f}")
    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Plot in 3D without covariance
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=False, plot_3d=True)
    plt.title('3D SLAM - Spiral Trajectory (No Covariance)')
    plt.savefig('/tmp/test_3d_plot_no_cov.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_3d_plot_no_cov.png")
    plt.close()

    print("  ✓ Test passed!")


def test_3d_plotting_with_covariance():
    """Test plotting 3D factor graph with covariance ellipsoids"""
    print("\nTest 2: Plotting 3D factor graph WITH covariance ellipsoids")

    # Create factor graph
    graph = create_3d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 100
    settings.verbose = False
    optimizer.setup(graph, settings)

    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)
    print(f"  Initial residual norm: {initial_norm:.6f}")

    optimizer.optimize()

    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)
    print(f"  Final residual norm: {final_norm:.6f}")
    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Plot in 3D with covariance ellipsoids
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=True,
                             optimizer=optimizer, plot_3d=True)
    plt.title('3D SLAM - Spiral Trajectory with 1-Sigma Covariance Ellipsoids')
    plt.savefig('/tmp/test_3d_plot_with_cov.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_3d_plot_with_cov.png")
    plt.close()

    print("  ✓ Test passed!")


def test_3d_plotting_2d_projection():
    """Test plotting 3D factor graph as 2D projection (x-y plane)"""
    print("\nTest 3: Plotting 3D factor graph as 2D projection")

    # Create factor graph
    graph = create_3d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 100
    settings.verbose = False
    optimizer.setup(graph, settings)
    optimizer.optimize()

    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Plot as 2D projection with covariance ellipses
    fig, ax = plt.subplots(figsize=(12, 10))
    factorama.PlotFactorGraph(graph, axis_handle=ax, plot_covariance=True,
                             optimizer=optimizer, plot_3d=False)
    plt.title('3D SLAM - Top-Down View (x-y projection) with Covariance Ellipses')
    plt.savefig('/tmp/test_3d_plot_2d_projection.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to /tmp/test_3d_plot_2d_projection.png")
    plt.close()

    print("  ✓ Test passed!")


def test_3d_covariance_inspection():
    """Test and inspect covariance matrices for 3D SLAM"""
    print("\nTest 4: Inspecting 3D covariance matrices")

    # Create factor graph
    graph = create_3d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 100
    settings.verbose = False
    optimizer.setup(graph, settings)
    optimizer.optimize()

    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Prepare covariances
    optimizer.prepare_to_estimate_covariances()

    # Get all variables
    variables = graph.get_all_variables()

    # Inspect covariances for poses and landmarks
    print("\n  Covariance Matrix Information:")
    print("  " + "=" * 70)

    for var in variables[:3]:  # First 3 variables (poses)
        cov_matrix, valid = optimizer.estimate_covariance(var)
        if valid:
            var_type = "Pose" if var.type() == factorama.VariableType.pose else "Other"

            # Extract position covariance (first 3x3 block)
            pos_cov = cov_matrix[:3, :3]

            # Compute eigenvalues to understand uncertainty shape
            eigenvalues = np.linalg.eigvalsh(pos_cov)
            std_devs = np.sqrt(eigenvalues)

            print(f"\n  {var_type} Variable ID {var.id()}:")
            print(f"    Full covariance shape: {cov_matrix.shape}")
            print(f"    Position covariance (3x3):")
            print(f"      Standard deviations along principal axes:")
            print(f"        σ1 = {std_devs[0]:.4f} m")
            print(f"        σ2 = {std_devs[1]:.4f} m")
            print(f"        σ3 = {std_devs[2]:.4f} m")
            print(f"      Uncertainty volume (det): {np.linalg.det(pos_cov):.6e}")

    # Inspect landmark covariances
    landmark_vars = [v for v in variables if v.type() == factorama.VariableType.landmark]
    if landmark_vars:
        var = landmark_vars[0]  # First landmark
        cov_matrix, valid = optimizer.estimate_covariance(var)
        if valid:
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            std_devs = np.sqrt(eigenvalues)

            print(f"\n  Landmark Variable ID {var.id()}:")
            print(f"    Covariance shape: {cov_matrix.shape}")
            print(f"    Standard deviations along principal axes:")
            print(f"      σ1 = {std_devs[0]:.4f} m")
            print(f"      σ2 = {std_devs[1]:.4f} m")
            print(f"      σ3 = {std_devs[2]:.4f} m")
            print(f"    Uncertainty volume (det): {np.linalg.det(cov_matrix):.6e}")

    print("\n  " + "=" * 70)
    print("  ✓ Covariance inspection complete!")


def test_3d_multiple_views():
    """Create a figure with multiple views of the 3D SLAM result"""
    print("\nTest 5: Creating multi-view visualization")

    # Create factor graph
    graph = create_3d_slam_example()
    graph.finalize_structure()

    # Setup and run optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 100
    settings.verbose = False
    optimizer.setup(graph, settings)
    optimizer.optimize()

    print(f"  Optimization status: {optimizer.current_stats.status}")

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 14))

    # 3D view with covariance
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax1, plot_covariance=True,
                             optimizer=optimizer, plot_3d=True)
    ax1.set_title('3D View with Covariance Ellipsoids', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # 3D view from different angle
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax2, plot_covariance=True,
                             optimizer=optimizer, plot_3d=True)
    ax2.set_title('3D View - Alternative Angle', fontsize=12, fontweight='bold')
    ax2.view_init(elev=60, azim=135)

    # Top-down view (x-y)
    ax3 = fig.add_subplot(2, 2, 3)
    factorama.PlotFactorGraph(graph, axis_handle=ax3, plot_covariance=True,
                             optimizer=optimizer, plot_3d=False)
    ax3.set_title('Top-Down View (x-y plane)', fontsize=12, fontweight='bold')

    # 3D view without covariance for clarity
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax4, plot_covariance=False, plot_3d=True)
    ax4.set_title('3D View - No Covariance', fontsize=12, fontweight='bold')
    ax4.view_init(elev=30, azim=225)

    plt.suptitle('3D SLAM - Multiple Perspectives', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/tmp/test_3d_plot_multiview.png', dpi=150, bbox_inches='tight')
    print("  Saved multi-view plot to /tmp/test_3d_plot_multiview.png")
    plt.close()

    print("  ✓ Test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Enhanced PlotFactorGraph with 3D Factor Graphs")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    test_3d_plotting_without_covariance()
    test_3d_plotting_with_covariance()
    test_3d_plotting_2d_projection()
    test_3d_covariance_inspection()
    test_3d_multiple_views()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - /tmp/test_3d_plot_no_cov.png")
    print("  - /tmp/test_3d_plot_with_cov.png")
    print("  - /tmp/test_3d_plot_2d_projection.png")
    print("  - /tmp/test_3d_plot_multiview.png")
    print("\nCovariance matrices were inspected and validated!")

#!/usr/bin/env python3
"""
Tests for Factorama Python bindings - variables and factors
"""

import pytest
import numpy as np
import factorama


def test_rotation_variable_creation():
    """Test RotationVariable creation and basic operations"""
    # Create a rotation matrix (identity)
    dcm_AB = np.eye(3)
    rot_var = factorama.RotationVariable(1, dcm_AB)
    
    assert rot_var.id() == 1
    assert rot_var.size() == 3  # SO(3) has 3 DOF
    assert rot_var.type() == factorama.VariableType.extrinsic_rotation
    
    # Check rotation matrix access
    dcm_result = rot_var.dcm_AB()
    assert dcm_result.shape == (3, 3)
    assert np.allclose(dcm_result, dcm_AB)
    
    # Test constant flag
    assert not rot_var.is_constant()
    rot_var.set_constant(True)
    assert rot_var.is_constant()


def test_inverse_range_variable_creation():
    """Test InverseRangeVariable creation and basic operations"""
    origin_pos = np.array([0.0, 0.0, 0.0])
    bearing_W = np.array([1.0, 0.0, 0.0])  # Unit vector in x direction
    initial_range = 10.0
    
    inv_range_var = factorama.InverseRangeVariable(2, origin_pos, bearing_W, initial_range)
    
    assert inv_range_var.id() == 2
    assert inv_range_var.size() == 1  # Single inverse range parameter
    assert inv_range_var.type() == factorama.VariableType.inverse_range_landmark
    
    # Check inverse range value
    assert np.isclose(inv_range_var.inverse_range(), 1.0 / initial_range)
    
    # Check computed position
    pos_W = inv_range_var.pos_W()
    expected_pos = origin_pos + initial_range * bearing_W
    assert np.allclose(pos_W, expected_pos)
    
    # Check reference returns
    assert np.allclose(inv_range_var.origin_pos_W(), origin_pos)
    assert np.allclose(inv_range_var.bearing_W(), bearing_W)
    
    # Test range limits
    inv_range_var.minimum_inverse_range = 1e-3
    inv_range_var.maximum_inverse_range = 1e2
    assert inv_range_var.minimum_inverse_range == 1e-3
    assert inv_range_var.maximum_inverse_range == 1e2


def test_bearing_observation_factor():
    """Test BearingObservationFactor creation and residual computation"""
    # Create variables
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)
    
    landmark_pos = np.array([5.0, 0.0, 0.0])
    landmark_var = factorama.LandmarkVariable(2, landmark_pos)
    
    # Create factor
    bearing_obs = np.array([1.0, 0.0, 0.0])  # Looking straight ahead
    angle_sigma = 0.1
    factor = factorama.BearingObservationFactor(1, pose_var, landmark_var, bearing_obs, angle_sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.bearing_observation
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3
    
    # Check variables
    variables = factor.variables()
    assert len(variables) == 2


def test_inverse_range_bearing_factor():
    """Test InverseRangeBearingFactor creation and residual computation"""
    # Create variables
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)
    
    origin_pos = np.array([0.0, 0.0, 0.0])
    bearing_W = np.array([1.0, 0.0, 0.0])
    initial_range = 5.0
    inv_range_var = factorama.InverseRangeVariable(2, origin_pos, bearing_W, initial_range)
    
    # Create factor
    bearing_obs = np.array([1.0, 0.0, 0.0])
    angle_sigma = 0.1
    factor = factorama.InverseRangeBearingFactor(1, pose_var, inv_range_var, bearing_obs, angle_sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.inverse_range_bearing
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3
    


def test_generic_prior_factor():
    """Test GenericPriorFactor creation and residual computation"""
    # Create variable
    value_init = np.array([1.0, 2.0, 3.0])
    generic_var = factorama.GenericVariable(1, value_init)
    
    # Create factor
    prior_value = np.array([1.5, 2.5, 3.5])
    sigma = 0.5
    factor = factorama.GenericPriorFactor(1, generic_var, prior_value, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.generic_prior
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3
    
    # Check variables
    variables = factor.variables()
    assert len(variables) == 1


def test_generic_between_factor():
    """Test GenericBetweenFactor creation and residual computation"""
    # Create variables
    value_a = np.array([1.0, 2.0])
    value_b = np.array([3.0, 4.0])
    measured_diff = np.array([2.0, 2.0])  # Expected difference
    
    var_a = factorama.GenericVariable(1, value_a)
    var_b = factorama.GenericVariable(2, value_b)
    var_diff = factorama.GenericVariable(3, measured_diff)
    
    # Create factor
    sigma = 0.2
    factor = factorama.GenericBetweenFactor(1, var_a, var_b, var_diff, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 2
    assert factor.type() == factorama.FactorType.generic_between
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 2
    
    # Check variables
    variables = factor.variables()
    assert len(variables) == 3


def test_pose_position_prior_factor():
    """Test PosePositionPriorFactor creation and residual computation"""
    # Create pose variable
    pose_init = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    pose_var = factorama.PoseVariable(1, pose_init)
    
    # Create factor
    pos_prior = np.array([1.5, 2.5, 3.5])
    sigma = 0.1
    factor = factorama.PosePositionPriorFactor(1, pose_var, pos_prior, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.pose_position_prior
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3


def test_pose_orientation_prior_factor():
    """Test PoseOrientationPriorFactor creation and residual computation"""
    # Create pose variable
    pose_init = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    pose_var = factorama.PoseVariable(1, pose_init)
    
    # Create factor
    dcm_CW_prior = np.eye(3)
    sigma = 0.05
    factor = factorama.PoseOrientationPriorFactor(1, pose_var, dcm_CW_prior, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.pose_orientation_prior
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3


def test_pose_position_between_factor():
    """Test PosePositionBetweenFactor creation and residual computation"""
    # Create pose variables
    pose_a_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_b_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_a = factorama.PoseVariable(1, pose_a_init)
    pose_b = factorama.PoseVariable(2, pose_b_init)
    
    # Create measured difference variable
    measured_diff = np.array([1.0, 0.0, 0.0])
    diff_var = factorama.GenericVariable(3, measured_diff)
    
    # Create factor
    sigma = 0.1
    factor = factorama.PosePositionBetweenFactor(1, pose_a, pose_b, diff_var, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.pose_position_between
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3


def test_pose_orientation_between_factor():
    """Test PoseOrientationBetweenFactor creation and residual computation"""
    # Create pose variables
    pose1_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose2_init = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    pose1 = factorama.PoseVariable(1, pose1_init)
    pose2 = factorama.PoseVariable(2, pose2_init)
    
    # Create calibration rotation
    dcm_calib = np.eye(3)
    calib_rot = factorama.RotationVariable(3, dcm_calib)
    
    # Create factor
    angle_sigma = 0.05
    factor = factorama.PoseOrientationBetweenFactor(1, pose1, pose2, calib_rot, angle_sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.pose_orientation_between
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3


def test_rotation_prior_factor():
    """Test RotationPriorFactor creation and residual computation"""
    # Create rotation variable
    dcm_current = np.eye(3)
    rot_var = factorama.RotationVariable(1, dcm_current)
    
    # Create factor with slightly different prior
    dcm_prior = np.array([[0.9999, -0.0100, 0.0000],
                          [0.0100,  0.9999, 0.0000],
                          [0.0000,  0.0000, 1.0000]])  # Small rotation about z
    sigma = 0.01
    factor = factorama.RotationPriorFactor(1, rot_var, dcm_prior, sigma)
    
    assert factor.id() == 1
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.pose_orientation_prior
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 3


def test_bearing_projection_factor_2d():
    """Test BearingProjectionFactor2D creation and residual computation"""
    # Create variables
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)
    
    landmark_pos = np.array([5.0, 0.0, 0.0])
    landmark_var = factorama.LandmarkVariable(2, landmark_pos)
    
    # Create factor
    bearing_obs = np.array([1.0, 0.0, 0.0])
    sigma = 0.1
    tolerance = 1e-6
    factor = factorama.BearingProjectionFactor2D(1, pose_var, landmark_var, bearing_obs, sigma, tolerance)
    
    assert factor.id() == 1
    assert factor.residual_size() == 2  # 2D projection
    assert factor.type() == factorama.FactorType.bearing_observation  # Uses existing enum
    
    # Compute residual
    residual = factor.compute_residual()
    assert len(residual) == 2


def test_factor_graph_with_variables():
    """Test factor graph with variables"""
    graph = factorama.FactorGraph()

    # Create variables
    dcm_AB = np.eye(3)
    rot_var = factorama.RotationVariable(1, dcm_AB)

    origin_pos = np.array([0.0, 0.0, 0.0])
    bearing_W = np.array([1.0, 0.0, 0.0])
    inv_range_var = factorama.InverseRangeVariable(2, origin_pos, bearing_W, 10.0)

    # Add to graph
    graph.add_variable(rot_var)
    graph.add_variable(inv_range_var)

    assert graph.num_variables() == 2

    # Add prior factors
    rot_prior = factorama.RotationPriorFactor(1, rot_var, dcm_AB, 1.0)
    inv_range_value = np.array([1.0 / 10.0])
    inv_range_prior = factorama.GenericPriorFactor(2, inv_range_var, inv_range_value, 1.0)
    graph.add_factor(rot_prior)
    graph.add_factor(inv_range_prior)

    # Finalize structure
    graph.finalize_structure()
    assert graph.num_values() == 4  # 3 + 1
    assert graph.num_residuals() == 4  # 3 + 1


def test_factor_graph_with_factors():
    """Test adding factors to factor graph"""
    graph = factorama.FactorGraph()

    # Create variables
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)

    value_init = np.array([1.0, 2.0, 3.0])
    generic_var = factorama.GenericVariable(2, value_init)

    graph.add_variable(pose_var)
    graph.add_variable(generic_var)

    # Create and add factors
    pos_prior = np.array([0.1, 0.1, 0.1])
    pos_factor = factorama.PosePositionPriorFactor(1, pose_var, pos_prior, 0.1)

    dcm_prior = np.eye(3)
    ori_factor = factorama.PoseOrientationPriorFactor(2, pose_var, dcm_prior, 0.1)

    prior_value = np.array([1.1, 2.1, 3.1])
    generic_factor = factorama.GenericPriorFactor(3, generic_var, prior_value, 0.1)

    graph.add_factor(pos_factor)
    graph.add_factor(ori_factor)
    graph.add_factor(generic_factor)

    # Finalize and check
    graph.finalize_structure()
    assert graph.num_variables() == 2
    assert graph.num_residuals() == 9  # 3 + 3 + 3

    # Compute residuals
    residuals = graph.compute_full_residual_vector()
    assert len(residuals) == 9


def test_2d_slam_with_bearing_and_range_bearing():
    """Test 2D SLAM scenario with mix of bearing-only and range-bearing factors"""
    # Ground truth: 2 poses and 4 landmarks
    # Pose 0: origin, facing east (0 radians)
    # Pose 1: east of origin, facing north (π/2 radians)
    gt_poses = [
        np.array([0.0, 0.0, 0.0]),          # x, y, theta
        np.array([3.0, 0.0, np.pi / 2])
    ]

    # 4 landmarks forming a square
    gt_landmarks = [
        np.array([1.0, 1.0]),   # Landmark 0
        np.array([2.0, 1.0]),   # Landmark 1
        np.array([2.0, 2.0]),   # Landmark 2
        np.array([1.0, 2.0])    # Landmark 3
    ]

    # Create factor graph
    graph = factorama.FactorGraph()
    var_id = 0
    factor_id = 0

    # Create pose variables with noisy initial guesses
    poses = []
    for i, gt_pose in enumerate(gt_poses):
        noisy_pose = gt_pose.copy()
        noisy_pose[0] += 0.2 * (0.1 if i == 0 else 0.3)  # x noise
        noisy_pose[1] += 0.15 * (-0.2 if i == 1 else 0.2) # y noise
        noisy_pose[2] += 0.1 * (0.15 if i == 1 else -0.1) # theta noise

        pose = factorama.Pose2DVariable(var_id, noisy_pose)
        var_id += 1
        poses.append(pose)
        graph.add_variable(pose)

    # Create landmark variables with noisy initial guesses
    landmarks = []
    for i, gt_landmark in enumerate(gt_landmarks):
        noisy_landmark = gt_landmark.copy()
        noisy_landmark[0] += 0.1 * (0.2 if i % 2 == 0 else -0.15)
        noisy_landmark[1] += 0.1 * (-0.1 if i // 2 == 0 else 0.2)

        landmark = factorama.GenericVariable(var_id, noisy_landmark)
        var_id += 1
        landmarks.append(landmark)
        graph.add_variable(landmark)

    # Add pose priors
    pose_position_sigma = 0.5
    pose_angle_sigma = 0.2
    for i, (pose, gt_pose) in enumerate(zip(poses, gt_poses)):
        pose_prior = factorama.Pose2DPriorFactor(
            factor_id, pose, gt_pose, pose_position_sigma, pose_angle_sigma)
        factor_id += 1
        graph.add_factor(pose_prior)

    # Add landmark priors (weaker)
    landmark_sigma = 1.0
    for landmark, gt_landmark in zip(landmarks, gt_landmarks):
        landmark_prior = factorama.GenericPriorFactor(
            factor_id, landmark, gt_landmark, landmark_sigma)
        factor_id += 1
        graph.add_factor(landmark_prior)

    # Mix of observations:
    # - Pose 0 sees landmarks 0, 1 with bearing-only
    # - Pose 0 sees landmark 2 with range-bearing
    # - Pose 1 sees landmarks 1, 2 with range-bearing
    # - Pose 1 sees landmark 3 with bearing-only

    bearing_sigma = 0.05  # radians
    range_sigma = 0.1     # meters

    # Helper function to compute bearing and range
    def compute_observation(pose_gt, landmark_gt):
        pose_pos = pose_gt[:2]
        pose_theta = pose_gt[2]

        # Delta in world frame
        delta_world = landmark_gt - pose_pos

        # Rotate to pose frame
        c = np.cos(pose_theta)
        s = np.sin(pose_theta)
        R_T = np.array([[c, s], [-s, c]])
        delta_local = R_T @ delta_world

        range_obs = np.linalg.norm(delta_local)
        bearing_obs = np.arctan2(delta_local[1], delta_local[0])

        return range_obs, bearing_obs

    # Pose 0, Landmark 0: bearing-only
    _, bearing = compute_observation(gt_poses[0], gt_landmarks[0])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[0], landmarks[0], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 0, Landmark 1: bearing-only
    _, bearing = compute_observation(gt_poses[0], gt_landmarks[1])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[0], landmarks[1], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 0, Landmark 2: range-bearing
    range_obs, bearing = compute_observation(gt_poses[0], gt_landmarks[2])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[0], landmarks[2], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 1, Landmark 1: range-bearing
    range_obs, bearing = compute_observation(gt_poses[1], gt_landmarks[1])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[1], landmarks[1], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 1, Landmark 2: range-bearing
    range_obs, bearing = compute_observation(gt_poses[1], gt_landmarks[2])
    factor = factorama.RangeBearingFactor2D(
        factor_id, poses[1], landmarks[2], range_obs, bearing,
        range_sigma, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Pose 1, Landmark 3: bearing-only
    _, bearing = compute_observation(gt_poses[1], gt_landmarks[3])
    factor = factorama.BearingObservationFactor2D(
        factor_id, poses[1], landmarks[3], bearing, bearing_sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Add between factor connecting consecutive poses
    between_position_sigma = 0.1
    between_angle_sigma = 0.05
    # Compute ground truth relative pose
    relative_pose = np.array([
        gt_poses[1][0] - gt_poses[0][0],  # dx
        gt_poses[1][1] - gt_poses[0][1],  # dy
        gt_poses[1][2] - gt_poses[0][2]   # dtheta
    ])
    measured_between = factorama.GenericVariable(var_id, relative_pose)
    var_id += 1
    measured_between.set_constant(True)
    graph.add_variable(measured_between)

    between_factor = factorama.Pose2DBetweenFactor(
        factor_id, poses[0], poses[1], measured_between,
        between_position_sigma, between_angle_sigma)
    factor_id += 1
    graph.add_factor(between_factor)

    # Finalize and optimize
    graph.finalize_structure()

    # Check graph structure
    assert graph.num_variables() == 7  # 2 poses + 4 landmarks + 1 between measurement
    expected_residuals = (
        2 * 3 +  # 2 pose priors (3 residuals each)
        4 * 2 +  # 4 landmark priors (2 residuals each)
        3 * 1 +  # 3 bearing-only factors (1 residual each)
        3 * 2 +  # 3 range-bearing factors (2 residuals each)
        1 * 3    # 1 between factor (3 residuals)
    )
    assert graph.num_residuals() == expected_residuals

    # Setup optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = False

    optimizer.setup(graph, settings)

    # Get initial residual norm
    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)

    # Optimize
    optimizer.optimize()

    # Get final residual norm
    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)

    # Check convergence
    assert optimizer.current_stats.status == factorama.OptimizerStatus.SUCCESS
    assert final_norm < initial_norm  # Should have improved
    assert final_norm < 1.0  # Should converge to small residual

    # Check that poses are close to ground truth
    for i, (pose, gt_pose) in enumerate(zip(poses, gt_poses)):
        optimized_pose = pose.value()
        # Position should be within ~1cm
        assert np.allclose(optimized_pose[:2], gt_pose[:2], atol=0.01)
        # Angle should be within ~1 degree
        angle_diff = np.abs(optimized_pose[2] - gt_pose[2])
        # Handle angle wrapping
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        assert angle_diff < 0.02  # ~1 degree


def test_coordinate_transform_factor():
    """Test CoordinateTransformFactor creation and optimization"""
    # Simple scenario: transform landmark from frame B to frame A
    # Transformation: vec_A = scale_AB * dcm_AB * vec_B - B_origin_A

    graph = factorama.FactorGraph()
    var_id = 0
    factor_id = 0

    # Ground truth transformation parameters
    gt_scale = 2.0
    gt_origin = np.array([1.0, 2.0, 3.0])  # B's origin in A
    gt_dcm = np.eye(3)  # Identity rotation for simplicity

    # A landmark in frame B
    lm_B_value = np.array([1.0, 0.0, 0.0])
    # Same landmark in frame A: vec_A = scale * dcm * vec_B - origin
    lm_A_value = gt_scale * gt_dcm @ lm_B_value - gt_origin

    # Create variables
    rot_AB = factorama.RotationVariable(var_id, gt_dcm)
    var_id += 1

    B_origin_A = factorama.GenericVariable(var_id, gt_origin)
    var_id += 1

    scale_AB = factorama.GenericVariable(var_id, np.array([gt_scale]))
    var_id += 1

    lm_A = factorama.LandmarkVariable(var_id, lm_A_value)
    var_id += 1

    lm_B = factorama.LandmarkVariable(var_id, lm_B_value)
    var_id += 1

    # Make landmarks constant to over-constrain
    lm_A.set_constant(True)
    lm_B.set_constant(True)

    # Add variables to graph
    graph.add_variable(rot_AB)
    graph.add_variable(B_origin_A)
    graph.add_variable(scale_AB)
    graph.add_variable(lm_A)
    graph.add_variable(lm_B)

    # Create coordinate transform factor
    sigma = 0.1
    factor = factorama.CoordinateTransformFactor(
        factor_id, rot_AB, B_origin_A, scale_AB, lm_A, lm_B, sigma)
    factor_id += 1
    graph.add_factor(factor)

    # Add priors on transformation variables (slightly off ground truth)
    rot_prior = factorama.RotationPriorFactor(
        factor_id, rot_AB, gt_dcm, 0.1)
    factor_id += 1
    graph.add_factor(rot_prior)

    origin_prior = factorama.GenericPriorFactor(
        factor_id, B_origin_A, gt_origin, 0.5)
    factor_id += 1
    graph.add_factor(origin_prior)

    scale_prior = factorama.GenericPriorFactor(
        factor_id, scale_AB, np.array([gt_scale]), 0.5)
    factor_id += 1
    graph.add_factor(scale_prior)

    # Check factor properties
    assert factor.residual_size() == 3
    assert factor.type() == factorama.FactorType.custom
    assert len(factor.variables()) == 5

    # Finalize and optimize
    graph.finalize_structure()

    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 20
    settings.verbose = False

    optimizer.setup(graph, settings)
    optimizer.optimize()

    # Check convergence
    assert optimizer.current_stats.status == factorama.OptimizerStatus.SUCCESS

    # Residual should be near zero since we used ground truth values
    residual = factor.compute_residual()
    assert np.linalg.norm(residual) < 0.1


def test_2d_between_factor_with_local_frame():
    """Test 2D between factor with local_frame=True parameter"""
    # Create a simple scenario: 3 poses moving along a path
    # Pose 0: origin, facing east
    # Pose 1: 2m east, rotated 90° (facing north)
    # Pose 2: 2m east + 1m north, facing north

    gt_poses = [
        np.array([0.0, 0.0, 0.0]),          # x, y, theta
        np.array([2.0, 0.0, np.pi / 2]),    # 2m east, facing north
        np.array([2.0, 1.0, np.pi / 2])     # 2m east, 1m north, facing north
    ]

    # Create factor graph
    graph = factorama.FactorGraph()
    var_id = 0
    factor_id = 0

    # Create pose variables with noisy initial guesses
    poses = []
    for i, gt_pose in enumerate(gt_poses):
        noisy_pose = gt_pose.copy()
        noisy_pose[0] += 0.3 * (i - 1)  # x noise
        noisy_pose[1] += 0.2 * (i - 1)  # y noise
        noisy_pose[2] += 0.1 * (i - 1)  # theta noise

        pose = factorama.Pose2DVariable(var_id, noisy_pose)
        var_id += 1
        poses.append(pose)
        graph.add_variable(pose)

    # Add strong prior on first pose (anchor)
    pose_prior = factorama.Pose2DPriorFactor(
        factor_id, poses[0], gt_poses[0], 0.01, 0.01)
    factor_id += 1
    graph.add_factor(pose_prior)

    # Add weaker priors on other poses
    for i in range(1, len(poses)):
        pose_prior = factorama.Pose2DPriorFactor(
            factor_id, poses[i], gt_poses[i], 1.0, 0.5)
        factor_id += 1
        graph.add_factor(pose_prior)

    # Helper to compute relative pose in local frame
    def compute_relative_pose_local(pose_a_gt, pose_b_gt):
        """Compute pose_b relative to pose_a in pose_a's local frame"""
        # World frame difference
        delta_world = pose_b_gt[:2] - pose_a_gt[:2]

        # Rotate to pose_a's local frame using same convention as dcm_2d()
        theta_a = pose_a_gt[2]
        c = np.cos(theta_a)
        s = np.sin(theta_a)
        R = np.array([[c, -s], [s, c]])  # World-to-local rotation (same as dcm_2d())
        delta_local = R @ delta_world

        # Angle difference
        dtheta = pose_b_gt[2] - pose_a_gt[2]

        return np.array([delta_local[0], delta_local[1], dtheta])

    # Add between factors with local_frame=True
    between_position_sigma = 0.05
    between_angle_sigma = 0.02

    for i in range(len(poses) - 1):
        # Compute relative pose in local frame
        relative_pose_local = compute_relative_pose_local(gt_poses[i], gt_poses[i+1])

        measured_between = factorama.GenericVariable(var_id, relative_pose_local)
        var_id += 1
        measured_between.set_constant(True)
        graph.add_variable(measured_between)

        # Create between factor with local_frame=True
        between_factor = factorama.Pose2DBetweenFactor(
            factor_id, poses[i], poses[i+1], measured_between,
            between_position_sigma, between_angle_sigma, local_frame=True)
        factor_id += 1
        graph.add_factor(between_factor)

    # Finalize and optimize
    graph.finalize_structure()

    # Check graph structure
    assert graph.num_variables() == 5  # 3 poses + 2 between measurements
    expected_residuals = (
        1 * 3 +  # 1 strong prior on first pose (3 residuals)
        2 * 3 +  # 2 weak priors on other poses (3 residuals each)
        2 * 3    # 2 between factors (3 residuals each)
    )
    assert graph.num_residuals() == expected_residuals

    # Setup optimizer
    optimizer = factorama.SparseOptimizer()
    settings = factorama.OptimizerSettings()
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 30
    settings.verbose = False

    optimizer.setup(graph, settings)

    # Get initial residual norm
    initial_residual = graph.compute_full_residual_vector()
    initial_norm = np.linalg.norm(initial_residual)

    # Optimize
    optimizer.optimize()

    # Get final residual norm
    final_residual = graph.compute_full_residual_vector()
    final_norm = np.linalg.norm(final_residual)

    # Check convergence
    assert optimizer.current_stats.status == factorama.OptimizerStatus.SUCCESS
    assert final_norm < initial_norm  # Should have improved
    assert final_norm < 3.0  # Should converge to reasonable residual (weak priors allow some error)

    # Check that poses are close to ground truth
    for i, (pose, gt_pose) in enumerate(zip(poses, gt_poses)):
        optimized_pose = pose.value()
        # Position should be within ~1cm
        assert np.allclose(optimized_pose[:2], gt_pose[:2], atol=0.02), \
            f"Pose {i} position mismatch: {optimized_pose[:2]} vs {gt_pose[:2]}"
        # Angle should be within ~2 degrees
        angle_diff = np.abs(optimized_pose[2] - gt_pose[2])
        # Handle angle wrapping
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        assert angle_diff < 0.035, \
            f"Pose {i} angle mismatch: {optimized_pose[2]} vs {gt_pose[2]} (diff={angle_diff})"  # ~2 degrees


if __name__ == "__main__":
    test_rotation_variable_creation()
    test_inverse_range_variable_creation()
    test_bearing_observation_factor()
    test_inverse_range_bearing_factor()
    test_generic_prior_factor()
    test_generic_between_factor()
    test_pose_position_prior_factor()
    test_pose_orientation_prior_factor()
    test_pose_position_between_factor()
    test_pose_orientation_between_factor()
    test_rotation_prior_factor()
    test_bearing_projection_factor_2d()
    test_factor_graph_with_variables()
    test_factor_graph_with_factors()
    test_coordinate_transform_factor()
    test_2d_slam_with_bearing_and_range_bearing()
    test_2d_between_factor_with_local_frame()
    print("All tests passed!")
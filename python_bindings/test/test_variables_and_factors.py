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
    print("All tests passed!")
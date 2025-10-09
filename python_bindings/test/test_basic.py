#!/usr/bin/env python3
"""
Basic tests for Factorama Python bindings
"""

import pytest
import numpy as np
import factorama


def test_imports():
    """Test that basic imports work"""
    assert hasattr(factorama, 'FactorGraph')
    assert hasattr(factorama, 'SparseOptimizer')
    assert hasattr(factorama, 'PoseVariable')
    assert hasattr(factorama, 'LandmarkVariable')


def test_factor_graph_creation():
    """Test factor graph creation and basic operations"""
    graph = factorama.FactorGraph()
    
    assert graph.num_variables() == 0
    assert graph.num_values() == 0
    assert graph.num_residuals() == 0


def test_pose_variable_creation():
    """Test pose variable creation and basic operations"""
    # Test SE(3) constructor
    pose_init = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # [tx, ty, tz, rx, ry, rz]
    pose_var = factorama.PoseVariable(1, pose_init)
    
    assert pose_var.id() == 1
    assert pose_var.size() == 6
    assert pose_var.type() == factorama.VariableType.pose
    assert np.allclose(pose_var.value(), pose_init)
    
    # Test position and rotation access
    position = pose_var.pos_W()
    assert len(position) == 3
    assert np.allclose(position, [1.0, 2.0, 3.0])


def test_landmark_variable_creation():
    """Test landmark variable creation and basic operations"""
    pos_init = np.array([10.0, 20.0, 30.0])
    landmark_var = factorama.LandmarkVariable(2, pos_init)
    
    assert landmark_var.id() == 2
    assert landmark_var.size() == 3
    assert landmark_var.type() == factorama.VariableType.landmark
    assert np.allclose(landmark_var.value(), pos_init)
    
    position = landmark_var.pos_W()
    assert np.allclose(position, pos_init)


def test_generic_variable_creation():
    """Test generic variable creation"""
    value_init = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    generic_var = factorama.GenericVariable(3, value_init)
    
    assert generic_var.id() == 3
    assert generic_var.size() == 5
    assert generic_var.type() == factorama.VariableType.generic
    assert np.allclose(generic_var.value(), value_init)


def test_factor_graph_with_variables():
    """Test adding variables to factor graph"""
    graph = factorama.FactorGraph()

    # Create variables
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)

    landmark_pos = np.array([5.0, 0.0, 0.0])
    landmark_var = factorama.LandmarkVariable(2, landmark_pos)

    # Add to graph
    graph.add_variable(pose_var)
    graph.add_variable(landmark_var)

    assert graph.num_variables() == 2

    # Add prior factors to make the graph well-conditioned (need at least 9 residuals for 9 values)
    pose_prior = factorama.GenericPriorFactor(1, pose_var, pose_init, 1.0)
    landmark_prior = factorama.GenericPriorFactor(2, landmark_var, landmark_pos, 1.0)
    graph.add_factor(pose_prior)
    graph.add_factor(landmark_prior)

    # Finalize structure
    graph.finalize_structure()
    assert graph.num_values() == 9  # 6 + 3
    assert graph.num_residuals() == 9  # 6 + 3


def test_optimizer_settings():
    """Test optimizer settings creation and modification"""
    settings = factorama.OptimizerSettings()
    
    # Test default values
    assert settings.method == factorama.OptimizerMethod.GaussNewton
    assert settings.max_num_iterations == 100
    assert settings.step_tolerance == 1e-6
    assert settings.verbose == False
    
    # Test modification
    settings.method = factorama.OptimizerMethod.LevenbergMarquardt
    settings.max_num_iterations = 50
    settings.verbose = True
    
    assert settings.method == factorama.OptimizerMethod.LevenbergMarquardt
    assert settings.max_num_iterations == 50
    assert settings.verbose == True


def test_optimizer_creation():
    """Test sparse optimizer creation"""
    optimizer = factorama.SparseOptimizer()

    # Create a simple factor graph
    graph = factorama.FactorGraph()
    pose_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_var = factorama.PoseVariable(1, pose_init)
    graph.add_variable(pose_var)

    # Add prior factor
    pose_prior = factorama.GenericPriorFactor(1, pose_var, pose_init, 1.0)
    graph.add_factor(pose_prior)

    graph.finalize_structure()

    # Setup optimizer
    settings = factorama.OptimizerSettings()
    optimizer.setup(graph, settings)

    # Verify settings
    retrieved_settings = optimizer.settings()
    assert retrieved_settings.max_num_iterations == 100


if __name__ == "__main__":
    test_imports()
    test_factor_graph_creation()
    test_pose_variable_creation()
    test_landmark_variable_creation()
    test_generic_variable_creation()
    test_factor_graph_with_variables()
    test_optimizer_settings()
    test_optimizer_creation()
    print("All tests passed!")
"""
Factorama Python bindings - factor graph optimization library
"""
from __future__ import annotations
import numpy
import scipy.sparse
import typing
__all__: list[str] = ['BearingObservationFactor', 'BearingProjectionFactor2D', 'Factor', 'FactorGraph', 'FactorType', 'GenericBetweenFactor', 'GenericPriorFactor', 'GenericVariable', 'InverseRangeBearingFactor', 'InverseRangeVariable', 'LandmarkVariable', 'OptimizerMethod', 'OptimizerSettings', 'OptimizerStats', 'PoseOrientationBetweenFactor', 'PoseOrientationPriorFactor', 'PosePositionBetweenFactor', 'PosePositionPriorFactor', 'PoseVariable', 'RotationPriorFactor', 'RotationVariable', 'SparseOptimizer', 'Variable', 'VariableType']
class BearingObservationFactor(Factor):
    def __init__(self, id: int, pose_var: PoseVariable, landmark_var: LandmarkVariable, bearing_C_observed: numpy.ndarray[numpy.float64[3, 1]], angle_sigma: float = 1.0) -> None:
        """
        Create a BearingObservationFactor
        """
class BearingProjectionFactor2D(Factor):
    def __init__(self, id: int, pose: PoseVariable, landmark: LandmarkVariable, bearing_C_observed: numpy.ndarray[numpy.float64[3, 1]], sigma: float = 1.0, along_tolerance_epsilon: float = 1e-06) -> None:
        """
        Create a BearingProjectionFactor2D
        """
class Factor:
    def compute_residual(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def id(self) -> int:
        ...
    def name(self) -> str:
        ...
    def residual_size(self) -> int:
        ...
    def type(self) -> FactorType:
        ...
    def variables(self) -> list[Variable]:
        ...
    def weight(self) -> float:
        ...
class FactorGraph:
    def __init__(self) -> None:
        ...
    def add_factor(self, arg0: Factor) -> None:
        ...
    def add_variable(self, arg0: Variable) -> None:
        ...
    def apply_increment(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def compute_full_jacobian_and_residual(self) -> None:
        ...
    def compute_full_jacobian_matrix(self) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def compute_full_residual_vector(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def compute_sparse_jacobian_matrix(self) -> scipy.sparse.csc_matrix:
        ...
    def detailed_factor_test(self, jacobian_tol: float, verbose: bool = False) -> bool:
        ...
    def finalize_structure(self) -> None:
        ...
    def get_all_factors(self) -> list[Factor]:
        ...
    def get_all_variables(self) -> list[Variable]:
        ...
    def get_variable(self, arg0: int) -> Variable:
        ...
    def get_variable_vector(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def jacobian(self) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def jacobian_valid(self) -> bool:
        ...
    def num_residuals(self) -> int:
        ...
    def num_values(self) -> int:
        ...
    def num_variables(self) -> int:
        ...
    def print_jacobian_and_residual(self, detailed: bool = False) -> None:
        ...
    def print_structure(self) -> None:
        ...
    def print_variables(self) -> None:
        ...
    def residual(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def residual_valid(self) -> bool:
        ...
    def set_variable_values_from_vector(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def set_verbose(self, arg0: bool) -> None:
        ...
    def sparse_jacobian(self) -> scipy.sparse.csc_matrix:
        ...
class FactorType:
    """
    Members:
    
      none
    
      bearing_observation
    
      inverse_range_bearing
    
      generic_prior
    
      generic_between
    
      pose_position_prior
    
      pose_orientation_prior
    
      pose_position_between
    
      pose_orientation_between
    """
    __members__: typing.ClassVar[dict[str, FactorType]]  # value = {'none': <FactorType.none: 0>, 'bearing_observation': <FactorType.bearing_observation: 1>, 'inverse_range_bearing': <FactorType.inverse_range_bearing: 2>, 'generic_prior': <FactorType.generic_prior: 3>, 'generic_between': <FactorType.generic_between: 4>, 'pose_position_prior': <FactorType.pose_position_prior: 5>, 'pose_orientation_prior': <FactorType.pose_orientation_prior: 6>, 'pose_position_between': <FactorType.pose_position_between: 7>, 'pose_orientation_between': <FactorType.pose_orientation_between: 8>}
    bearing_observation: typing.ClassVar[FactorType]  # value = <FactorType.bearing_observation: 1>
    generic_between: typing.ClassVar[FactorType]  # value = <FactorType.generic_between: 4>
    generic_prior: typing.ClassVar[FactorType]  # value = <FactorType.generic_prior: 3>
    inverse_range_bearing: typing.ClassVar[FactorType]  # value = <FactorType.inverse_range_bearing: 2>
    none: typing.ClassVar[FactorType]  # value = <FactorType.none: 0>
    pose_orientation_between: typing.ClassVar[FactorType]  # value = <FactorType.pose_orientation_between: 8>
    pose_orientation_prior: typing.ClassVar[FactorType]  # value = <FactorType.pose_orientation_prior: 6>
    pose_position_between: typing.ClassVar[FactorType]  # value = <FactorType.pose_position_between: 7>
    pose_position_prior: typing.ClassVar[FactorType]  # value = <FactorType.pose_position_prior: 5>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GenericBetweenFactor(Factor):
    def __init__(self, id: int, var_a: Variable, var_b: Variable, measured_diff: Variable, sigma: float = 1.0) -> None:
        """
        Create a GenericBetweenFactor
        """
class GenericPriorFactor(Factor):
    def __init__(self, id: int, variable: Variable, prior_value: numpy.ndarray[numpy.float64[m, 1]], sigma: float = 1.0) -> None:
        """
        Create a GenericPriorFactor
        """
class GenericVariable(Variable):
    def __init__(self, id: int, initial_value: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        """
        Create a GenericVariable with arbitrary dimension
        """
    def set_is_constant(self, arg0: bool) -> None:
        ...
class InverseRangeBearingFactor(Factor):
    def __init__(self, id: int, pose_var: PoseVariable, inverse_range_var: InverseRangeVariable, bearing_C_observed: numpy.ndarray[numpy.float64[3, 1]], angle_sigma: float = 1.0) -> None:
        """
        Create an InverseRangeBearingFactor
        """
    def bearing_C_obs(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
class InverseRangeVariable(Variable):
    maximum_inverse_range: float
    minimum_inverse_range: float
    def __init__(self, id: int, origin_pos_W: numpy.ndarray[numpy.float64[3, 1]], bearing_W: numpy.ndarray[numpy.float64[3, 1]], initial_range: float) -> None:
        """
        Create an InverseRangeVariable
        """
    def bearing_W(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def inverse_range(self) -> float:
        ...
    def origin_pos_W(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def pos_W(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def set_is_constant(self, arg0: bool) -> None:
        ...
class LandmarkVariable(Variable):
    def __init__(self, id: int, pos_W_init: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Create a LandmarkVariable with 3D position
        """
    def pos_W(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def set_is_constant(self, arg0: bool) -> None:
        ...
class OptimizerMethod:
    """
    Members:
    
      GaussNewton
    
      LevenbergMarquardt
    """
    GaussNewton: typing.ClassVar[OptimizerMethod]  # value = <OptimizerMethod.GaussNewton: 0>
    LevenbergMarquardt: typing.ClassVar[OptimizerMethod]  # value = <OptimizerMethod.LevenbergMarquardt: 1>
    __members__: typing.ClassVar[dict[str, OptimizerMethod]]  # value = {'GaussNewton': <OptimizerMethod.GaussNewton: 0>, 'LevenbergMarquardt': <OptimizerMethod.LevenbergMarquardt: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OptimizerSettings:
    check_rank_deficiency: bool
    initial_lambda: float
    lambda_down_factor: float
    lambda_up_factor: float
    learning_rate: float
    max_lambda: float
    max_num_iterations: int
    method: OptimizerMethod
    residual_tolerance: float
    step_tolerance: float
    verbose: bool
    def __init__(self) -> None:
        ...
class OptimizerStats:
    chi2: float
    current_iteration: int
    damping_parameter: float
    delta_norm: float
    rank: int
    residual_norm: float
    valid: bool
    def __init__(self) -> None:
        ...
class PoseOrientationBetweenFactor(Factor):
    def __init__(self, id: int, pose1: PoseVariable, pose2: PoseVariable, calibration_rotation_12: RotationVariable, angle_sigma: float = 1.0) -> None:
        """
        Create a PoseOrientationBetweenFactor
        """
class PoseOrientationPriorFactor(Factor):
    def __init__(self, id: int, pose: PoseVariable, rotvec_prior: numpy.ndarray[numpy.float64[3, 1]], sigma: float = 1.0) -> None:
        """
        Create a PoseOrientationPriorFactor
        """
class PosePositionBetweenFactor(Factor):
    def __init__(self, id: int, pose_a: PoseVariable, pose_b: PoseVariable, measured_diff: Variable, sigma: float = 1.0) -> None:
        """
        Create a PosePositionBetweenFactor
        """
class PosePositionPriorFactor(Factor):
    def __init__(self, id: int, pose: PoseVariable, pos_prior: numpy.ndarray[numpy.float64[3, 1]], sigma: float = 1.0) -> None:
        """
        Create a PosePositionPriorFactor
        """
class PoseVariable(Variable):
    @typing.overload
    def __init__(self, id: int, pose_CW_init: numpy.ndarray[numpy.float64[6, 1]]) -> None:
        """
        Create a PoseVariable with SE(3) pose
        """
    @typing.overload
    def __init__(self, id: int, pos_W: numpy.ndarray[numpy.float64[3, 1]], dcm_CW: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        """
        Create a PoseVariable with position and rotation matrix
        """
    def dcm_CW(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    def pos_W(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def rot_CW(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def set_is_constant(self, arg0: bool) -> None:
        ...
class RotationPriorFactor(Factor):
    def __init__(self, id: int, rotation: RotationVariable, dcm_AB_prior: numpy.ndarray[numpy.float64[3, 3]], sigma: float = 1.0) -> None:
        """
        Create a RotationPriorFactor
        """
class RotationVariable(Variable):
    def __init__(self, id: int, dcm_AB: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        """
        Create a RotationVariable with DCM
        """
    def dcm_AB(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    def rotation(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    def set_is_constant(self, arg0: bool) -> None:
        ...
class SparseOptimizer:
    current_stats: OptimizerStats
    initial_stats: OptimizerStats
    def __init__(self) -> None:
        ...
    def optimize(self) -> None:
        ...
    def settings(self) -> OptimizerSettings:
        ...
    def setup(self, arg0: FactorGraph, arg1: OptimizerSettings) -> None:
        ...
class Variable:
    def apply_increment(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def clone(self) -> Variable:
        ...
    def id(self) -> int:
        ...
    def is_constant(self) -> bool:
        ...
    def name(self) -> str:
        ...
    def set_value_from_vector(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def size(self) -> int:
        ...
    def type(self) -> VariableType:
        ...
    def value(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
class VariableType:
    """
    Members:
    
      none
    
      pose
    
      landmark
    
      inverse_range_landmark
    
      extrinsic_rotation
    
      generic
    """
    __members__: typing.ClassVar[dict[str, VariableType]]  # value = {'none': <VariableType.none: 0>, 'pose': <VariableType.pose: 1>, 'landmark': <VariableType.landmark: 2>, 'inverse_range_landmark': <VariableType.inverse_range_landmark: 3>, 'extrinsic_rotation': <VariableType.extrinsic_rotation: 4>, 'generic': <VariableType.generic: 5>}
    extrinsic_rotation: typing.ClassVar[VariableType]  # value = <VariableType.extrinsic_rotation: 4>
    generic: typing.ClassVar[VariableType]  # value = <VariableType.generic: 5>
    inverse_range_landmark: typing.ClassVar[VariableType]  # value = <VariableType.inverse_range_landmark: 3>
    landmark: typing.ClassVar[VariableType]  # value = <VariableType.landmark: 2>
    none: typing.ClassVar[VariableType]  # value = <VariableType.none: 0>
    pose: typing.ClassVar[VariableType]  # value = <VariableType.pose: 1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...

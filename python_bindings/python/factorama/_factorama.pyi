"""
Factorama Python bindings - factor graph optimization library
"""
from __future__ import annotations
import numpy
import numpy.typing
import scipy.sparse
import typing
__all__: list[str] = ['BearingObservationFactor', 'BearingObservationFactor2D', 'BearingProjectionFactor2D', 'CoordinateTransformFactor', 'ExpMapSO3', 'Factor', 'FactorGraph', 'FactorType', 'GenericBetweenFactor', 'GenericPriorFactor', 'GenericVariable', 'InverseRangeBearingFactor', 'InverseRangeVariable', 'LandmarkVariable', 'LinearVelocityFactor', 'LogMapSO3', 'OptimizerMethod', 'OptimizerSettings', 'OptimizerStats', 'OptimizerStatus', 'PlaneFactor', 'PlanePriorFactor', 'PlaneVariable', 'Pose2DBetweenFactor', 'Pose2DPriorFactor', 'Pose2DVariable', 'PoseOrientationBetweenFactor', 'PoseOrientationPriorFactor', 'PosePositionBetweenFactor', 'PosePositionPriorFactor', 'PoseVariable', 'RangeBearingFactor2D', 'RotationPriorFactor', 'RotationVariable', 'SparseOptimizer', 'Variable', 'VariableType']
class BearingObservationFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose_var: PoseVariable, landmark_var: LandmarkVariable, bearing_C_observed: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], angle_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a BearingObservationFactor
        """
    def bearing_C_obs(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
class BearingObservationFactor2D(Factor):
    def __init__(self, id: typing.SupportsInt, pose_var: Pose2DVariable, landmark_var: Variable, bearing_angle_obs: typing.SupportsFloat, angle_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a BearingObservationFactor2D
        """
    def bearing_angle_obs(self) -> float:
        ...
class BearingProjectionFactor2D(Factor):
    def __init__(self, id: typing.SupportsInt, pose: PoseVariable, landmark: LandmarkVariable, bearing_C_observed: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], sigma: typing.SupportsFloat = 1.0, along_tolerance_epsilon: typing.SupportsFloat = 1e-06) -> None:
        """
        Create a BearingProjectionFactor2D
        """
class CoordinateTransformFactor(Factor):
    def __init__(self, id: typing.SupportsInt, rot_AB: RotationVariable, B_origin_A: GenericVariable, scale_AB: GenericVariable, lm_A: LandmarkVariable, lm_B: LandmarkVariable, sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a CoordinateTransformFactor
        """
    def weight(self) -> float:
        ...
class Factor:
    def compute_residual(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
class FactorGraph:
    def __init__(self) -> None:
        ...
    def add_factor(self, arg0: Factor) -> None:
        ...
    def add_variable(self, arg0: Variable) -> None:
        ...
    def apply_increment(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    def compute_full_jacobian_and_residual(self) -> None:
        ...
    def compute_full_jacobian_matrix(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def compute_full_residual_vector(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def compute_sparse_jacobian_matrix(self) -> scipy.sparse.csc_matrix:
        ...
    def detailed_factor_test(self, jacobian_tol: typing.SupportsFloat, verbose: bool = False) -> bool:
        ...
    def finalize_structure(self) -> None:
        ...
    def get_all_factors(self) -> list[Factor]:
        ...
    def get_all_variables(self) -> list[Variable]:
        ...
    def get_variable(self, arg0: typing.SupportsInt) -> Variable:
        ...
    def get_variable_vector(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def jacobian(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
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
    def residual(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def residual_valid(self) -> bool:
        ...
    def set_variable_values_from_vector(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
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
    
      plane_factor
    
      plane_prior
    
      bearing_observation_2d
    
      range_bearing_2d
    
      pose_2d_prior
    
      pose_2d_between
    
      custom
    """
    __members__: typing.ClassVar[dict[str, FactorType]]  # value = {'none': <FactorType.none: 0>, 'bearing_observation': <FactorType.bearing_observation: 1>, 'inverse_range_bearing': <FactorType.inverse_range_bearing: 2>, 'generic_prior': <FactorType.generic_prior: 3>, 'generic_between': <FactorType.generic_between: 4>, 'pose_position_prior': <FactorType.pose_position_prior: 5>, 'pose_orientation_prior': <FactorType.pose_orientation_prior: 6>, 'pose_position_between': <FactorType.pose_position_between: 7>, 'pose_orientation_between': <FactorType.pose_orientation_between: 8>, 'plane_factor': <FactorType.plane_factor: 9>, 'plane_prior': <FactorType.plane_prior: 10>, 'bearing_observation_2d': <FactorType.bearing_observation_2d: 11>, 'range_bearing_2d': <FactorType.range_bearing_2d: 12>, 'pose_2d_prior': <FactorType.pose_2d_prior: 13>, 'pose_2d_between': <FactorType.pose_2d_between: 14>, 'custom': <FactorType.custom: 15>}
    bearing_observation: typing.ClassVar[FactorType]  # value = <FactorType.bearing_observation: 1>
    bearing_observation_2d: typing.ClassVar[FactorType]  # value = <FactorType.bearing_observation_2d: 11>
    custom: typing.ClassVar[FactorType]  # value = <FactorType.custom: 15>
    generic_between: typing.ClassVar[FactorType]  # value = <FactorType.generic_between: 4>
    generic_prior: typing.ClassVar[FactorType]  # value = <FactorType.generic_prior: 3>
    inverse_range_bearing: typing.ClassVar[FactorType]  # value = <FactorType.inverse_range_bearing: 2>
    none: typing.ClassVar[FactorType]  # value = <FactorType.none: 0>
    plane_factor: typing.ClassVar[FactorType]  # value = <FactorType.plane_factor: 9>
    plane_prior: typing.ClassVar[FactorType]  # value = <FactorType.plane_prior: 10>
    pose_2d_between: typing.ClassVar[FactorType]  # value = <FactorType.pose_2d_between: 14>
    pose_2d_prior: typing.ClassVar[FactorType]  # value = <FactorType.pose_2d_prior: 13>
    pose_orientation_between: typing.ClassVar[FactorType]  # value = <FactorType.pose_orientation_between: 8>
    pose_orientation_prior: typing.ClassVar[FactorType]  # value = <FactorType.pose_orientation_prior: 6>
    pose_position_between: typing.ClassVar[FactorType]  # value = <FactorType.pose_position_between: 7>
    pose_position_prior: typing.ClassVar[FactorType]  # value = <FactorType.pose_position_prior: 5>
    range_bearing_2d: typing.ClassVar[FactorType]  # value = <FactorType.range_bearing_2d: 12>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
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
    def __init__(self, id: typing.SupportsInt, var_a: Variable, var_b: Variable, measured_diff: Variable, sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a GenericBetweenFactor
        """
class GenericPriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, variable: Variable, prior_value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a GenericPriorFactor
        """
class GenericVariable(Variable):
    def __init__(self, id: typing.SupportsInt, initial_value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        """
        Create a GenericVariable with arbitrary dimension
        """
class InverseRangeBearingFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose_var: PoseVariable, inverse_range_var: InverseRangeVariable, bearing_C_observed: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], angle_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create an InverseRangeBearingFactor
        """
    def bearing_C_obs(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
class InverseRangeVariable(Variable):
    def __init__(self, id: typing.SupportsInt, origin_pos_W: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], bearing_W: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], initial_range: typing.SupportsFloat) -> None:
        """
        Create an InverseRangeVariable
        """
    def bearing_W(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
    def inverse_range(self) -> float:
        ...
    def origin_pos_W(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
    def pos_W(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
    @property
    def maximum_inverse_range(self) -> float:
        ...
    @maximum_inverse_range.setter
    def maximum_inverse_range(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def minimum_inverse_range(self) -> float:
        ...
    @minimum_inverse_range.setter
    def minimum_inverse_range(self, arg0: typing.SupportsFloat) -> None:
        ...
class LandmarkVariable(Variable):
    def __init__(self, id: typing.SupportsInt, pos_W_init: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        """
        Create a LandmarkVariable with 3D position
        """
    def pos_W(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
class LinearVelocityFactor(Factor):
    def __init__(self, id: typing.SupportsInt, var_1: Variable, var_2: Variable, velocity_variable: Variable, dt: typing.SupportsFloat, sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a LinearVelocityFactor
        """
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
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
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
    method: OptimizerMethod
    verbose: bool
    def __init__(self) -> None:
        ...
    @property
    def initial_lambda(self) -> float:
        ...
    @initial_lambda.setter
    def initial_lambda(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def lambda_down_factor(self) -> float:
        ...
    @lambda_down_factor.setter
    def lambda_down_factor(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def lambda_up_factor(self) -> float:
        ...
    @lambda_up_factor.setter
    def lambda_up_factor(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def learning_rate(self) -> float:
        ...
    @learning_rate.setter
    def learning_rate(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def max_lambda(self) -> float:
        ...
    @max_lambda.setter
    def max_lambda(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def max_num_iterations(self) -> int:
        ...
    @max_num_iterations.setter
    def max_num_iterations(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def residual_tolerance(self) -> float:
        ...
    @residual_tolerance.setter
    def residual_tolerance(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def step_tolerance(self) -> float:
        ...
    @step_tolerance.setter
    def step_tolerance(self, arg0: typing.SupportsFloat) -> None:
        ...
class OptimizerStats:
    status: OptimizerStatus
    valid: bool
    def __init__(self) -> None:
        ...
    @property
    def chi2(self) -> float:
        ...
    @chi2.setter
    def chi2(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def current_iteration(self) -> int:
        ...
    @current_iteration.setter
    def current_iteration(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def damping_parameter(self) -> float:
        ...
    @damping_parameter.setter
    def damping_parameter(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def delta_norm(self) -> float:
        ...
    @delta_norm.setter
    def delta_norm(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def rank(self) -> int:
        ...
    @rank.setter
    def rank(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def residual_norm(self) -> float:
        ...
    @residual_norm.setter
    def residual_norm(self, arg0: typing.SupportsFloat) -> None:
        ...
class OptimizerStatus:
    """
    Members:
    
      SUCCESS
    
      RUNNING
    
      SINGULAR_HESSIAN
    
      ILL_CONDITIONED
    
      DIVERGED
    
      FAILED
    """
    DIVERGED: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.DIVERGED: 4>
    FAILED: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.FAILED: 5>
    ILL_CONDITIONED: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.ILL_CONDITIONED: 3>
    RUNNING: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.RUNNING: 1>
    SINGULAR_HESSIAN: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.SINGULAR_HESSIAN: 2>
    SUCCESS: typing.ClassVar[OptimizerStatus]  # value = <OptimizerStatus.SUCCESS: 0>
    __members__: typing.ClassVar[dict[str, OptimizerStatus]]  # value = {'SUCCESS': <OptimizerStatus.SUCCESS: 0>, 'RUNNING': <OptimizerStatus.RUNNING: 1>, 'SINGULAR_HESSIAN': <OptimizerStatus.SINGULAR_HESSIAN: 2>, 'ILL_CONDITIONED': <OptimizerStatus.ILL_CONDITIONED: 3>, 'DIVERGED': <OptimizerStatus.DIVERGED: 4>, 'FAILED': <OptimizerStatus.FAILED: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PlaneFactor(Factor):
    @typing.overload
    def __init__(self, id: typing.SupportsInt, point_var: Variable, plane_var: PlaneVariable, sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a PlaneFactor
        """
    @typing.overload
    def __init__(self, id: typing.SupportsInt, point_var: Variable, plane_var: PlaneVariable, sigma: typing.SupportsFloat, do_distance_scaling: bool, dist_scaling_r0: typing.SupportsFloat, dist_scaling_p0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        """
        Create a PlaneFactor with distance scaling
        """
class PlanePriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, plane: PlaneVariable, normal_prior: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], distance_prior: typing.SupportsFloat, normal_sigma: typing.SupportsFloat = 1.0, distance_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a PlanePriorFactor
        """
class PlaneVariable(Variable):
    def __init__(self, id: typing.SupportsInt, normal_vector: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], distance: typing.SupportsFloat) -> None:
        """
        Create a PlaneVariable with normal vector and distance
        """
    def distance_from_origin(self) -> float:
        ...
    def distance_from_point(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> float:
        ...
    def unit_vector(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
class Pose2DBetweenFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose_a: Pose2DVariable, pose_b: Pose2DVariable, measured_between_variable: Variable, position_sigma: typing.SupportsFloat, angle_sigma: typing.SupportsFloat, local_frame: bool = False) -> None:
        """
        Create a Pose2DBetweenFactor
        """
class Pose2DPriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose_var: Pose2DVariable, pose_prior: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], position_sigma: typing.SupportsFloat, angle_sigma: typing.SupportsFloat) -> None:
        """
        Create a Pose2DPriorFactor
        """
class Pose2DVariable(Variable):
    def __init__(self, id: typing.SupportsInt, pose_2d: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        """
        Create a Pose2DVariable with 2D pose [x, y, theta]
        """
    def dcm_2d(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 2]"]:
        ...
    def pos_2d(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]:
        ...
    def set_pos_2d(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]) -> None:
        ...
    def set_theta(self, arg0: typing.SupportsFloat) -> None:
        ...
    def theta(self) -> float:
        ...
class PoseOrientationBetweenFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose1: PoseVariable, pose2: PoseVariable, calibration_rotation_12: RotationVariable, angle_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a PoseOrientationBetweenFactor
        """
class PoseOrientationPriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose: PoseVariable, rotvec_prior: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"], sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a PoseOrientationPriorFactor
        """
class PosePositionBetweenFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose_a: PoseVariable, pose_b: PoseVariable, measured_diff: Variable, sigma: typing.SupportsFloat = 1.0, local_frame: bool = False) -> None:
        """
        Create a PosePositionBetweenFactor
        """
class PosePositionPriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, pose: PoseVariable, pos_prior: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a PosePositionPriorFactor
        """
class PoseVariable(Variable):
    @typing.overload
    def __init__(self, id: typing.SupportsInt, pose_CW_init: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[6, 1]"]) -> None:
        """
        Create a PoseVariable with SE(3) pose
        """
    @typing.overload
    def __init__(self, id: typing.SupportsInt, pos_W: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], dcm_CW: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"]) -> None:
        """
        Create a PoseVariable with position and rotation matrix
        """
    def dcm_CW(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
        ...
    def pos_W(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
    def rot_CW(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
class RangeBearingFactor2D(Factor):
    def __init__(self, id: typing.SupportsInt, pose_var: Pose2DVariable, landmark_var: Variable, range_obs: typing.SupportsFloat, bearing_angle_obs: typing.SupportsFloat, range_sigma: typing.SupportsFloat = 1.0, bearing_sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a RangeBearingFactor2D
        """
    def bearing_angle_obs(self) -> float:
        ...
    def range_obs(self) -> float:
        ...
class RotationPriorFactor(Factor):
    def __init__(self, id: typing.SupportsInt, rotation: RotationVariable, dcm_AB_prior: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"], sigma: typing.SupportsFloat = 1.0) -> None:
        """
        Create a RotationPriorFactor
        """
class RotationVariable(Variable):
    def __init__(self, id: typing.SupportsInt, dcm_AB: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"]) -> None:
        """
        Create a RotationVariable with DCM
        """
    def dcm_AB(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
        ...
    def rotation(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
        ...
class SparseOptimizer:
    current_stats: OptimizerStats
    initial_stats: OptimizerStats
    def __init__(self) -> None:
        ...
    def estimate_covariance(self, variable: Variable) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], bool]:
        """
        Estimate covariance matrix for a specific variable, returns (matrix, valid)
        """
    def optimize(self) -> None:
        ...
    def prepare_to_estimate_covariances(self) -> None:
        ...
    def print_all_covariances(self) -> None:
        ...
    def settings(self) -> OptimizerSettings:
        ...
    def setup(self, arg0: FactorGraph, arg1: OptimizerSettings) -> None:
        ...
class Variable:
    """
    """
    def apply_increment(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    def clone(self) -> Variable:
        ...
    def id(self) -> int:
        ...
    def is_constant(self) -> bool:
        ...
    def name(self) -> str:
        ...
    def set_constant(self, arg0: bool) -> None:
        ...
    def set_value_from_vector(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    def size(self) -> int:
        ...
    def type(self) -> VariableType:
        ...
    def value(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
    
      plane
    
      pose_2d
    """
    __members__: typing.ClassVar[dict[str, VariableType]]  # value = {'none': <VariableType.none: 0>, 'pose': <VariableType.pose: 1>, 'landmark': <VariableType.landmark: 2>, 'inverse_range_landmark': <VariableType.inverse_range_landmark: 3>, 'extrinsic_rotation': <VariableType.extrinsic_rotation: 4>, 'generic': <VariableType.generic: 5>, 'plane': <VariableType.plane: 6>, 'pose_2d': <VariableType.pose_2d: 7>}
    extrinsic_rotation: typing.ClassVar[VariableType]  # value = <VariableType.extrinsic_rotation: 4>
    generic: typing.ClassVar[VariableType]  # value = <VariableType.generic: 5>
    inverse_range_landmark: typing.ClassVar[VariableType]  # value = <VariableType.inverse_range_landmark: 3>
    landmark: typing.ClassVar[VariableType]  # value = <VariableType.landmark: 2>
    none: typing.ClassVar[VariableType]  # value = <VariableType.none: 0>
    plane: typing.ClassVar[VariableType]  # value = <VariableType.plane: 6>
    pose: typing.ClassVar[VariableType]  # value = <VariableType.pose: 1>
    pose_2d: typing.ClassVar[VariableType]  # value = <VariableType.pose_2d: 7>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def ExpMapSO3(omega: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
    """
    Exponential map from so(3) to SO(3). Converts a rotation vector to a rotation matrix.
    """
def LogMapSO3(R: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Logarithm map from SO(3) to so(3). Converts a rotation matrix to a rotation vector.
    """

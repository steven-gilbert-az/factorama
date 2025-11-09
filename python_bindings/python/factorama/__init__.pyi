"""

Factorama Python bindings

A Python interface to the Factorama C++ factor graph optimization library.
"""
from __future__ import annotations
from factorama._factorama import BearingObservationFactor
from factorama._factorama import BearingProjectionFactor2D
from factorama._factorama import Factor
from factorama._factorama import FactorGraph
from factorama._factorama import FactorType
from factorama._factorama import GenericBetweenFactor
from factorama._factorama import GenericPriorFactor
from factorama._factorama import GenericVariable
from factorama._factorama import InverseRangeBearingFactor
from factorama._factorama import InverseRangeVariable
from factorama._factorama import LandmarkVariable
from factorama._factorama import LinearVelocityFactor
from factorama._factorama import PlaneFactor
from factorama._factorama import PlanePriorFactor
from factorama._factorama import PlaneVariable
from factorama._factorama import OptimizerMethod
from factorama._factorama import OptimizerSettings
from factorama._factorama import OptimizerStats
from factorama._factorama import OptimizerStatus
from factorama._factorama import PoseOrientationBetweenFactor
from factorama._factorama import PoseOrientationPriorFactor
from factorama._factorama import PosePositionBetweenFactor
from factorama._factorama import PosePositionPriorFactor
from factorama._factorama import PoseVariable
from factorama._factorama import RotationPriorFactor
from factorama._factorama import RotationVariable
from factorama._factorama import SparseOptimizer
from factorama._factorama import Variable
from factorama._factorama import VariableType
from factorama._factorama import ExpMapSO3
from factorama._factorama import LogMapSO3
from . import _factorama
from typing import Optional, Any

def PlotFactorGraph(
    factor_graph: FactorGraph,
    axis_handle: Optional[Any] = None,
    plot_covariance: bool = False,
    optimizer: Optional[SparseOptimizer] = None,
    plot_3d: bool = True
) -> Any: ...

__all__: list[str] = ['BearingObservationFactor', 'BearingProjectionFactor2D', 'ExpMapSO3', 'Factor', 'FactorGraph', 'FactorType', 'GenericBetweenFactor', 'GenericPriorFactor', 'GenericVariable', 'InverseRangeBearingFactor', 'InverseRangeVariable', 'LandmarkVariable', 'LinearVelocityFactor', 'LogMapSO3', 'OptimizerMethod', 'OptimizerSettings', 'OptimizerStats', 'OptimizerStatus', 'PlaneFactor', 'PlanePriorFactor', 'PlaneVariable', 'PlotFactorGraph', 'PoseOrientationBetweenFactor', 'PoseOrientationPriorFactor', 'PosePositionBetweenFactor', 'PosePositionPriorFactor', 'PoseVariable', 'RotationPriorFactor', 'RotationVariable', 'SparseOptimizer', 'Variable', 'VariableType']
__version__: str

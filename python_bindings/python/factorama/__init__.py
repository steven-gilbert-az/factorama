"""
Factorama Python bindings

A Python interface to the Factorama C++ factor graph optimization library.
"""

from ._factorama import *

# Get version from package metadata
try:
    from importlib.metadata import version
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import version

try:
    __version__ = version("factorama")
except Exception:
    # Fallback if package metadata not available
    __version__ = "unknown"


def PlotFactorGraph(factor_graph, axis_handle=None, plot_covariance=False, optimizer=None, plot_3d=True):
    """
    Plot a factor graph showing poses and landmarks.

    Args:
        factor_graph: FactorGraph object to visualize
        axis_handle: Optional matplotlib axis handle (if None, creates new figure)
        plot_covariance: Whether to plot covariance ellipses (requires optimizer)
        optimizer: SparseOptimizer object (required if plot_covariance=True)
        plot_3d: If True, creates 3D plot; if False, creates 2D plot (x,y only)

    Returns:
        matplotlib axis handle
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for PlotFactorGraph")

    # Create figure/axis if not provided
    if axis_handle is None:
        if plot_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = axis_handle

    # Get all variables from the factor graph
    variables = factor_graph.get_all_variables()

    pose_positions = []
    landmark_positions = []

    # Extract positions from variables
    for var in variables:
        if var.type() == VariableType.pose:
            # PoseVariable has pos_W() method
            pos = var.pos_W()
            pose_positions.append([pos[0], pos[1], pos[2]])
        elif var.type() == VariableType.landmark:
            # LandmarkVariable has pos_W() method
            pos = var.pos_W()
            landmark_positions.append([pos[0], pos[1], pos[2]])

    # Convert to numpy arrays for easier plotting
    if pose_positions:
        pose_positions = np.array(pose_positions)
    if landmark_positions:
        landmark_positions = np.array(landmark_positions)

    # Plot poses and landmarks
    if plot_3d:
        # 3D plotting
        if len(pose_positions) > 0:
            ax.scatter(pose_positions[:, 0], pose_positions[:, 1], pose_positions[:, 2],
                      c='green', marker='o', s=50, label='Poses', alpha=0.8)

        if len(landmark_positions) > 0:
            ax.scatter(landmark_positions[:, 0], landmark_positions[:, 1], landmark_positions[:, 2],
                      c='blue', marker='o', s=50, label='Landmarks', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        # 2D plotting (x, y only)
        if len(pose_positions) > 0:
            ax.scatter(pose_positions[:, 0], pose_positions[:, 1],
                      c='green', marker='o', s=50, label='Poses', alpha=0.8)

        if len(landmark_positions) > 0:
            ax.scatter(landmark_positions[:, 0], landmark_positions[:, 1],
                      c='blue', marker='o', s=50, label='Landmarks', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Factor Graph Visualization')

    # Make axes equal for better visualization
    if plot_3d:
        # For 3D plots, setting equal aspect ratio is more complex
        if len(pose_positions) > 0 or len(landmark_positions) > 0:
            all_positions = []
            if len(pose_positions) > 0:
                all_positions.append(pose_positions)
            if len(landmark_positions) > 0:
                all_positions.append(landmark_positions)

            if all_positions:
                all_positions = np.vstack(all_positions)
                max_range = np.array([all_positions[:, 0].max() - all_positions[:, 0].min(),
                                    all_positions[:, 1].max() - all_positions[:, 1].min(),
                                    all_positions[:, 2].max() - all_positions[:, 2].min()]).max() / 2.0
                mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
                mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
                mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_aspect('equal', adjustable='box')

    # TODO: Add covariance plotting when plot_covariance=True and optimizer is provided
    if plot_covariance and optimizer is not None:
        print("Covariance plotting not yet implemented")

    return ax
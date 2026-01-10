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


def PlotFactorGraph(factor_graph, axis_handle=None, plot_covariance=False, optimizer=None, plot_3d=True, plot_measurements=True, plot_between_factors=True):
    """
    Plot a factor graph showing poses and landmarks.

    Args:
        factor_graph: FactorGraph object to visualize
        axis_handle: Optional matplotlib axis handle (if None, creates new figure)
        plot_covariance: Whether to plot covariance ellipses/ellipsoids (requires optimizer)
        optimizer: SparseOptimizer object (required if plot_covariance=True)
        plot_3d: If True, creates 3D plot; if False, creates 2D plot (x,y only)
        plot_measurements: Whether to plot bearing/range-bearing measurement lines (default True)
        plot_between_factors: Whether to plot between factor constraints (default True)

    Returns:
        matplotlib axis handle
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        from matplotlib.patches import Ellipse
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

    # Separate lists for different variable types
    pose_3d_positions = []
    pose_3d_vars = []
    landmark_3d_positions = []
    landmark_3d_vars = []
    pose_2d_positions = []
    pose_2d_orientations = []
    pose_2d_vars = []
    landmark_2d_positions = []
    landmark_2d_vars = []

    # Extract positions from variables
    for var in variables:
        if var.type() == VariableType.pose:
            # 3D PoseVariable has pos_W() method
            pos = var.pos_W()
            pose_3d_positions.append([pos[0], pos[1], pos[2]])
            pose_3d_vars.append(var)
        elif var.type() == VariableType.landmark:
            # 3D LandmarkVariable has pos_W() method
            pos = var.pos_W()
            landmark_3d_positions.append([pos[0], pos[1], pos[2]])
            landmark_3d_vars.append(var)
        elif var.type() == VariableType.pose_2d:
            # 2D Pose2DVariable has pos_2d() and theta() methods
            pos = var.pos_2d()
            theta = var.theta()
            pose_2d_positions.append([pos[0], pos[1]])
            pose_2d_orientations.append(theta)
            pose_2d_vars.append(var)
        elif var.type() == VariableType.generic and var.size() == 2:
            # 2D landmarks stored as GenericVariable with size 2
            val = var.value()
            landmark_2d_positions.append([val[0], val[1]])
            landmark_2d_vars.append(var)

    # Convert to numpy arrays for easier plotting
    if pose_3d_positions:
        pose_3d_positions = np.array(pose_3d_positions)
    if landmark_3d_positions:
        landmark_3d_positions = np.array(landmark_3d_positions)
    if pose_2d_positions:
        pose_2d_positions = np.array(pose_2d_positions)
        pose_2d_orientations = np.array(pose_2d_orientations)
    if landmark_2d_positions:
        landmark_2d_positions = np.array(landmark_2d_positions)

    # Prepare covariances if requested
    covariances = {}
    if plot_covariance and optimizer is not None:
        optimizer.prepare_to_estimate_covariances()

        # Get covariances for all variables
        for var in variables:
            cov_matrix, valid = optimizer.estimate_covariance(var)
            if valid:
                covariances[var.id()] = cov_matrix

    # Plot poses and landmarks
    if plot_3d:
        # 3D plotting
        if len(pose_3d_positions) > 0:
            ax.scatter(pose_3d_positions[:, 0], pose_3d_positions[:, 1], pose_3d_positions[:, 2],
                      c='green', marker='o', s=50, label='3D Poses', alpha=0.8)

            # Plot 3D pose orientation arrows (RGB for XYZ axes)
            arrow_length = _compute_arrow_length_3d(pose_3d_positions, landmark_3d_positions,
                                                     pose_2d_positions, landmark_2d_positions)
            _plot_3d_pose_arrows(ax, pose_3d_vars, arrow_length)

            # Plot 3D covariance ellipsoids if available
            if plot_covariance:
                for i, var in enumerate(pose_3d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        # Extract position covariance (first 3x3 block for position)
                        if cov.shape[0] >= 3:
                            pos_cov = cov[:3, :3]
                            _plot_covariance_ellipsoid_3d(ax, pose_3d_positions[i], pos_cov,
                                                          color='green', alpha=0.1)

        if len(landmark_3d_positions) > 0:
            ax.scatter(landmark_3d_positions[:, 0], landmark_3d_positions[:, 1], landmark_3d_positions[:, 2],
                      c='blue', marker='o', s=50, label='3D Landmarks', alpha=0.8)

            # Plot 3D covariance ellipsoids if available
            if plot_covariance:
                for i, var in enumerate(landmark_3d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        if cov.shape[0] >= 3:
                            _plot_covariance_ellipsoid_3d(ax, landmark_3d_positions[i], cov,
                                                          color='blue', alpha=0.1)

        if len(pose_2d_positions) > 0:
            # For 2D poses in 3D plot, set z=0
            ax.scatter(pose_2d_positions[:, 0], pose_2d_positions[:, 1],
                      np.zeros(len(pose_2d_positions)),
                      c='red', marker='^', s=50, label='2D Poses', alpha=0.8)

        if len(landmark_2d_positions) > 0:
            # For 2D landmarks in 3D plot, set z=0
            ax.scatter(landmark_2d_positions[:, 0], landmark_2d_positions[:, 1],
                      np.zeros(len(landmark_2d_positions)),
                      c='cyan', marker='o', s=50, label='2D Landmarks', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        # 2D plotting (x, y only)
        if len(pose_3d_positions) > 0:
            ax.scatter(pose_3d_positions[:, 0], pose_3d_positions[:, 1],
                      c='green', marker='o', s=50, label='3D Poses', alpha=0.8)

            # Plot 2D projection of position covariance
            if plot_covariance:
                for i, var in enumerate(pose_3d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        if cov.shape[0] >= 2:
                            # Extract x-y covariance (first 2x2 block)
                            pos_cov_2d = cov[:2, :2]
                            _plot_covariance_ellipse_2d(ax, pose_3d_positions[i, :2], pos_cov_2d,
                                                        color='green', alpha=0.2)

        if len(landmark_3d_positions) > 0:
            ax.scatter(landmark_3d_positions[:, 0], landmark_3d_positions[:, 1],
                      c='blue', marker='o', s=50, label='3D Landmarks', alpha=0.8)

            # Plot 2D projection of covariance
            if plot_covariance:
                for i, var in enumerate(landmark_3d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        if cov.shape[0] >= 2:
                            pos_cov_2d = cov[:2, :2]
                            _plot_covariance_ellipse_2d(ax, landmark_3d_positions[i, :2], pos_cov_2d,
                                                        color='blue', alpha=0.2)

        if len(pose_2d_positions) > 0:
            ax.scatter(pose_2d_positions[:, 0], pose_2d_positions[:, 1],
                      c='red', marker='^', s=50, label='2D Poses', alpha=0.8)

            # Plot orientation arrows for 2D poses
            arrow_length = _compute_arrow_length(pose_2d_positions, landmark_2d_positions,
                                                 pose_3d_positions, landmark_3d_positions)
            for i in range(len(pose_2d_positions)):
                dx = arrow_length * np.cos(pose_2d_orientations[i])
                dy = arrow_length * np.sin(pose_2d_orientations[i])
                ax.arrow(pose_2d_positions[i, 0], pose_2d_positions[i, 1], dx, dy,
                        head_width=arrow_length*0.3, head_length=arrow_length*0.4,
                        fc='red', ec='red', alpha=0.6, linewidth=1.5)

            # Plot 2D covariance ellipses if available
            if plot_covariance:
                for i, var in enumerate(pose_2d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        if cov.shape[0] >= 2:
                            # Extract x-y covariance (first 2x2 block)
                            pos_cov_2d = cov[:2, :2]
                            _plot_covariance_ellipse_2d(ax, pose_2d_positions[i], pos_cov_2d,
                                                        color='red', alpha=0.2)

        if len(landmark_2d_positions) > 0:
            ax.scatter(landmark_2d_positions[:, 0], landmark_2d_positions[:, 1],
                      c='cyan', marker='o', s=50, label='2D Landmarks', alpha=0.8)

            # Plot 2D covariance ellipses if available
            if plot_covariance:
                for i, var in enumerate(landmark_2d_vars):
                    if var.id() in covariances:
                        cov = covariances[var.id()]
                        if cov.shape[0] >= 2:
                            _plot_covariance_ellipse_2d(ax, landmark_2d_positions[i], cov,
                                                        color='cyan', alpha=0.2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Plot bearing and range-bearing measurement lines
    plotted_measurement_types = set()
    if plot_measurements:
        factors = factor_graph.get_all_factors()
        plotted_measurement_types = _plot_bearing_lines(ax, factors, plot_3d)

    # Plot between factor constraints
    plotted_between_types = set()
    if plot_between_factors:
        factors = factor_graph.get_all_factors()
        plotted_between_types = _plot_between_factors(ax, factors, plot_3d)

    # Add legend and grid
    # Add legend entries for measurement lines if they were plotted
    if 'bearing_3d' in plotted_measurement_types or 'bearing_2d' in plotted_measurement_types:
        ax.plot([], [], color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Bearing measurements')
    if 'range_bearing_2d' in plotted_measurement_types:
        ax.plot([], [], color='purple', linestyle='-', linewidth=1.5, alpha=0.7, label='Range-bearing measurements')
    if plotted_between_types:
        ax.plot([], [], color='magenta', linestyle=':', linewidth=2, alpha=0.7, label='Between factors')

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Factor Graph Visualization')

    # Make axes equal for better visualization
    if plot_3d:
        # For 3D plots, setting equal aspect ratio is more complex
        all_positions = []
        if len(pose_3d_positions) > 0:
            all_positions.append(pose_3d_positions)
        if len(landmark_3d_positions) > 0:
            all_positions.append(landmark_3d_positions)
        if len(pose_2d_positions) > 0:
            # Add 2D positions with z=0 for bounds calculation
            pose_2d_with_z = np.column_stack([pose_2d_positions, np.zeros(len(pose_2d_positions))])
            all_positions.append(pose_2d_with_z)
        if len(landmark_2d_positions) > 0:
            landmark_2d_with_z = np.column_stack([landmark_2d_positions, np.zeros(len(landmark_2d_positions))])
            all_positions.append(landmark_2d_with_z)

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

    return ax


def _compute_arrow_length(pose_2d_positions, landmark_2d_positions,
                          pose_3d_positions, landmark_3d_positions):
    """Compute appropriate arrow length based on data scale."""
    import numpy as np

    all_2d_positions = []
    if len(pose_2d_positions) > 0:
        all_2d_positions.append(pose_2d_positions)
    if len(landmark_2d_positions) > 0:
        all_2d_positions.append(landmark_2d_positions)
    if len(pose_3d_positions) > 0:
        all_2d_positions.append(pose_3d_positions[:, :2])
    if len(landmark_3d_positions) > 0:
        all_2d_positions.append(landmark_3d_positions[:, :2])

    if all_2d_positions:
        all_2d_positions = np.vstack(all_2d_positions)
        x_range = all_2d_positions[:, 0].max() - all_2d_positions[:, 0].min()
        y_range = all_2d_positions[:, 1].max() - all_2d_positions[:, 1].min()
        avg_range = (x_range + y_range) / 2.0
        return max(0.05 * avg_range, 0.1)  # 5% of average range, minimum 0.1
    else:
        return 0.5  # default


def _compute_arrow_length_3d(pose_3d_positions, landmark_3d_positions,
                              pose_2d_positions, landmark_2d_positions):
    """Compute appropriate arrow length for 3D visualization based on scene scale."""
    import numpy as np

    all_positions = []
    if len(pose_3d_positions) > 0:
        all_positions.append(pose_3d_positions)
    if len(landmark_3d_positions) > 0:
        all_positions.append(landmark_3d_positions)
    if len(pose_2d_positions) > 0:
        # Add 2D positions with z=0 for scale calculation
        pose_2d_with_z = np.column_stack([pose_2d_positions, np.zeros(len(pose_2d_positions))])
        all_positions.append(pose_2d_with_z)
    if len(landmark_2d_positions) > 0:
        landmark_2d_with_z = np.column_stack([landmark_2d_positions, np.zeros(len(landmark_2d_positions))])
        all_positions.append(landmark_2d_with_z)

    if all_positions:
        all_positions = np.vstack(all_positions)
        x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
        y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
        z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
        avg_range = (x_range + y_range + z_range) / 3.0
        return max(0.08 * avg_range, 0.2)  # 8% of average range, minimum 0.2
    else:
        return 1.0  # default


def _plot_3d_pose_arrows(ax, pose_3d_vars, arrow_length):
    """
    Plot RGB direction arrows for 3D poses showing their orientation.

    Args:
        ax: Matplotlib 3D axis
        pose_3d_vars: List of 3D pose variables
        arrow_length: Length of the arrows to draw
    """
    import numpy as np

    for pose_var in pose_3d_vars:
        # Get pose position and rotation
        pos_W = pose_var.pos_W()
        dcm_CW = pose_var.dcm_CW()

        # dcm_CW transforms vectors from world to camera frame
        # The inverse (transpose) gives us camera axes in world frame
        dcm_WC = dcm_CW.T

        # Extract the three axes of the camera frame expressed in world coordinates
        # These are the columns of dcm_WC
        x_axis_W = dcm_WC[:, 0]  # Camera X-axis in world frame
        y_axis_W = dcm_WC[:, 1]  # Camera Y-axis in world frame
        z_axis_W = dcm_WC[:, 2]  # Camera Z-axis in world frame

        # Plot arrows using quiver (ROS convention: X=red, Y=green, Z=blue)
        # X-axis: red
        ax.quiver(pos_W[0], pos_W[1], pos_W[2],
                 x_axis_W[0], x_axis_W[1], x_axis_W[2],
                 color='red', length=arrow_length,
                 arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # Y-axis: green
        ax.quiver(pos_W[0], pos_W[1], pos_W[2],
                 y_axis_W[0], y_axis_W[1], y_axis_W[2],
                 color='green', length=arrow_length,
                 arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # Z-axis: blue
        ax.quiver(pos_W[0], pos_W[1], pos_W[2],
                 z_axis_W[0], z_axis_W[1], z_axis_W[2],
                 color='blue', length=arrow_length,
                 arrow_length_ratio=0.3, linewidth=2, alpha=0.8)


def _plot_covariance_ellipse_2d(ax, center, cov_matrix, n_std=1.0, color='blue', alpha=0.3):
    """
    Plot a 2D covariance ellipse representing n-sigma confidence region.

    Args:
        ax: Matplotlib axis
        center: [x, y] center position
        cov_matrix: 2x2 covariance matrix
        n_std: Number of standard deviations (default=1.0 for 1-sigma)
        color: Ellipse color
        alpha: Ellipse transparency
    """
    import numpy as np
    from matplotlib.patches import Ellipse

    # Eigenvalue decomposition to get ellipse parameters
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute angle of major axis
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Width and height are 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create and add ellipse
    ellipse = Ellipse(xy=(center[0], center[1]), width=width, height=height,
                     angle=angle, facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5)
    ax.add_patch(ellipse)


def _plot_covariance_ellipsoid_3d(ax, center, cov_matrix, n_std=1.0, color='blue', alpha=0.1, resolution=20):
    """
    Plot a 3D covariance ellipsoid representing n-sigma confidence region.

    Args:
        ax: Matplotlib 3D axis
        center: [x, y, z] center position
        cov_matrix: 3x3 covariance matrix
        n_std: Number of standard deviations (default=1.0 for 1-sigma)
        color: Ellipsoid color
        alpha: Ellipsoid transparency
        resolution: Number of points for mesh
    """
    import numpy as np

    # Eigenvalue decomposition
    D, V = np.linalg.eigh(cov_matrix)

    # Generate unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Stack into 3×N array for vectorized transformation
    pts = np.stack([x.flatten(), y.flatten(), z.flatten()])

    # Transform sphere → ellipsoid using eigendecomposition
    ellipsoid = n_std * V @ np.diag(np.sqrt(D)) @ pts

    # Reshape back to grid and translate to center
    X = ellipsoid[0].reshape(x.shape) + center[0]
    Y = ellipsoid[1].reshape(y.shape) + center[1]
    Z = ellipsoid[2].reshape(z.shape) + center[2]

    # Plot surface
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=True, linewidth=0)


def _plot_bearing_lines(ax, factors, plot_3d):
    """
    Plot bearing and range-bearing measurement lines from poses to landmarks.

    Args:
        ax: Matplotlib axis (2D or 3D)
        factors: List of all factors in the graph
        plot_3d: If True, plotting in 3D; if False, 2D

    Returns:
        set: Set of measurement types that were plotted ('bearing_3d', 'bearing_2d', 'range_bearing_2d')
    """
    import numpy as np

    plotted_types = set()

    for factor in factors:
        factor_type = factor.type()
        variables = factor.variables()

        # 3D Bearing Observation Factor
        if factor_type == FactorType.bearing_observation:
            if len(variables) < 2:
                continue

            pose_var = variables[0]
            landmark_var = variables[1]

            # Get pose position and rotation
            pos_W = pose_var.pos_W()
            dcm_CW = pose_var.dcm_CW()

            # Get bearing in camera frame
            bearing_C = factor.bearing_C_obs()

            # Transform bearing to world frame: bearing_W = dcm_WC @ bearing_C = dcm_CW.T @ bearing_C
            bearing_W = dcm_CW.T @ bearing_C

            # Compute distance to landmark (for line length)
            landmark_pos = landmark_var.pos_W()
            distance = np.linalg.norm(landmark_pos - pos_W)

            # End point of bearing line
            end_point = pos_W + distance * bearing_W.flatten()

            # Plot the line
            if plot_3d:
                ax.plot([pos_W[0], end_point[0]],
                       [pos_W[1], end_point[1]],
                       [pos_W[2], end_point[2]],
                       color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
            else:
                # 2D projection (x-y plane)
                ax.plot([pos_W[0], end_point[0]],
                       [pos_W[1], end_point[1]],
                       color='orange', linestyle='--', linewidth=1.5, alpha=0.6)

            plotted_types.add('bearing_3d')

        # 2D Bearing Observation Factor
        elif factor_type == FactorType.bearing_observation_2d:
            if len(variables) < 2:
                continue

            pose_var = variables[0]
            landmark_var = variables[1]

            # Get pose position and orientation
            pos_2d = pose_var.pos_2d()
            theta = pose_var.theta()

            # Get bearing angle in pose frame
            bearing_angle = factor.bearing_angle_obs()

            # Transform bearing to world frame
            # Bearing vector in pose frame
            bearing_pose = np.array([np.cos(bearing_angle), np.sin(bearing_angle)])

            # Rotation matrix from pose to world (2D rotation)
            c = np.cos(theta)
            s = np.sin(theta)
            R_WC = np.array([[c, -s], [s, c]])

            bearing_W = R_WC @ bearing_pose

            # Compute distance to landmark (for line length)
            landmark_pos = landmark_var.value()
            distance = np.linalg.norm(landmark_pos.flatten()[:2] - pos_2d.flatten())

            # End point of bearing line
            end_point = pos_2d.flatten() + distance * bearing_W

            # Plot the line
            if plot_3d:
                # Plot at z=0
                ax.plot([pos_2d[0], end_point[0]],
                       [pos_2d[1], end_point[1]],
                       [0, 0],
                       color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
            else:
                ax.plot([pos_2d[0], end_point[0]],
                       [pos_2d[1], end_point[1]],
                       color='orange', linestyle='--', linewidth=1.5, alpha=0.6)

            plotted_types.add('bearing_2d')

        # 2D Range-Bearing Factor
        elif factor_type == FactorType.range_bearing_2d:
            if len(variables) < 2:
                continue

            pose_var = variables[0]
            landmark_var = variables[1]

            # Get pose position and orientation
            pos_2d = pose_var.pos_2d()
            theta = pose_var.theta()

            # Get range and bearing measurements
            range_obs = factor.range_obs()
            bearing_angle = factor.bearing_angle_obs()

            # Transform bearing to world frame
            # Bearing vector in pose frame
            bearing_pose = np.array([np.cos(bearing_angle), np.sin(bearing_angle)])

            # Rotation matrix from pose to world (2D rotation)
            c = np.cos(theta)
            s = np.sin(theta)
            R_WC = np.array([[c, -s], [s, c]])

            bearing_W = R_WC @ bearing_pose

            # End point using measured range
            end_point = pos_2d.flatten() + range_obs * bearing_W

            # Plot the line (solid line for range-bearing since we have range info)
            if plot_3d:
                # Plot at z=0
                ax.plot([pos_2d[0], end_point[0]],
                       [pos_2d[1], end_point[1]],
                       [0, 0],
                       color='purple', linestyle='-', linewidth=1.5, alpha=0.7)
            else:
                ax.plot([pos_2d[0], end_point[0]],
                       [pos_2d[1], end_point[1]],
                       color='purple', linestyle='-', linewidth=1.5, alpha=0.7)

            plotted_types.add('range_bearing_2d')

    return plotted_types


def _plot_between_factors(ax, factors, plot_3d):
    """
    Plot between factor constraints as dotted lines connecting poses.

    Args:
        ax: Matplotlib axis (2D or 3D)
        factors: List of all factors in the graph
        plot_3d: If True, plotting in 3D; if False, 2D

    Returns:
        set: Set of between factor types that were plotted
    """
    import numpy as np

    plotted_types = set()

    for factor in factors:
        factor_type = factor.type()
        variables = factor.variables()

        # Check if this is a between factor type
        is_between_factor = False
        pose_var_a = None
        pose_var_b = None

        # 2D Pose Between Factor
        if factor_type == FactorType.pose_2d_between:
            if len(variables) >= 2:
                pose_var_a = variables[0]
                pose_var_b = variables[1]
                is_between_factor = True

        # 3D Pose Position Between Factor
        elif factor_type == FactorType.pose_position_between:
            if len(variables) >= 2:
                pose_var_a = variables[0]
                pose_var_b = variables[1]
                is_between_factor = True

        # 3D Pose Orientation Between Factor
        elif factor_type == FactorType.pose_orientation_between:
            if len(variables) >= 2:
                pose_var_a = variables[0]
                pose_var_b = variables[1]
                is_between_factor = True

        # Generic Between Factor (could connect any variables)
        elif factor_type == FactorType.generic_between:
            if len(variables) >= 2:
                pose_var_a = variables[0]
                pose_var_b = variables[1]
                is_between_factor = True

        # If we found a between factor, plot the connection
        if is_between_factor and pose_var_a is not None and pose_var_b is not None:
            # Get positions based on variable type
            pos_a = None
            pos_b = None

            # Handle 3D poses
            if pose_var_a.type() == VariableType.pose:
                pos_a = pose_var_a.pos_W()
            elif pose_var_a.type() == VariableType.pose_2d:
                pos_a_2d = pose_var_a.pos_2d()
                pos_a = np.array([pos_a_2d[0], pos_a_2d[1], 0.0])

            if pose_var_b.type() == VariableType.pose:
                pos_b = pose_var_b.pos_W()
            elif pose_var_b.type() == VariableType.pose_2d:
                pos_b_2d = pose_var_b.pos_2d()
                pos_b = np.array([pos_b_2d[0], pos_b_2d[1], 0.0])

            # Plot the line if we successfully extracted positions
            if pos_a is not None and pos_b is not None:
                if plot_3d:
                    ax.plot([pos_a[0], pos_b[0]],
                           [pos_a[1], pos_b[1]],
                           [pos_a[2], pos_b[2]],
                           color='magenta', linestyle=':', linewidth=2, alpha=0.7)
                else:
                    # 2D projection (x-y plane)
                    ax.plot([pos_a[0], pos_b[0]],
                           [pos_a[1], pos_b[1]],
                           color='magenta', linestyle=':', linewidth=2, alpha=0.7)

                plotted_types.add(str(factor_type))

    return plotted_types
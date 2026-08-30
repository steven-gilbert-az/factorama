#!/usr/bin/env python3
"""
Demo: 3D Pose Direction Arrows

This script demonstrates the RGB direction arrows for 3D pose variables.
Following ROS convention:
- X-axis: Red
- Y-axis: Green
- Z-axis: Blue

The arrow length automatically scales based on the scene size.
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


def create_demo_graph():
    """Create a simple demo with 4 poses at different orientations"""
    graph = factorama.FactorGraph()

    # Create 4 poses with different orientations to show the RGB arrows clearly
    poses_gt = [
        # [x, y, z, rx, ry, rz] - position and Rodrigues rotation
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),           # Origin, no rotation
        np.array([5.0, 0.0, 0.0, 0.0, 0.0, np.pi/4]),       # Rotated around Z (yaw)
        np.array([5.0, 5.0, 0.0, 0.0, np.pi/4, np.pi/4]),   # Rotated around Y (pitch) and Z
        np.array([0.0, 5.0, 5.0, np.pi/4, 0.0, 0.0]),       # Rotated around X (roll), raised in Z
    ]

    # Create pose variables
    poses = []
    for i, pose_gt in enumerate(poses_gt):
        pose = factorama.PoseVariable(i, pose_gt)
        poses.append(pose)
        graph.add_variable(pose)

        # Add a strong prior to lock each pose in place
        prior = factorama.GenericPriorFactor(100 + i, pose, pose_gt, 0.001)
        graph.add_factor(prior)

    graph.finalize_structure()
    return graph


if __name__ == "__main__":
    print("=" * 70)
    print("Demo: 3D Pose Direction Arrows (ROS Convention)")
    print("=" * 70)
    print("\nCreating demo graph with 4 poses at different orientations...")

    # Create the demo graph
    graph = create_demo_graph()

    print(f"Graph has {graph.num_variables()} pose variables")
    print("\nEach pose is shown with three direction arrows:")
    print("  - Red arrow:   X-axis direction")
    print("  - Green arrow: Y-axis direction")
    print("  - Blue arrow:  Z-axis direction")

    # Create figure with two views
    fig = plt.figure(figsize=(16, 7))

    # View 1: Default 3D view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax1, plot_3d=True, plot_measurements=False)
    ax1.set_title('3D View - Pose Orientation Frames (RGB = XYZ)', fontsize=14, fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # View 2: Different angle
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    factorama.PlotFactorGraph(graph, axis_handle=ax2, plot_3d=True, plot_measurements=False)
    ax2.set_title('3D View - Alternative Angle', fontsize=14, fontweight='bold')
    ax2.view_init(elev=15, azim=135)

    plt.suptitle('3D Pose Direction Arrows Demo (ROS Convention)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = '/tmp/demo_3d_pose_arrows.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

#pragma once

#include <Eigen/Dense>

namespace factorama
{

    constexpr double PI = 3.14159265358979323846;

    /// Create skew-symmetric matrix from 3D vector
    Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d &v);

    /// Unskew (vee operator) — recovers vector from skew-symmetric matrix
    Eigen::Vector3d unskew(const Eigen::Matrix3d &m);

    /// Exponential map from so(3) to SO(3)
    Eigen::Matrix3d ExpMapSO3(const Eigen::Vector3d &omega);

    /// Logarithm map from SO(3) to so(3)
    Eigen::Vector3d LogMapSO3(const Eigen::Matrix3d &R);

    /// Compute the inverse right Jacobian for SO(3)
    Eigen::Matrix3d compute_inverse_right_jacobian_so3(const Eigen::Vector3d &omega);

} // namespace factorama
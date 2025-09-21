#include "random_utils.hpp"
#include <stdexcept>
#include <cmath>

namespace factorama
{

Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d skew;
    skew << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return skew;
}

Eigen::Vector3d unskew(const Eigen::Matrix3d &m)
{
    return 0.5 * Eigen::Vector3d(m(2, 1) - m(1, 2),
                                 m(0, 2) - m(2, 0),
                                 m(1, 0) - m(0, 1));
}

Eigen::Matrix3d ExpMapSO3(const Eigen::Vector3d &omega)
{
    double theta = omega.norm();

    if (theta < 1e-10)
    {
        // First-order approximation for very small angles
        return Eigen::Matrix3d::Identity() + skew_symmetric(omega);
    }

    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d K = skew_symmetric(axis);

    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

Eigen::Vector3d LogMapSO3(const Eigen::Matrix3d &R)
{
    double cos_theta = (R.trace() - 1.0) * 0.5;
    cos_theta = std::min(std::max(cos_theta, -1.0), 1.0); // Clamp for safety
    double theta = std::acos(cos_theta);

    Eigen::Matrix3d Omega = 0.5 * (R - R.transpose());

    if (theta < 1e-5)
    {
        // Use first-order approximation: log(R) ≈ 0.5 * (R - Rᵀ)
        return unskew(Omega);
    }

    if (std::abs(theta - PI) < 1e-3)
    {
        // For angles near π, use Eigen fallback
        Eigen::AngleAxisd aa(R);
        return aa.angle() * aa.axis();
    }

    // General case
    double scale = theta / (2.0 * std::sin(theta));
    return unskew(scale * (R - R.transpose()));
}

Eigen::Matrix3d compute_inverse_right_jacobian_so3(const Eigen::Vector3d &omega)
{
    double theta = omega.norm();

    if (theta < 1e-8)
    {
        // First-order approximation for small angles
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(omega);
    }

    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d K = skew_symmetric(axis);

    double half_theta = 0.5 * theta;
    double tan_half_theta = std::tan(half_theta);

    if (std::abs(tan_half_theta) < 1e-12) {
        throw std::runtime_error("compute_inverse_right_jacobian_so3: tan(half_theta) too close to zero");
    }

    double cot_half_theta = 1.0 / tan_half_theta;

    return (half_theta * cot_half_theta) * Eigen::Matrix3d::Identity() +
           (1.0 - half_theta * cot_half_theta) * (axis * axis.transpose()) -
           half_theta * K;
}

} // namespace factorama
#include "plane_prior_factor.hpp"
#include <cassert>

namespace factorama
{

PlanePriorFactor::PlanePriorFactor(int id,
                                   PlaneVariable* plane,
                                   const Eigen::Vector3d &normal_prior,
                                   double distance_prior,
                                   double normal_sigma,
                                   double distance_sigma)
    : plane_(plane),
      normal_prior_(normal_prior.normalized()),
      distance_prior_(distance_prior),
      weight_normal_(1.0 / normal_sigma),
      weight_distance_(1.0 / distance_sigma)
{
    id_ = id;
    assert(plane != nullptr && "plane cannot be nullptr");
    assert(normal_sigma > 0.0 && "normal_sigma must be greater than zero");
    assert(distance_sigma > 0.0 && "distance_sigma must be greater than zero");
    assert(normal_prior.squaredNorm() > 0.0 && "normal_prior must be nonzero");
}

Eigen::VectorXd PlanePriorFactor::compute_residual() const
{
    Eigen::Vector3d n_current = plane_->unit_vector();
    double d_current = plane_->distance_from_origin();

    Eigen::VectorXd res(4);

    // Normal residual (first 3 components)
    // Project the difference onto the tangent space at n_current
    // to ensure we're measuring deviation along the manifold
    Eigen::Vector3d n_diff = n_current - normal_prior_;
    Eigen::Matrix3d tangent_projection = Eigen::Matrix3d::Identity() - n_current * n_current.transpose();
    res.head<3>() = weight_normal_ * tangent_projection * n_diff;

    // Distance residual (last component)
    res(3) = weight_distance_ * (d_current - distance_prior_);

    return res;
}

void PlanePriorFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
{
    jacobians.clear();

    if (plane_->is_constant())
    {
        jacobians.emplace_back();  // Empty Jacobian
    }
    else
    {
        Eigen::Vector3d n_current = plane_->unit_vector();

        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(4, 4);

        // First 3 rows: normal residual w.r.t. plane parameters
        // The residual is r = weight * (I - n*n^T) * (n - n_prior)
        // We need to differentiate this accounting for the fact that both
        // the projection matrix and (n - n_prior) depend on n.
        //
        // Expanding: r = weight * [n*(n^T*n_prior) - n_prior]
        // So: dr/dn = weight * [(n^T*n_prior)*I + n*n_prior^T]
        //
        // But apply_increment normalizes, so effective change is (I - n*n^T)*dx
        // Therefore: dr/dx = dr/dn * (I - n*n^T)
        //                  = weight * [(n^T*n_prior)*I + n*n_prior^T] * (I - n*n^T)
        //                  = weight * [(n^T*n_prior)*I + n*n_prior^T - 2*(n^T*n_prior)*n*n^T]

        double n_dot_prior = n_current.dot(normal_prior_);
        Eigen::Matrix3d nn_T = n_current * n_current.transpose();

        J.block<3, 3>(0, 0) = weight_normal_ * (n_dot_prior * Eigen::Matrix3d::Identity() +
                                                 n_current * normal_prior_.transpose() -
                                                 2.0 * n_dot_prior * nn_T);

        // Normal residual doesn't depend on distance
        J.block<3, 1>(0, 3).setZero();

        // Last row: distance residual w.r.t. plane parameters
        // Distance residual doesn't depend on normal
        J.block<1, 3>(3, 0).setZero();

        // Distance residual w.r.t. distance
        J(3, 3) = weight_distance_;

        jacobians.emplace_back(J);
    }
}

std::vector<Variable *> PlanePriorFactor::variables()
{
    return {plane_};
}

std::string PlanePriorFactor::name() const
{
    return "PlanePriorFactor(" + plane_->name() + ")";
}

} // namespace factorama

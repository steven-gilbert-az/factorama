#include "plane_factor.hpp"
#include <cassert>

namespace factorama
{

PlaneFactor::PlaneFactor(int id,
                         Variable* point_var,
                         PlaneVariable* plane_var,
                         double sigma)
    : point_var_(point_var), plane_var_(plane_var), weight_(1.0 / sigma)
{
    id_ = id;
    assert(point_var != nullptr && "point_var cannot be nullptr");
    assert(plane_var != nullptr && "plane_var cannot be nullptr");
    assert(point_var_->size() == 3 && "PlaneFactor: point variable must be 3D");
    assert(sigma > 0.0 && "PlaneFactor: sigma must be greater than zero");
}

Eigen::VectorXd PlaneFactor::compute_residual() const
{
    // Residual = signed distance from point to plane
    // r = n^T * p + d
    // where n is unit normal, p is point, d is distance from origin

    Eigen::Vector3d point_pos = point_var_->value();
    double distance = plane_var_->distance_from_point(point_pos);

    Eigen::VectorXd res(1);
    res(0) = weight_ * distance;
    return res;
}

void PlaneFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
{
    jacobians.clear();

    // Residual: r = weight * (n^T * p + d)
    // where n is plane normal (3D), p is point (3D), d is plane distance

    Eigen::Vector3d point_pos = point_var_->value();
    Eigen::Vector3d normal = plane_var_->unit_vector();

    // Jacobian w.r.t. point (1x3)
    // dr/dp = weight * n^T
    if (point_var_->is_constant())
    {
        jacobians.emplace_back();  // Empty Jacobian
    }
    else
    {
        Eigen::MatrixXd J_point(1, 3);
        J_point = weight_ * normal.transpose();
        jacobians.emplace_back(J_point);
    }

    // Jacobian w.r.t. plane (1x4)
    // Plane value is [nx, ny, nz, d]
    // Since n is a unit vector, we need to project onto the tangent space
    // to avoid trying to grow/shrink it. The tangent space is perpendicular to n.
    // dr/dn_tangent = p^T * (I - n*n^T) = (p - (p^T*n)*n)^T
    // dr/dd = 1
    if (plane_var_->is_constant())
    {
        jacobians.emplace_back();  // Empty Jacobian
    }
    else
    {
        Eigen::MatrixXd J_plane(1, 4);

        // Project point onto tangent space of unit sphere at normal
        double p_dot_n = point_pos.dot(normal);
        Eigen::Vector3d p_tangent = point_pos - p_dot_n * normal;

        J_plane.block<1, 3>(0, 0) = weight_ * p_tangent.transpose();
        J_plane(0, 3) = weight_;
        jacobians.emplace_back(J_plane);
    }
}

std::vector<Variable *> PlaneFactor::variables()
{
    return {point_var_, plane_var_};
}

std::string PlaneFactor::name() const
{
    return "PlaneFactor(" + point_var_->name() + ", " + plane_var_->name() + ")";
}

} // namespace factorama

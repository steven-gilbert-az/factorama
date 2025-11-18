#include "plane_factor.hpp"
#include <cassert>

namespace factorama
{

    PlaneFactor::PlaneFactor(int id,
                             Variable *point_var,
                             PlaneVariable *plane_var,
                             double sigma)
        : point_var_(point_var), plane_var_(plane_var), weight_(1.0 / sigma),

          do_distance_scaling_(false),
          dist_scaling_r0_(1.0),
          dist_scaling_p0_(Eigen::Vector3d::Zero())
    {
        id_ = id;
        assert(point_var != nullptr && "point_var cannot be nullptr");
        assert(plane_var != nullptr && "plane_var cannot be nullptr");
        assert(point_var_->size() == 3 && "PlaneFactor: point variable must be 3D");
        assert(sigma > 0.0 && "PlaneFactor: sigma must be greater than zero");
    }

    PlaneFactor::PlaneFactor(int id,
                             Variable *point_var,
                             PlaneVariable *plane_var,
                             double sigma,
                             bool do_distance_scaling,
                             double dist_scaling_r0,
                             Eigen::Vector3d dist_scaling_p0)
        : point_var_(point_var),
          plane_var_(plane_var),
          weight_(1.0 / sigma),
          do_distance_scaling_(do_distance_scaling),
          dist_scaling_r0_(dist_scaling_r0),
          dist_scaling_p0_(dist_scaling_p0)
    {
        id_ = id;
        assert(point_var != nullptr && "point_var cannot be nullptr");
        assert(plane_var != nullptr && "plane_var cannot be nullptr");
        assert(point_var_->size() == 3 && "PlaneFactor: point variable must be 3D");
        assert(sigma > 0.0 && "PlaneFactor: sigma must be greater than zero");
    }

    Eigen::VectorXd PlaneFactor::compute_residual() const
    {
        if (!do_distance_scaling_)
        {
            // Old residual: simple signed distance from point to plane
            // r = n^T * p + d
            // where n is unit normal, p is point, d is distance from origin

            Eigen::Vector3d point_pos = point_var_->value();
            double distance = plane_var_->distance_from_point(point_pos);

            Eigen::VectorXd res(1);
            res(0) = weight_ * distance;
            return res;
        }
        else
        {
            // New residual with distance scaling
            // Extract values
            Eigen::Vector3d p = point_var_->value();       // 3D point
            Eigen::Vector3d n = plane_var_->unit_vector(); // unit normal
            double d = plane_var_->distance_from_origin(); // plane offset: n·x = d

            // Range term
            Eigen::Vector3d diff = p - dist_scaling_p0_;
            double r = diff.norm();
            double scale = 1.0 + r / dist_scaling_r0_;

            // Signed distance
            double dist = n.dot(p) - d;

            Eigen::VectorXd res(1);
            res(0) = weight_ * (dist / scale);
            return res;
        }
    }

    void PlaneFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
    {
        jacobians.clear();

        if (!do_distance_scaling_)
        {
            // Old Jacobians: without distance scaling
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
        else
        {
            // New Jacobians with distance scaling
            // Extract values
            Eigen::Vector3d p = point_var_->value();
            Eigen::Vector3d n = plane_var_->unit_vector();
            double d = plane_var_->distance_from_origin();

            // Range term
            Eigen::Vector3d diff = p - dist_scaling_p0_;
            double r = diff.norm();
            double scale = 1.0 + r / dist_scaling_r0_;

            // Signed distance
            double dist = n.dot(p) - d;

            // Common partials
            Eigen::Vector3d dscale_dp = Eigen::Vector3d::Zero();
            if (r > 1e-12)
            {
                dscale_dp = (1.0 / dist_scaling_r0_) * (diff / r);
            }

            // dr/dp = weight * [ (n*scale - dist*dscale_dp) / scale^2 ]
            Eigen::Vector3d dr_dp =
                (n * scale - dist * dscale_dp) / (scale * scale);

            // --- Jacobian w.r.t point (1x3) ---
            if (point_var_->is_constant())
            {
                jacobians.emplace_back(); // Empty
            }
            else
            {
                Eigen::MatrixXd J_point(1, 3);
                J_point = weight_ * dr_dp.transpose();
                jacobians.emplace_back(J_point);
            }

            // --- Jacobian w.r.t plane (1x4) ---
            // Plane variables = [nx, ny, nz, d], with n on S2
            //
            // dr/d(n_hat) = weight * [ (p - p0 proj) term but all divided by scale ]
            //
            // Original plane tangent projection preserved:
            // dn_tangent = (I - n*n^T)
            //
            // ∂dist/∂n_tangent = p^T * (I - n*n^T)  (same as your old code)
            //
            // Chain rule: residual = dist/scale, scale independent of n
            //
            // => dr/dn_tangent = (1/scale) * (p_tangent)^T
            //
            // dr/dd = -1/scale

            if (plane_var_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                Eigen::MatrixXd J_plane(1, 4);

                // Tangent component for unit-vector parameterization
                double p_dot_n = p.dot(n);
                Eigen::Vector3d p_tangent = p - p_dot_n * n;

                J_plane.block<1, 3>(0, 0) = weight_ * (p_tangent.transpose() / scale);
                J_plane(0, 3) = weight_ * (-1.0 / scale);

                jacobians.emplace_back(J_plane);
            }
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

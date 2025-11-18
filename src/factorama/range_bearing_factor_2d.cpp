#include "range_bearing_factor_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace factorama
{
    RangeBearingFactor2D::RangeBearingFactor2D(
        int id,
        Pose2DVariable* pose_var,
        Variable* landmark_var,
        double range_obs,
        double bearing_angle_obs,
        double range_sigma,
        double bearing_sigma)
        : pose_var_(pose_var),
          landmark_var_(landmark_var),
          range_obs_(range_obs),
          bearing_angle_obs_(bearing_angle_obs),
          range_weight_(1.0 / range_sigma),
          bearing_weight_(1.0 / bearing_sigma)
    {
        id_ = id;
        assert(pose_var != nullptr && "pose_var cannot be nullptr");
        assert(landmark_var != nullptr && "landmark_var cannot be nullptr");
        assert(landmark_var->size() == 2 && "landmark_var must be 2D");
        assert(range_obs >= 0.0 && "range_obs must be non-negative");
        assert(range_sigma > 0.0 && "range_sigma must be greater than zero");
        assert(bearing_sigma > 0.0 && "bearing_sigma must be greater than zero");
    }

    Eigen::VectorXd RangeBearingFactor2D::compute_residual() const
    {
        // Get pose state
        Eigen::Vector2d pos_pose = pose_var_->pos_2d();

        // Get landmark position
        Eigen::Vector2d pos_landmark = landmark_var_->value().head<2>();

        // Compute delta in world frame
        Eigen::Vector2d delta_world = pos_landmark - pos_pose;

        // Rotate delta into pose frame
        Eigen::Matrix2d R_T = pose_var_->dcm_2d().transpose();
        Eigen::Vector2d delta_local = R_T * delta_world;

        // Compute predicted range
        double range_pred = delta_local.norm();

        // Compute predicted bearing angle in pose frame
        double bearing_pred = std::atan2(delta_local(1), delta_local(0));

        // Compute residuals
        double range_error = range_pred - range_obs_;
        double bearing_error = bearing_pred - bearing_angle_obs_;
        bearing_error = wrap_angle(bearing_error);

        Eigen::VectorXd res(2);
        res(0) = range_weight_ * range_error;
        res(1) = bearing_weight_ * bearing_error;
        return res;
    }

    void RangeBearingFactor2D::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        jacobians.clear();

        // Get pose state
        Eigen::Vector2d pos_pose = pose_var_->pos_2d();
        double theta = pose_var_->theta();
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Get landmark position
        Eigen::Vector2d pos_landmark = landmark_var_->value().head<2>();

        // Compute delta in world and local frames
        Eigen::Vector2d delta_world = pos_landmark - pos_pose;
        Eigen::Matrix2d R_T = pose_var_->dcm_2d().transpose();
        Eigen::Vector2d delta_local = R_T * delta_world;

        double dx_local = delta_local(0);
        double dy_local = delta_local(1);
        double r_sq = dx_local * dx_local + dy_local * dy_local;
        double r = std::sqrt(r_sq);

        // Check for degenerate case (landmark at pose position)
        if (r < MIN_DISTANCE_FROM_POSE)
        {
            throw std::runtime_error("RangeBearingFactor2D: Landmark too close to pose");
        }

        // --- Jacobian w.r.t. pose [x, y, theta] (2x3) ---
        if (pose_var_->is_constant())
        {
            jacobians.emplace_back();  // Empty Jacobian
        }
        else
        {
            Eigen::MatrixXd J_pose(2, 3);

            // First row: derivatives of range w.r.t. pose
            // ∂range/∂px = -(dx_local * cos - dy_local * sin) / r
            // ∂range/∂py = -(dx_local * sin + dy_local * cos) / r
            // ∂range/∂theta = 0
            J_pose(0, 0) = -(dx_local * c - dy_local * s) / r;
            J_pose(0, 1) = -(dx_local * s + dy_local * c) / r;
            J_pose(0, 2) = 0.0;

            // Second row: derivatives of bearing w.r.t. pose
            // ∂bearing/∂px = (dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂py = (dy_local * sin - dx_local * cos) / r^2
            // ∂bearing/∂theta = -1
            J_pose(1, 0) = (dy_local * c + dx_local * s) / r_sq;
            J_pose(1, 1) = (dy_local * s - dx_local * c) / r_sq;
            J_pose(1, 2) = -1.0;

            // Apply weights
            J_pose.row(0) *= range_weight_;
            J_pose.row(1) *= bearing_weight_;

            jacobians.emplace_back(J_pose);
        }

        // --- Jacobian w.r.t. landmark [x, y] (2x2) ---
        if (landmark_var_->is_constant())
        {
            jacobians.emplace_back();  // Empty Jacobian
        }
        else
        {
            Eigen::MatrixXd J_landmark(2, 2);

            // First row: derivatives of range w.r.t. landmark
            // ∂range/∂lx = (dx_local * cos - dy_local * sin) / r
            // ∂range/∂ly = (dx_local * sin + dy_local * cos) / r
            J_landmark(0, 0) = (dx_local * c - dy_local * s) / r;
            J_landmark(0, 1) = (dx_local * s + dy_local * c) / r;

            // Second row: derivatives of bearing w.r.t. landmark
            // ∂bearing/∂lx = -(dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂ly = -(dy_local * sin - dx_local * cos) / r^2
            J_landmark(1, 0) = -(dy_local * c + dx_local * s) / r_sq;
            J_landmark(1, 1) = -(dy_local * s - dx_local * c) / r_sq;

            // Apply weights
            J_landmark.row(0) *= range_weight_;
            J_landmark.row(1) *= bearing_weight_;

            jacobians.emplace_back(J_landmark);
        }
    }

    double RangeBearingFactor2D::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π]
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

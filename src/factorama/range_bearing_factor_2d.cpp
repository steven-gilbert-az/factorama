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

    void RangeBearingFactor2D::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
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

        result.resize(2);
        result(0) = range_weight_ * range_error;
        result(1) = bearing_weight_ * bearing_error;
    }

    void RangeBearingFactor2D::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 2 variables
        if(jacobians.size() == 0) {
            jacobians.resize(2);
        }
        else if(jacobians.size() != 2) {
            jacobians.clear();
            jacobians.resize(2);
        }

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
            jacobians[0] = Eigen::MatrixXd();
        }
        else
        {
            if(jacobians[0].rows() != size_ || jacobians[0].cols() != 3) {
                jacobians[0].resize(size_, 3);
            }

            // First row: derivatives of range w.r.t. pose
            // ∂range/∂px = -(dx_local * cos - dy_local * sin) / r
            // ∂range/∂py = -(dx_local * sin + dy_local * cos) / r
            // ∂range/∂theta = 0
            jacobians[0](0, 0) = -range_weight_ * (dx_local * c - dy_local * s) / r;
            jacobians[0](0, 1) = -range_weight_ * (dx_local * s + dy_local * c) / r;
            jacobians[0](0, 2) = 0.0;

            // Second row: derivatives of bearing w.r.t. pose
            // ∂bearing/∂px = (dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂py = (dy_local * sin - dx_local * cos) / r^2
            // ∂bearing/∂theta = -1
            jacobians[0](1, 0) = bearing_weight_ * (dy_local * c + dx_local * s) / r_sq;
            jacobians[0](1, 1) = bearing_weight_ * (dy_local * s - dx_local * c) / r_sq;
            jacobians[0](1, 2) = -bearing_weight_;
        }

        // --- Jacobian w.r.t. landmark [x, y] (2x2) ---
        if (landmark_var_->is_constant())
        {
            jacobians[1] = Eigen::MatrixXd();
        }
        else
        {
            if(jacobians[1].rows() != size_ || jacobians[1].cols() != 2) {
                jacobians[1].resize(size_, 2);
            }

            // First row: derivatives of range w.r.t. landmark
            // ∂range/∂lx = (dx_local * cos - dy_local * sin) / r
            // ∂range/∂ly = (dx_local * sin + dy_local * cos) / r
            jacobians[1](0, 0) = range_weight_ * (dx_local * c - dy_local * s) / r;
            jacobians[1](0, 1) = range_weight_ * (dx_local * s + dy_local * c) / r;

            // Second row: derivatives of bearing w.r.t. landmark
            // ∂bearing/∂lx = -(dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂ly = -(dy_local * sin - dx_local * cos) / r^2
            jacobians[1](1, 0) = -bearing_weight_ * (dy_local * c + dx_local * s) / r_sq;
            jacobians[1](1, 1) = -bearing_weight_ * (dy_local * s - dx_local * c) / r_sq;
        }
    }

    double RangeBearingFactor2D::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π]
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

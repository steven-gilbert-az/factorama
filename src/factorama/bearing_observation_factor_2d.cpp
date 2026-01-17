#include "bearing_observation_factor_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace factorama
{
    BearingObservationFactor2D::BearingObservationFactor2D(int id, Pose2DVariable *pose_var, Variable *landmark_var,
                                                           double bearing_angle_obs, double angle_sigma)
        : pose_var_(pose_var)
        , landmark_var_(landmark_var)
        , bearing_angle_obs_(bearing_angle_obs)
        , weight_(1.0 / angle_sigma)
    {
        id_ = id;
        assert(pose_var != nullptr && "pose_var cannot be nullptr");
        assert(landmark_var != nullptr && "landmark_var cannot be nullptr");
        assert(landmark_var->size() == 2 && "landmark_var must be 2D");
        assert(angle_sigma > 0.0 && "angle_sigma must be greater than zero");
    }

    Eigen::VectorXd BearingObservationFactor2D::compute_residual() const
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

        // Compute bearing angle in pose frame
        double bearing_pred = std::atan2(delta_local(1), delta_local(0));

        // Compute angular residual with wrapping
        double angular_error = bearing_pred - bearing_angle_obs_;
        angular_error = wrap_angle(angular_error);

        Eigen::VectorXd res(1);
        res(0) = weight_ * angular_error;
        return res;
    }

    void BearingObservationFactor2D::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
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

        // Compute bearing angle in pose frame
        double bearing_pred = std::atan2(delta_local(1), delta_local(0));

        // Compute angular residual with wrapping
        double angular_error = bearing_pred - bearing_angle_obs_;
        angular_error = wrap_angle(angular_error);

        result.resize(1);
        result(0) = weight_ * angular_error;
    }

    void BearingObservationFactor2D::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 2 variables
        if (jacobians.size() == 0) {
            jacobians.resize(2);
        } else if (jacobians.size() != 2) {
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

        // Check for degenerate case (landmark at pose position)
        if (r_sq < MIN_DISTANCE_FROM_POSE * MIN_DISTANCE_FROM_POSE) {
            throw std::runtime_error("BearingObservationFactor2D: Landmark too close to pose");
        }

        // --- Jacobian w.r.t. pose [x, y, theta] (1x3) ---
        if (pose_var_->is_constant()) {
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != 3) {
                jacobians[0].resize(size_, 3);
            }

            // Derivatives of bearing angle w.r.t. pose position
            // ∂bearing/∂px = (dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂py = (dy_local * sin - dx_local * cos) / r^2
            jacobians[0](0, 0) = weight_ * (dy_local * c + dx_local * s) / r_sq;
            jacobians[0](0, 1) = weight_ * (dy_local * s - dx_local * c) / r_sq;

            // Derivative w.r.t. pose orientation
            // ∂bearing/∂theta = -1
            jacobians[0](0, 2) = -weight_;
        }

        // --- Jacobian w.r.t. landmark [x, y] (1x2) ---
        if (landmark_var_->is_constant()) {
            jacobians[1] = Eigen::MatrixXd();
        } else {
            if (jacobians[1].rows() != size_ || jacobians[1].cols() != 2) {
                jacobians[1].resize(size_, 2);
            }

            // Derivatives of bearing angle w.r.t. landmark position
            // ∂bearing/∂lx = -(dy_local * cos + dx_local * sin) / r^2
            // ∂bearing/∂ly = -(dy_local * sin - dx_local * cos) / r^2
            jacobians[1](0, 0) = -weight_ * (dy_local * c + dx_local * s) / r_sq;
            jacobians[1](0, 1) = -weight_ * (dy_local * s - dx_local * c) / r_sq;
        }
    }

    double BearingObservationFactor2D::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π]
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

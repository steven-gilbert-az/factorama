#include "pose_2d_prior_factor.hpp"
#include <cmath>

namespace factorama
{
    Pose2DPriorFactor::Pose2DPriorFactor(int id, Pose2DVariable *pose_var, const Eigen::Vector3d& pose_prior,
                                         double position_sigma, double angle_sigma)
        : pose_var_(pose_var)
        , pose_prior_(pose_prior)
        , position_weight_(1.0 / position_sigma)
        , angle_weight_(1.0 / angle_sigma)
    {
        id_ = id;
        assert(pose_var != nullptr && "pose_var cannot be nullptr");
        assert(position_sigma > 0.0 && "position_sigma must be greater than zero");
        assert(angle_sigma > 0.0 && "angle_sigma must be greater than zero");
    }

    Eigen::VectorXd Pose2DPriorFactor::compute_residual() const
    {
        Eigen::VectorXd res(3);

        // Position residual (weighted)
        Eigen::Vector2d pos_error = pose_var_->pos_2d() - pose_prior_.head<2>();
        res.head<2>() = position_weight_ * pos_error;

        // Angle residual (weighted and wrapped to handle ±π discontinuity)
        double angle_error = pose_var_->theta() - pose_prior_(2);
        angle_error = wrap_angle(angle_error); // Critical: wrap to [-π, π]
        res(2) = angle_weight_ * angle_error;

        return res;
    }

    void Pose2DPriorFactor::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
    {
        result.resize(3);

        // Position residual (weighted)
        Eigen::Vector2d pos_error = pose_var_->pos_2d() - pose_prior_.head<2>();
        result.head<2>() = position_weight_ * pos_error;

        // Angle residual (weighted and wrapped to handle ±π discontinuity)
        double angle_error = pose_var_->theta() - pose_prior_(2);
        angle_error = wrap_angle(angle_error); // Critical: wrap to [-π, π]
        result(2) = angle_weight_ * angle_error;
    }

    void Pose2DPriorFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 1 variable
        if (jacobians.size() == 0) {
            jacobians.resize(1);
        } else if (jacobians.size() != 1) {
            jacobians.clear();
            jacobians.resize(1);
        }

        if (pose_var_->is_constant()) {
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != size_) {
                jacobians[0].resize(size_, size_);
            }
            jacobians[0].setZero();

            // Position part (2x2 identity block, weighted)
            jacobians[0].block<2, 2>(0, 0).diagonal().array() = position_weight_;

            // Angle part (1x1 identity, weighted)
            jacobians[0](2, 2) = angle_weight_;
        }
    }

    double Pose2DPriorFactor::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π]
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

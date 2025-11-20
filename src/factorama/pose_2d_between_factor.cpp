#include "pose_2d_between_factor.hpp"
#include <cmath>

namespace factorama
{
    Pose2DBetweenFactor::Pose2DBetweenFactor(int id,
                                             Pose2DVariable* pose_a,
                                             Pose2DVariable* pose_b,
                                             Variable* measured_between_variable,
                                             double position_sigma,
                                             double angle_sigma,
                                             bool local_frame)
        : pose_a_(pose_a),
          pose_b_(pose_b),
          measured_between_variable_(measured_between_variable),
          position_weight_(1.0 / position_sigma),
          angle_weight_(1.0 / angle_sigma),
          local_frame_(local_frame)
    {
        id_ = id;
        assert(pose_a != nullptr && "pose_a cannot be nullptr");
        assert(pose_b != nullptr && "pose_b cannot be nullptr");
        assert(measured_between_variable != nullptr && "measured_between_variable cannot be nullptr");
        assert(measured_between_variable->size() == 3 &&
               "measured_between_variable must be 3-dimensional [dx, dy, dtheta]");
        assert(position_sigma > 0.0 && "position_sigma must be greater than zero");
        assert(angle_sigma > 0.0 && "angle_sigma must be greater than zero");
    }

    Eigen::VectorXd Pose2DBetweenFactor::compute_residual() const
    {
        Eigen::VectorXd res(3);

        // Position residual
        Eigen::Vector2d relative_pos_W = pose_b_->pos_2d() - pose_a_->pos_2d();
        Eigen::Vector2d measured_pos = measured_between_variable_->value().head<2>();
        Eigen::Vector2d pos_error;

        if (local_frame_)
        {
            // Measurement in pose_a's local frame: transform world difference to pose_a's frame
            Eigen::Matrix2d dcm_AW = pose_a_->dcm_2d();
            Eigen::Vector2d relative_pos_A = dcm_AW * relative_pos_W;
            pos_error = relative_pos_A - measured_pos;
        }
        else
        {
            // Measurement in world frame (original behavior)
            pos_error = relative_pos_W - measured_pos;
        }
        res.head<2>() = position_weight_ * pos_error;

        // Angle residual (wrapped to handle ±π discontinuity) - same for both modes
        double relative_angle = pose_b_->theta() - pose_a_->theta();
        double measured_angle = measured_between_variable_->value()(2);
        double angle_error = relative_angle - measured_angle;
        angle_error = wrap_angle(angle_error);  // Critical: wrap to [-π, π]
        res(2) = angle_weight_ * angle_error;

        return res;
    }

    void Pose2DBetweenFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        jacobians.clear();

        if (local_frame_)
        {
            // Local frame: residual_pos = dcm_AW * (pos_b - pos_a) - measured_pos
            Eigen::Vector2d relative_pos_W = pose_b_->pos_2d() - pose_a_->pos_2d();
            Eigen::Matrix2d dcm_AW = pose_a_->dcm_2d();

            // Jacobian w.r.t. pose_a
            if (pose_a_->is_constant())
            {
                jacobians.emplace_back();  // Empty Jacobian
            }
            else
            {
                Eigen::MatrixXd J_a = Eigen::MatrixXd::Zero(3, 3);

                // Position part: d(dcm_AW * relative_pos_W)/d(pos_a) = -dcm_AW
                J_a.block<2, 2>(0, 0) = -position_weight_ * dcm_AW;

                // Rotation part: d(dcm_AW * relative_pos_W)/d(theta_a)
                // For 2D: dR/dtheta = J_perp * R where J_perp = [[0, -1], [1, 0]]
                // So d(R * v)/d(theta) = J_perp * R * v
                Eigen::Matrix2d J_perp;
                J_perp << 0, -1,
                          1,  0;
                Eigen::Vector2d dres_dtheta = J_perp * dcm_AW * relative_pos_W;
                J_a.block<2, 1>(0, 2) = position_weight_ * dres_dtheta;

                // Angle part: d(res_angle)/d(theta_a) = -angle_weight
                J_a(2, 2) = -angle_weight_;

                jacobians.emplace_back(J_a);
            }

            // Jacobian w.r.t. pose_b
            if (pose_b_->is_constant())
            {
                jacobians.emplace_back();  // Empty Jacobian
            }
            else
            {
                Eigen::MatrixXd J_b = Eigen::MatrixXd::Zero(3, 3);

                // Position part: d(dcm_AW * relative_pos_W)/d(pos_b) = dcm_AW
                J_b.block<2, 2>(0, 0) = position_weight_ * dcm_AW;

                // Rotation part: no dependency on pose_b's rotation for position
                // (already captured in pose_a's rotation derivative)

                // Angle part: d(res_angle)/d(theta_b) = angle_weight
                J_b(2, 2) = angle_weight_;

                jacobians.emplace_back(J_b);
            }
        }
        else
        {
            // World frame: residual = (pos_b - pos_a) - measured (original behavior)
            // Jacobian w.r.t. pose_a
            if (pose_a_->is_constant())
            {
                jacobians.emplace_back();  // Empty Jacobian
            }
            else
            {
                Eigen::MatrixXd J_a = Eigen::MatrixXd::Zero(3, 3);

                // Position part: d(res_pos)/d(pose_a) = -position_weight * I
                J_a.block<2, 2>(0, 0) = -position_weight_ * Eigen::Matrix2d::Identity();

                // Angle part: d(res_angle)/d(theta_a) = -angle_weight
                J_a(2, 2) = -angle_weight_;

                jacobians.emplace_back(J_a);
            }

            // Jacobian w.r.t. pose_b
            if (pose_b_->is_constant())
            {
                jacobians.emplace_back();  // Empty Jacobian
            }
            else
            {
                Eigen::MatrixXd J_b = Eigen::MatrixXd::Zero(3, 3);

                // Position part: d(res_pos)/d(pose_b) = position_weight * I
                J_b.block<2, 2>(0, 0) = position_weight_ * Eigen::Matrix2d::Identity();

                // Angle part: d(res_angle)/d(theta_b) = angle_weight
                J_b(2, 2) = angle_weight_;

                jacobians.emplace_back(J_b);
            }
        }

        // Jacobian w.r.t. measured_between_variable (same for both modes)
        if (measured_between_variable_->is_constant())
        {
            jacobians.emplace_back();  // Empty Jacobian
        }
        else
        {
            Eigen::MatrixXd J_meas = Eigen::MatrixXd::Zero(3, 3);

            // Position part: d(res_pos)/d(measured_pos) = -position_weight * I
            J_meas.block<2, 2>(0, 0) = -position_weight_ * Eigen::Matrix2d::Identity();

            // Angle part: d(res_angle)/d(measured_angle) = -angle_weight
            J_meas(2, 2) = -angle_weight_;

            jacobians.emplace_back(J_meas);
        }
    }

    double Pose2DBetweenFactor::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π]
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

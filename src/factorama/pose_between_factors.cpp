#include "factorama/pose_between_factors.hpp"
#include "factorama/rotation_variable.hpp"

namespace factorama
{
    void PoseOrientationBetweenFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
    {
        jacobians.clear();

        if (do_so3_nudge_)
        {
            // Use analytical jacobians for SO(3) manifold
            Eigen::Matrix3d dcm_P1W = pose1_->dcm_CW();
            Eigen::Matrix3d dcm_P2W = pose2_->dcm_CW();
            Eigen::Matrix3d dcm_P1_P2 = calibration_rotation_12_->rotation();
            Eigen::Matrix3d error = dcm_P2W * dcm_P1W.transpose() * dcm_P1_P2;
            Eigen::Vector3d residual_vec = LogMapSO3(error);
            
            // Compute inverse right Jacobian of the residual
            Eigen::Matrix3d Jr_inv = compute_inverse_right_jacobian_so3(residual_vec);


            // Jacobian w.r.t. pose1 rotation (only rotation part matters for this factor)
            if (pose1_->is_constant()) {
                jacobians.emplace_back();
            } else {
                Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(3, 6);
                J1.block<3, 3>(0, 3) = -weight_ * Jr_inv.transpose() * dcm_P1_P2.transpose();
                jacobians.emplace_back(J1);
            }

            // Jacobian w.r.t. pose2 rotation
            if (pose2_->is_constant()) {
                jacobians.emplace_back();
            } else {
                Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(3, 6);
                Eigen::Matrix3d J_pose2 = Jr_inv.transpose() * error.transpose();
                J2.block<3, 3>(0, 3) = weight_ * J_pose2;
                jacobians.emplace_back(J2);
            }

            // Jacobian w.r.t. calibration rotation
            if (calibration_rotation_12_->is_constant()) {
                jacobians.emplace_back();
            } else {
                Eigen::Matrix3d J_cal =  Jr_inv.transpose() * dcm_P1_P2.transpose();
                jacobians.emplace_back(weight_ * J_cal);
            }
        }
        else
        {
            // Fall back to numerical jacobians for non-manifold mode
            const double epsilon = 1e-6;
            Eigen::VectorXd r0 = compute_residual();

            if(pose1_->is_constant()) {
                jacobians.push_back(Eigen::MatrixXd());
            }
            else {
                Eigen::MatrixXd J;
                Eigen::VectorXd pose_value = pose1_->value();
                PoseVariable pose1_copy = *pose1_;
                J = Eigen::MatrixXd(residual_size(), pose1_->size());
                for (int i = 0; i < pose1_->size(); ++i)
                {
                    Eigen::VectorXd dx = Eigen::VectorXd::Zero(pose1_->size());
                    dx(i) = epsilon;
                    pose1_copy.set_value_from_vector(pose_value + dx);

                    Eigen::VectorXd r_plus = compute_residual(&pose1_copy, pose2_.get(), calibration_rotation_12_.get());
                    auto diff = r_plus - r0;
                    J.col(i) = diff / epsilon;
                }
                jacobians.push_back(J);
            }

            if(pose2_->is_constant()) {
                jacobians.push_back(Eigen::MatrixXd());
            }
            else {
                Eigen::MatrixXd J;
                Eigen::VectorXd pose_value = pose2_->value();
                PoseVariable pose2_copy = *pose2_;
                J = Eigen::MatrixXd(residual_size(), pose2_->size());
                for (int i = 0; i < pose2_->size(); ++i)
                {
                    Eigen::VectorXd dx = Eigen::VectorXd::Zero(pose2_->size());
                    dx(i) = epsilon;
                    pose2_copy.set_value_from_vector(pose_value + dx);

                    Eigen::VectorXd r_plus = compute_residual(pose1_.get(), &pose2_copy, calibration_rotation_12_.get());
                    auto diff = r_plus - r0;
                    J.col(i) = diff / epsilon;
                }
                jacobians.push_back(J);
            }

            if(calibration_rotation_12_->is_constant()) {
                jacobians.push_back(Eigen::MatrixXd());
            }
            else {
                Eigen::MatrixXd J;
                Eigen::VectorXd cal_value = calibration_rotation_12_->value();
                RotationVariable cal_copy = *calibration_rotation_12_;
                J = Eigen::MatrixXd(residual_size(), calibration_rotation_12_->size());
                for (int i = 0; i < calibration_rotation_12_->size(); ++i)
                {
                    Eigen::VectorXd dx = Eigen::VectorXd::Zero(calibration_rotation_12_->size());
                    dx(i) = epsilon;
                    Eigen::VectorXd new_value = cal_value + dx;
                    cal_copy.set_value_from_vector(new_value);

                    Eigen::VectorXd r_plus = compute_residual(pose1_.get(), pose2_.get(), &cal_copy);
                    auto diff = r_plus - r0;
                    J.col(i) = diff / epsilon;
                }
                jacobians.push_back(J);
            }
        }
    }

} // namespace factorama
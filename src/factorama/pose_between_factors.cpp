#include "factorama/pose_between_factors.hpp"
#include "factorama/rotation_variable.hpp"

namespace factorama
{
    void PoseOrientationBetweenFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
    {
        // Ensure jacobians vector has correct size for 3 variables
        if(jacobians.size() == 0) {
            jacobians.resize(3);
        }
        else if(jacobians.size() != 3) {
            jacobians.clear();
            jacobians.resize(3);
        }

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
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if(jacobians[0].rows() != size_ || jacobians[0].cols() != 6) {
                jacobians[0].resize(size_, 6);
            }
            jacobians[0].setZero();
            jacobians[0].block<3, 3>(0, 3) = -weight_ * Jr_inv.transpose() * dcm_P1_P2.transpose();
        }

        // Jacobian w.r.t. pose2 rotation
        if (pose2_->is_constant()) {
            jacobians[1] = Eigen::MatrixXd();
        } else {
            if(jacobians[1].rows() != size_ || jacobians[1].cols() != 6) {
                jacobians[1].resize(size_, 6);
            }
            jacobians[1].setZero();
            Eigen::Matrix3d J_pose2 = Jr_inv.transpose() * error.transpose();
            jacobians[1].block<3, 3>(0, 3) = weight_ * J_pose2;
        }

        // Jacobian w.r.t. calibration rotation
        if (calibration_rotation_12_->is_constant()) {
            jacobians[2] = Eigen::MatrixXd();
        } else {
            if(jacobians[2].rows() != size_ || jacobians[2].cols() != 3) {
                jacobians[2].resize(size_, 3);
            }
            jacobians[2] = weight_ * Jr_inv.transpose() * dcm_P1_P2.transpose();
        }
    }

} // namespace factorama
#include "factorama/pose_between_factors.hpp"
#include "factorama/rotation_variable.hpp"

namespace factorama
{
    void PoseOrientationBetweenFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
    {
        jacobians.clear();

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

} // namespace factorama
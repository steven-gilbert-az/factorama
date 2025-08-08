#include "factorama/bearing_observation_factor.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{

    void BearingObservationFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians_out) const
    {
        const Eigen::Matrix3d &dcm_CW = pose_var_->dcm_CW();
        const Eigen::Vector3d &t_CW = pose_var_->pos_W();
        const Eigen::Vector3d landmark_W = landmark_var_->pos_W();

        Eigen::Vector3d pos_C = dcm_CW * (landmark_W - t_CW);
        double pos_C_norm = pos_C.norm();
        
        if (pos_C_norm < MIN_DISTANCE_FROM_CAMERA) {
            throw std::runtime_error("BearingObservationFactor: Landmark too close to camera");
        }
        
        Eigen::Vector3d bearing_C = pos_C / pos_C_norm;

        // Derivative of normalized vector
        Eigen::Matrix3d d_bearing_d_pos = (Eigen::Matrix3d::Identity() - bearing_C * bearing_C.transpose()) / pos_C_norm;

        // Jacobian w.r.t. landmark (position only)
        Eigen::MatrixXd J_landmark;
        if(!landmark_var_->is_constant())
        {
            J_landmark = weight_ * d_bearing_d_pos * dcm_CW;
        }

        // Jacobian w.r.t. pose (translation + rotation)
        Eigen::MatrixXd J_pose;
        if(!pose_var_->is_constant())
        {
            J_pose = Eigen::MatrixXd::Zero(3, 6);
            J_pose.block<3, 3>(0, 0) = -weight_ * d_bearing_d_pos * dcm_CW;

            // Rotation part — depends on do_so3_nudge mode
            if (do_so3_nudge_)
            {
                Eigen::Matrix3d skew = -skew_symmetric(pos_C);
                J_pose.block<3, 3>(0, 3) = weight_ * d_bearing_d_pos * skew;
            }
            else
            {
                // Use simple skew-symmetric jacobian (original approach)
                Eigen::Matrix3d skew = -dcm_CW * skew_symmetric(landmark_W - t_CW);
                J_pose.block<3, 3>(0, 3) = weight_ * d_bearing_d_pos * skew;
            }
        }

        jacobians_out.clear();
        jacobians_out.push_back(J_pose);
        jacobians_out.push_back(J_landmark);
    }

}
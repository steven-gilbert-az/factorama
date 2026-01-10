#include "factorama/bearing_observation_factor.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{

    void BearingObservationFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians_out) const
    {
        // Ensure jacobians vector has correct size for 2 variables
        if(jacobians_out.size() == 0) {
            jacobians_out.resize(2);
        }
        else if(jacobians_out.size() != 2) {
            jacobians_out.clear();
            jacobians_out.resize(2);
        }

        const Eigen::Matrix3d &dcm_CW = pose_var_->dcm_CW();
        const Eigen::Vector3d &pos_W_cam = pose_var_->pos_W();
        const Eigen::Vector3d landmark_W = landmark_var_->pos_W();

        Eigen::Vector3d pos_C = dcm_CW * (landmark_W - pos_W_cam);
        double pos_C_norm = pos_C.norm();

        if (pos_C_norm < MIN_DISTANCE_FROM_CAMERA) {
            throw std::runtime_error("BearingObservationFactor: Landmark too close to camera");
        }

        Eigen::Vector3d bearing_C = pos_C / pos_C_norm;

        // Derivative of normalized vector
        Eigen::Matrix3d d_bearing_d_pos;
        d_bearing_d_pos.noalias() =
            (-bearing_C * bearing_C.transpose());
        d_bearing_d_pos.diagonal().array() += 1.0;

        // Scale d_bearing_d_pos by weight
        d_bearing_d_pos *= (weight_ / pos_C_norm);

        // Jacobian w.r.t. pose (translation + rotation)
        if(pose_var_->is_constant())
        {
            jacobians_out[0] = Eigen::MatrixXd();
        }
        else
        {
            if(jacobians_out[0].rows() != size_ || jacobians_out[0].cols() != 6) {
                jacobians_out[0].resize(size_, 6);
            }
            jacobians_out[0].setZero();
            jacobians_out[0].block<3, 3>(0, 0).noalias() = -1.0 * d_bearing_d_pos * dcm_CW;

            Eigen::Matrix3d skew = -skew_symmetric(pos_C);
            jacobians_out[0].block<3, 3>(0, 3).noalias() = d_bearing_d_pos * skew;
        }

        // Jacobian w.r.t. landmark (position only)
        if(landmark_var_->is_constant())
        {
            jacobians_out[1] = Eigen::MatrixXd();
        }
        else
        {
            if(jacobians_out[1].rows() != size_ || jacobians_out[1].cols() != 3) {
                jacobians_out[1].resize(size_, 3);
            }
            jacobians_out[1].noalias() = d_bearing_d_pos * dcm_CW;
        }
    }

}
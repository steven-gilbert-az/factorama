#include "factorama/inverse_range_bearing_factor.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{

    void InverseRangeBearingFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians_out) const
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
        const Eigen::Vector3d &pos_C_W = pose_var_->pos_W();
        const Eigen::Vector3d landmark_W = inverse_range_var_->pos_W();
        const Eigen::Vector3d delta = landmark_W - pos_C_W;

        Eigen::Vector3d pos_C = dcm_CW * delta;
        double pos_C_norm = pos_C.norm();

        if (pos_C_norm < MIN_DISTANCE_FROM_CAMERA) {
            throw std::runtime_error("InverseRangeBearingFactor: Landmark too close to camera");
        }

        Eigen::Vector3d bearing_C = pos_C / pos_C_norm;

        Eigen::Matrix3d d_bearing_d_pos = (Eigen::Matrix3d::Identity() - bearing_C * bearing_C.transpose()) / pos_C_norm;

        // Jacobian w.r.t. pose
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
            jacobians_out[0].block<3, 3>(0, 0) = -weight_ * d_bearing_d_pos * dcm_CW;

            // Rotation part
            Eigen::Matrix3d skew = -skew_symmetric(pos_C);
            jacobians_out[0].block<3, 3>(0, 3) = weight_ * d_bearing_d_pos * skew;
        }

        // Jacobian w.r.t. inverse range
        double inv_range = inverse_range_var_->inverse_range();

        if (std::abs(inv_range) < MIN_INVERSE_RANGE) {
            throw std::runtime_error("InverseRangeBearingFactor: Inverse range too close to zero");
        }

        Eigen::Vector3d dir = inverse_range_var_->bearing_W();
        Eigen::Vector3d d_posW_d_inv = -dir / (inv_range * inv_range); // d(pos_W) / d(inv_range)
        Eigen::Vector3d d_posC_d_inv = dcm_CW * d_posW_d_inv;

        if(inverse_range_var_->is_constant())
        {
            jacobians_out[1] = Eigen::MatrixXd();
        }
        else
        {
            if(jacobians_out[1].rows() != size_ || jacobians_out[1].cols() != 1) {
                jacobians_out[1].resize(size_, 1);
            }
            jacobians_out[1] = weight_ * d_bearing_d_pos * d_posC_d_inv;
        }
    }


}
#include "factorama/bearing_projection_factor_2d.hpp"
#include <iostream>

namespace factorama
{
    void BearingProjectionFactor2D::compute_tangent_basis()
    {
        // Build orthonormal basis T spanning k_perp (3x2 matrix)
        // Strategy: pick axis most orthogonal to k, take cross products, normalize
        
        // Find the coordinate axis most orthogonal to k
        Eigen::Vector3d k_abs = bearing_C_observed_.cwiseAbs();
        Eigen::Vector3d u1;
        
        if (k_abs.x() <= k_abs.y() && k_abs.x() <= k_abs.z()) {
            u1 = Eigen::Vector3d::UnitX();
        } else if (k_abs.y() <= k_abs.z()) {
            u1 = Eigen::Vector3d::UnitY();
        } else {
            u1 = Eigen::Vector3d::UnitZ();
        }
        
        // First tangent vector: u1 - (k^T u1) k, then normalize
        Eigen::Vector3d t1 = u1 - bearing_C_observed_.dot(u1) * bearing_C_observed_;
        t1.normalize();
        
        // Second tangent vector: k x t1 (already normalized since k and t1 are orthonormal)
        Eigen::Vector3d t2 = bearing_C_observed_.cross(t1);
        
        // Pack into T_ matrix
        T_.col(0) = t1;
        T_.col(1) = t2;
    }

    Eigen::VectorXd BearingProjectionFactor2D::compute_residual() const
    {
        // Get pose and landmark data
        Eigen::Vector3d pos_W = landmark_->pos_W();
        Eigen::Vector3d pos_W_cam = pose_->pos_W();
        Eigen::Matrix3d dcm_CW = pose_->dcm_CW();
       
        // Compute camera-frame point: y = dcm_CW * (X - t_CW)
        // (unit vector)
        Eigen::Vector3d bearing_C = (dcm_CW * (pos_W - pos_W_cam)).normalized();
        
        // Compute along-ray depth
        double depth = bearing_C_observed_.dot(bearing_C);
        
        // Handle numerical edge case
        if (depth <= reverse_depth_tolerance_) {
            // Down-weight by returning large residual with effectively zero Jacobian
            Eigen::Vector2d large_residual = Eigen::Vector2d::Constant(1.0);
            std::cout << "Warning: BearingProjectionFactor2D depth=" << depth 
                      << " <= eps=" << reverse_depth_tolerance_ << ", down-weighting factor" << std::endl;
            return weight_ * large_residual;
        }
        
        // Compute residual: r = (T^T y)
        Eigen::Vector2d r = T_.transpose() * bearing_C;
        
        return weight_ * r;
    }

    void BearingProjectionFactor2D::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        jacobians.clear();
        
        // Get pose and landmark data
        Eigen::Vector3d pos_W = landmark_->pos_W();
        Eigen::Vector3d pos_W_cam = pose_->pos_W();
        Eigen::Matrix3d dcm_CW = pose_->dcm_CW();
       
        // Compute camera-frame point: y = dcm_CW * (X - t_CW)
        // (unit vector)
        Eigen::Vector3d pos_C = dcm_CW * (pos_W - pos_W_cam);
        double pos_C_norm = pos_C.norm();
        Eigen::Vector3d bearing_C = pos_C / pos_C_norm;
        
        // Compute along-ray depth
        double depth = bearing_C_observed_.dot(bearing_C);
        
        // Handle edge case: if depth too small, zero out Jacobians to avoid reversing
        if (depth <= reverse_depth_tolerance_) {
            if (pose_->is_constant()) {
                jacobians.emplace_back(); // Empty Jacobian for constant pose
            } else {
                jacobians.emplace_back(Eigen::MatrixXd::Zero(2, 6));
            }
            
            if (landmark_->is_constant()) {
                jacobians.emplace_back(); // Empty Jacobian for constant landmark
            } else {
                jacobians.emplace_back(Eigen::MatrixXd::Zero(2, 3));
            }
            return;
        }
        
        // Derivative of normalized vector
        Eigen::Matrix3d d_bearing_d_pos = (Eigen::Matrix3d::Identity() - bearing_C * bearing_C.transpose()) / pos_C_norm;


        // Compute Jacobian w.r.t. camera-frame point y (2x3):
        // dr/dy = (1/alpha) * T^T - (1/alpha^2) * (T^T y) * k^T
        Eigen::Matrix<double, 2, 3> J_y = T_.transpose();

        
        // Jacobian w.r.t. pose (2x6):
        // dy/dxi = [-[y]_x  -I_3] for right-perturbation convention
        if (pose_->is_constant()) {
            jacobians.emplace_back(); // Empty Jacobian for constant pose
        } else {
            Eigen::Matrix<double, 3, 6> dy_dpose;
            dy_dpose.block<3, 3>(0, 0) = -weight_ * d_bearing_d_pos * dcm_CW;  // -[y]_x for rotation part
            Eigen::Matrix3d skew = -skew_symmetric(pos_C);
            dy_dpose.block<3, 3>(0, 3) = weight_ * d_bearing_d_pos * skew;  // -I for translation part
            
            Eigen::Matrix<double, 2, 6> J_pose = J_y * dy_dpose;
            jacobians.emplace_back(J_pose);
        }
        
        // Jacobian w.r.t. landmark (2x3):
        // dy/dX = dcm_CW
        if (landmark_->is_constant()) {
            jacobians.emplace_back(); // Empty Jacobian for constant landmark
        } else {
            // Eigen::Matrix<double, 2, 3> J_landmark = J_y * dcm_CW;
            // jacobians.emplace_back(J_landmark);
            Eigen::Matrix3d dy_dlandmark = weight_ * d_bearing_d_pos * dcm_CW;
            Eigen::Matrix<double, 2, 3> J_landmark = J_y * dy_dlandmark;
            jacobians.emplace_back(J_landmark);
        }
    }
}
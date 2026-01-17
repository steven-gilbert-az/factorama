#include "pose_prior_factors.hpp"
#include <cassert>

namespace factorama
{

    // PosePositionPriorFactor implementations
    PosePositionPriorFactor::PosePositionPriorFactor(int id, PoseVariable *pose, const Eigen::Vector3d& pos_prior,
                                                     double sigma)
        : pose_(pose)
        , pos_prior_(pos_prior)
        , weight_(1.0 / sigma)
    {
        id_ = id;
        assert(pose != nullptr && "pose cannot be nullptr");
        assert(sigma > 0.0 && "Sigma must be greater than zero");
    }

    Eigen::VectorXd PosePositionPriorFactor::compute_residual() const
    {
        Eigen::Vector3d res = pose_->pos_W() - pos_prior_;
        return weight_ * res;
    }

    void PosePositionPriorFactor::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
    {
        result = weight_ * (pose_->pos_W() - pos_prior_);
    }

    void PosePositionPriorFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 1 variable
        if (jacobians.size() == 0) {
            jacobians.resize(1);
        } else if (jacobians.size() != 1) {
            jacobians.clear();
            jacobians.resize(1);
        }

        if (pose_->is_constant()) {
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != 6) {
                jacobians[0].resize(size_, 6);
            }
            jacobians[0].setZero();
            jacobians[0].block<3, 3>(0, 0).diagonal().array() = weight_;
        }
    }

    std::vector<Variable *> PosePositionPriorFactor::variables()
    {
        return {pose_};
    }

    std::string PosePositionPriorFactor::name() const
    {
        return "PosePositionPriorFactor(" + pose_->name() + ")";
    }

    // PoseOrientationPriorFactor implementations
    PoseOrientationPriorFactor::PoseOrientationPriorFactor(int id, PoseVariable *pose,
                                                           const Eigen::Matrix3d& dcm_CW_prior, double sigma)
        : pose_(pose)
        , weight_(1.0 / sigma)
    {
        id_ = id;
        assert(pose != nullptr && "pose cannot be nullptr");
        assert(sigma > 0.0 && "Sigma must be greater than zero");

        rot_CW_prior_ = LogMapSO3(dcm_CW_prior);
    }

    Eigen::VectorXd PoseOrientationPriorFactor::compute_residual() const
    {
        // Use full SO(3) manifold approach
        // For a prior factor: r = log(dcm_current * dcm_prior^T)
        Eigen::Matrix3d dcm_CW_current = pose_->dcm_CW();
        Eigen::Matrix3d dcm_CW_prior = ExpMapSO3(rot_CW_prior_);
        Eigen::Matrix3d dcm_error = dcm_CW_current * dcm_CW_prior.transpose();
        Eigen::Vector3d res = LogMapSO3(dcm_error);

        return weight_ * res;
    }

    void PoseOrientationPriorFactor::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
    {
        // Use full SO(3) manifold approach
        // For a prior factor: r = log(dcm_current * dcm_prior^T)
        Eigen::Matrix3d dcm_CW_current = pose_->dcm_CW();
        Eigen::Matrix3d dcm_CW_prior = ExpMapSO3(rot_CW_prior_);
        Eigen::Matrix3d dcm_error = dcm_CW_current * dcm_CW_prior.transpose();
        Eigen::Vector3d res = LogMapSO3(dcm_error);

        result = weight_ * res;
    }

    void PoseOrientationPriorFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 1 variable
        if (jacobians.size() == 0) {
            jacobians.resize(1);
        } else if (jacobians.size() != 1) {
            jacobians.clear();
            jacobians.resize(1);
        }

        if (pose_->is_constant()) {
            // constant variable - empty jacobian
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != 6) {
                jacobians[0].resize(size_, 6);
            }
            jacobians[0].setZero();

            // Use manifold Jacobian for SO(3)
            // For r = log(dcm_current * dcm_prior^T)
            // dr/d(rot_current) = Jr_inv(log(dcm_current * dcm_prior^T))
            Eigen::Matrix3d dcm_CW_current = pose_->dcm_CW();
            Eigen::Matrix3d dcm_CW_prior = ExpMapSO3(rot_CW_prior_);
            Eigen::Matrix3d dcm_error = dcm_CW_current * dcm_CW_prior.transpose();
            Eigen::Vector3d rotvec_error = LogMapSO3(dcm_error);

            // Compute inverse right Jacobian Jr_inv of the error rotation
            Eigen::Matrix3d Jr_inv = compute_inverse_right_jacobian_so3(rotvec_error);
            jacobians[0].block<3, 3>(0, 3) = weight_ * Jr_inv;
        }
    }

    std::vector<Variable *> PoseOrientationPriorFactor::variables()
    {
        return {pose_};
    }

    std::string PoseOrientationPriorFactor::name() const
    {
        return "PoseOrientationPriorFactor(" + pose_->name() + ")";
    }

} // namespace factorama
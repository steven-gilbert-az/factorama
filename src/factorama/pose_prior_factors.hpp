#pragma once
#include <cassert>
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{

    class PosePositionPriorFactor : public Factor
    {
    public:
        PosePositionPriorFactor(int id,
                                std::shared_ptr<PoseVariable> pose,
                                const Eigen::Vector3d &pos_prior,
                                double sigma = 1.0)
            : id_(id), pose_(std::move(pose)), pos_prior_(pos_prior), weight_(1.0 / sigma)
        {
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        int residual_size() const override
        {
            return 3;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::Vector3d res = pose_->pos_W() - pos_prior_;
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            if (pose_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 6);
                J.block<3, 3>(0, 0) = weight_ * Eigen::Matrix3d::Identity();
                jacobians.emplace_back(J);
            }
        }


        std::vector<std::shared_ptr<Variable>> variables() override
        {
            return {pose_};
        }

        double weight() const override
        {
            return weight_;
        }

        std::string name() const override
        {
            return "PosePositionPriorFactor(" + pose_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_position_prior;
        }

    private:
        int id_;
        std::shared_ptr<PoseVariable> pose_;
        Eigen::Vector3d pos_prior_;
        double weight_;
    };

    class PoseOrientationPriorFactor : public Factor
    {
    public:
        PoseOrientationPriorFactor(int id,
                                   std::shared_ptr<PoseVariable> pose,
                                   const Eigen::Vector3d &rotvec_prior,
                                   double sigma = 1.0,
                                   bool do_so3_nudge = true)
            : id_(id), pose_(std::move(pose)), rot_CW_prior_(rotvec_prior), weight_(1.0 / sigma), do_so3_nudge_(do_so3_nudge)
        {
            assert(do_so3_nudge == pose_->do_so3_nudge() && "do_so3_nudge must match pose");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        int residual_size() const override
        {
            return 3;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::Vector3d res;
            
            if (do_so3_nudge_)
            {
                // Use full SO(3) manifold approach
                // For a prior factor: r = log(dcm_current * dcm_prior^T)
                Eigen::Matrix3d dcm_CW_current = pose_->dcm_CW();
                Eigen::Matrix3d dcm_CW_prior = ExpMapSO3(rot_CW_prior_);
                Eigen::Matrix3d dcm_error = dcm_CW_current * dcm_CW_prior.transpose();
                res = LogMapSO3(dcm_error);
            }
            else
            {
                // Use simple R3 vector subtraction
                res = pose_->rot_CW() - rot_CW_prior_;
            }
            
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            if (pose_->is_constant())
            {
                // constant variable - empty jacobian
                jacobians.emplace_back(Eigen::MatrixXd());
            }
            else
            {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 6);
                
                if (do_so3_nudge_)
                {
                    // Use manifold Jacobian for SO(3)
                    // For r = log(dcm_current * dcm_prior^T)
                    // dr/d(rot_current) = Jr_inv(log(dcm_current * dcm_prior^T))
                    Eigen::Matrix3d dcm_CW_current = pose_->dcm_CW();
                    Eigen::Matrix3d dcm_CW_prior = ExpMapSO3(rot_CW_prior_);
                    Eigen::Matrix3d dcm_error = dcm_CW_current * dcm_CW_prior.transpose();
                    Eigen::Vector3d rotvec_error = LogMapSO3(dcm_error);
                    
                    // Compute inverse right Jacobian Jr_inv of the error rotation
                    Eigen::Matrix3d Jr_inv = compute_inverse_right_jacobian_so3(rotvec_error);
                    J.block<3, 3>(0, 3) = weight_ * Jr_inv;
                }
                else
                {
                    // Simple linear case: r = (rotvec_current - rotvec_prior)
                    // dr/d(rot_current) = I
                    J.block<3, 3>(0, 3) = weight_ * Eigen::Matrix3d::Identity();
                }
                jacobians.emplace_back(J);
            }
        }


        std::vector<std::shared_ptr<Variable>> variables() override
        {
            return {pose_};
        }

        double weight() const override
        {
            return weight_;
        }

        std::string name() const override
        {
            return "PoseOrientationPriorFactor(" + pose_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_orientation_prior;
        }

    private:
        int id_;
        std::shared_ptr<PoseVariable> pose_;
        Eigen::Vector3d rot_CW_prior_;
        double weight_;
        bool do_so3_nudge_;
    };

}
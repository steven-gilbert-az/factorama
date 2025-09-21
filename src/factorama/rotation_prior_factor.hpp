#pragma once
#include <cassert>
#include "factorama/types.hpp"
#include "factorama/rotation_variable.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{
    /**
     * @brief Prior constraint for rotation variables using SO(3) manifold
     *
     * Applies manifold-aware prior constraints to RotationVariable instances.
     * Uses proper SO(3) manifold operations.
     *
     * @code
     * Eigen::Matrix3d prior_rotation = Eigen::Matrix3d::Identity();
     * auto rotation_prior = std::make_shared<RotationPriorFactor>(
     *     factor_id++, rotation_var, prior_rotation, sigma);
     * @endcode
     */
    class RotationPriorFactor : public Factor
    {
    public:
        /**
         * @brief Construct rotation prior factor
         * @param id Unique factor identifier
         * @param rotation Target rotation variable
         * @param dcm_AB_prior Prior rotation matrix
         * @param sigma Standard deviation of angular prior (radians)
         */
        RotationPriorFactor(int id,
                                   RotationVariable* rotation,
                                   const Eigen::Matrix3d &dcm_AB_prior,
                                   double sigma = 1.0)
            : id_(id), rotation_(rotation), dcm_AB_prior_(dcm_AB_prior), weight_(1.0 / sigma)
        {
            assert(rotation != nullptr && "rotation variable cannot be nullptr");
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
            
            // Use full SO(3) manifold approach
            // For a prior factor: r = log(dcm_current * dcm_prior^T)
            Eigen::Matrix3d dcm_AB_current = rotation_->dcm_AB();
            Eigen::Matrix3d dcm_error = dcm_AB_current * dcm_AB_prior_.transpose();
            res = LogMapSO3(dcm_error);

            
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            if (rotation_->is_constant())
            {
                // constant variable - empty jacobian
                jacobians.emplace_back(Eigen::MatrixXd());
            }
            else
            {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 6);

                // Use manifold Jacobian for SO(3)
                // For r = log(dcm_current * dcm_prior^T)
                // dr/d(rot_current) = Jr_inv(log(dcm_current * dcm_prior^T))
                Eigen::Matrix3d dcm_AB_current = rotation_->dcm_AB();
                Eigen::Matrix3d dcm_error = dcm_AB_current * dcm_AB_prior_.transpose();
                Eigen::Vector3d rotvec_error = LogMapSO3(dcm_error);
                
                // Compute inverse right Jacobian Jr_inv of the error rotation
                Eigen::Matrix3d Jr_inv = compute_inverse_right_jacobian_so3(rotvec_error);
                J.block<3, 3>(0, 3) = weight_ * Jr_inv;

                jacobians.emplace_back(J);
            }
        }


        std::vector<Variable *> variables() override
        {
            return {rotation_};
        }

        double weight() const override
        {
            return weight_;
        }

        std::string name() const override
        {
            return "PoseOrientationPriorFactor" + std::to_string(id()) + "(" + rotation_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_orientation_prior;
        }

    private:
        int id_;
        RotationVariable* rotation_;
        Eigen::Matrix3d dcm_AB_prior_;
        double weight_;
    };

}
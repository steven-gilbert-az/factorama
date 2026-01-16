#pragma once
#include <cassert>
#include "factorama/base_types.hpp"
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
    class RotationPriorFactor final : public Factor
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
            : rotation_(rotation), dcm_AB_prior_(dcm_AB_prior), weight_(1.0 / sigma), size_(3)
        {
            id_ = id;
            assert(rotation != nullptr && "rotation variable cannot be nullptr");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int residual_size() const override
        {
            return size_;
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

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override
        {
            // Use full SO(3) manifold approach
            // For a prior factor: r = log(dcm_current * dcm_prior^T)
            Eigen::Matrix3d dcm_AB_current = rotation_->dcm_AB();
            Eigen::Matrix3d dcm_error = dcm_AB_current * dcm_AB_prior_.transpose();
            Eigen::Vector3d res = LogMapSO3(dcm_error);

            result = weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            // Ensure jacobians vector has correct size for 1 variable
            if(jacobians.size() == 0) {
                jacobians.resize(1);
            }
            else if(jacobians.size() != 1) {
                jacobians.clear();
                jacobians.resize(1);
            }

            if (rotation_->is_constant())
            {
                // constant variable - empty jacobian
                jacobians[0] = Eigen::MatrixXd();
            }
            else
            {
                if(jacobians[0].rows() != size_ || jacobians[0].cols() != 3) {
                    jacobians[0].resize(size_, 3);
                }
                jacobians[0].setZero();

                // Use manifold Jacobian for SO(3)
                // For r = log(dcm_current * dcm_prior^T)
                // dr/d(rot_current) = Jr_inv(log(dcm_current * dcm_prior^T))
                Eigen::Matrix3d dcm_AB_current = rotation_->dcm_AB();
                Eigen::Matrix3d dcm_error = dcm_AB_current * dcm_AB_prior_.transpose();
                Eigen::Vector3d rotvec_error = LogMapSO3(dcm_error);

                // Compute inverse right Jacobian Jr_inv of the error rotation
                Eigen::Matrix3d Jr_inv = compute_inverse_right_jacobian_so3(rotvec_error);
                jacobians[0].block<3, 3>(0, 0) = weight_ * Jr_inv;
            }
        }


        std::vector<Variable *> variables() override
        {
            return {rotation_};
        }

        double weight() const
        {
            return weight_;
        }

        std::string name() const override
        {
            return "RotationPriorFactor" + std::to_string(id()) + "(" + rotation_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_orientation_prior;
        }

    private:
        RotationVariable* rotation_;
        Eigen::Matrix3d dcm_AB_prior_;
        double weight_;
        int size_;
    };

}
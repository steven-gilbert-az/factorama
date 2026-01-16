#pragma once

#include "factorama/base_types.hpp"
#include "factorama/rotation_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/generic_variable.hpp"
#include "factorama/numerical_jacobian.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace factorama
{
    /**
     * @brief Coordinate transformation constraint factor
     *
     * Constrains the relationship between the same landmark expressed in two different
     * coordinate frames A and B, accounting for rotation, translation, and scaling.
     *
     * The transformation follows: vec_A = scale_AB * dcm_AB * vec_B - B_origin_A
     *
     * @code
     * auto transform_factor = std::make_shared<CoordinateTransformFactor>(
     *     factor_id++, rot_AB, B_origin_A, scale_AB, lm_A, lm_B, sigma);
     * @endcode
     */
    class CoordinateTransformFactor final : public Factor
    {
    public:
        /**
         * @brief Construct coordinate transform factor
         * @param id Unique factor identifier
         * @param rot_AB Rotation that transforms a vector from frame B to frame A
         * @param B_origin_A The origin of frame B in A's coordinate frame
         * @param scale_AB Scale factor transforming frame B's scale into frame A's scale
         * @param lm_A Landmark in coordinate frame A
         * @param lm_B Same landmark expressed in coordinate frame B
         * @param sigma Standard deviation of the constraint (default 1.0)
         */
        CoordinateTransformFactor(int id,
            RotationVariable* rot_AB,
            GenericVariable* B_origin_A,
            GenericVariable* scale_AB,
            LandmarkVariable* lm_A,
            LandmarkVariable* lm_B,
            double sigma = 1.0)
            : rot_AB_(rot_AB),
              B_origin_A_(B_origin_A),
              scale_AB_(scale_AB),
              lm_A_(lm_A),
              lm_B_(lm_B),
              weight_(1.0 / sigma),
              size_(3)
        {
            id_ = id;
            assert(rot_AB != nullptr && "rot_AB cannot be nullptr");
            assert(B_origin_A != nullptr && "B_origin_A cannot be nullptr");
            assert(scale_AB != nullptr && "scale_AB cannot be nullptr");
            assert(lm_A != nullptr && "lm_A cannot be nullptr");
            assert(lm_B != nullptr && "lm_B cannot be nullptr");
            assert(B_origin_A->size() == 3 && "B_origin_A must be 3D");
            assert(scale_AB->size() == 1 && "scale_AB must be scalar (1D)");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int residual_size() const override
        {
            return size_;
        }

        double weight() const { return weight_; }

        Eigen::VectorXd compute_residual() const override
        {
            // Transform lm_B into frame A
            const Eigen::Matrix3d& dcm_AB = rot_AB_->rotation();
            double scale = scale_AB_->value()[0];
            const Eigen::Vector3d& lm_B = lm_B_->value();
            const Eigen::Vector3d& B_origin_A = B_origin_A_->value();
            const Eigen::Vector3d& lm_A = lm_A_->value();

            // vec_A = scale_AB * dcm_AB * vec_B - B_origin_A
            Eigen::Vector3d lm_A_predicted = scale * dcm_AB * lm_B - B_origin_A;

            // Residual is the difference between actual and predicted
            Eigen::Vector3d residual = lm_A - lm_A_predicted;

            return weight_ * residual;
        }

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override
        {
            // Transform lm_B into frame A
            const Eigen::Matrix3d& dcm_AB = rot_AB_->rotation();
            double scale = scale_AB_->value()[0];
            const Eigen::Vector3d& lm_B = lm_B_->value();
            const Eigen::Vector3d& B_origin_A = B_origin_A_->value();
            const Eigen::Vector3d& lm_A = lm_A_->value();

            // vec_A = scale_AB * dcm_AB * vec_B - B_origin_A
            Eigen::Vector3d lm_A_predicted = scale * dcm_AB * lm_B - B_origin_A;

            // Residual is the difference between actual and predicted
            result = weight_ * (lm_A - lm_A_predicted);
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override
        {
            // Use numerical jacobians for now
            // Note: ComputeNumericalJacobians restores all variable values after computation
            //ComputeNumericalJacobians(*const_cast<CoordinateTransformFactor*>(this), jacobians);


            //compute_analytical_jacobians(jacobians);
            compute_numerical_jacobians(jacobians);
        }

        void compute_numerical_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const {
            ComputeNumericalJacobians(*const_cast<CoordinateTransformFactor*>(this), jacobians);
        }

        void compute_analytical_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const {
            // TODO: implement
            (void)jacobians;
        }

        std::vector<Variable*> variables() override
        {
            return {rot_AB_, B_origin_A_, scale_AB_, lm_A_, lm_B_};
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::custom;
        }

        std::string name() const override
        {
            return "CoordinateTransformFactor" + std::to_string(id());
        }

    private:
        RotationVariable* rot_AB_;
        GenericVariable* B_origin_A_;
        GenericVariable* scale_AB_;
        LandmarkVariable* lm_A_;
        LandmarkVariable* lm_B_;
        double weight_;
        int size_;
    };

} // namespace factorama

#pragma once
#include "factorama/base_types.hpp"
#include <cassert>

namespace factorama
{

    /**
     * @brief Generic linear prior constraint for linear variable types
     *
     * Applies linear prior constraints by penalizing deviations from expected values.
     * Use only with linear variables (LandmarkVariable, GenericVariable) - not for
     * rotations or poses which require manifold-aware constraints.
     *
     * @code
     * auto landmark_prior = std::make_shared<GenericPriorFactor>(
     *     factor_id++, landmark, Eigen::Vector3d(0.0, 0.0, 5.0), 1.0);
     * @endcode
     */
    class GenericPriorFactor final : public Factor
    {
    public:
        /**
         * @brief Construct generic prior factor
         * @param id Unique factor identifier
         * @param variable Target variable for prior constraint
         * @param prior_value Expected value for the variable
         * @param sigma Standard deviation of prior measurement
         */
        GenericPriorFactor(int id,
                           Variable* variable,
                           const Eigen::VectorXd &prior_value,
                           double sigma = 1.0)
            : variable_(variable), prior_(prior_value), weight_(1.0 / sigma)
        {
            id_ = id;
            assert(variable != nullptr && "variable cannot be nullptr");
            assert(prior_.size() == variable_->size() && "Prior size must match variable size");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
            size_ = prior_.size();
        }

        int residual_size() const override
        {
            return size_;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::VectorXd res = variable_->value() - prior_;
            return weight_ * res;
        }

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override {
            result = weight_ * (variable_->value() - prior_);
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            if(jacobians.size() == 0) {
                jacobians.emplace_back(); // create a jacobian element
            }
            else if(jacobians.size() > 1) {
                jacobians.clear();
                jacobians.emplace_back();
            }
            
            if (variable_->is_constant())
            {
                jacobians.emplace_back(); // Empty Jacobian if variable is constant
                jacobians[0] = Eigen::MatrixXd();
            }
            else
            {
                if(jacobians[0].rows() != size_ || jacobians[0].cols() != size_) {
                    jacobians[0].resize(size_, size_);
                }
                jacobians[0].setZero();
                jacobians[0].diagonal().array() = weight_;
            }
        }

        std::vector<Variable *> variables() override
        {
            return {variable_};
        }

        double weight() const
        {
            return weight_;
        }

        std::string name() const override
        {
            return "GenericPriorFactor(" + variable_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::generic_prior;
        }

    private:
        Variable* variable_;
        Eigen::VectorXd prior_;
        double weight_;
        int size_;
    };

}
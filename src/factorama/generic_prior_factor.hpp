#pragma once
#include "factorama/types.hpp"
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
    class GenericPriorFactor : public Factor
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
            : id_(id), variable_(variable), prior_(prior_value), weight_(1.0 / sigma)
        {
            assert(variable != nullptr && "variable cannot be nullptr");
            assert(prior_.size() == variable_->size() && "Prior size must match variable size");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        int residual_size() const override
        {
            return prior_.size();
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::VectorXd res = variable_->value() - prior_;
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            if (variable_->is_constant())
            {
                jacobians.emplace_back(); // Empty Jacobian if variable is constant
            }
            else
            {
                Eigen::MatrixXd J = weight_ * Eigen::MatrixXd::Identity(prior_.size(), prior_.size());
                jacobians.emplace_back(J);
            }
        }

        std::vector<Variable *> variables() override
        {
            return {variable_};
        }

        double weight() const override
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
        int id_;
        Variable* variable_;
        Eigen::VectorXd prior_;
        double weight_;
    };

}
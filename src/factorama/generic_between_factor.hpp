#pragma once
#include "factorama/types.hpp"
#include <cassert>

namespace factorama
{

    /**
     * @brief Generic linear relative constraint between variables
     *
     * Constrains the linear difference between two variables to match a measured difference variable.
     * Use only with linear variables (LandmarkVariable, GenericVariable) - not for
     * rotations or poses which require manifold-aware constraints.
     *
     * @code
     * auto relative_constraint = std::make_shared<GenericBetweenFactor>(
     *     factor_id++, var_a, var_b, measured_diff_variable, sigma);
     * @endcode
     */
    class GenericBetweenFactor : public Factor
    {
    public:
        /**
         * @brief Construct generic between factor
         * @param id Unique factor identifier
         * @param var_a First variable
         * @param var_b Second variable
         * @param measured_diff Variable representing measured difference (var_b - var_a)
         * @param sigma Standard deviation of measurement
         */
        GenericBetweenFactor(int id,
                             Variable* var_a,
                             Variable* var_b,
                             Variable* measured_diff,
                             double sigma = 1.0);

        int id() const override { return id_; }
        int residual_size() const override { return measured_diff_->size(); }
        double weight() const override { return weight_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::generic_between; }

        Eigen::VectorXd compute_residual() const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;
        std::string name() const override;

    private:
        int id_;
        Variable* var_a_;
        Variable* var_b_;
        Variable* measured_diff_;
        double weight_;
    };

} // namespace factorama
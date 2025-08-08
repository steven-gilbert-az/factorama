#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cassert>

namespace factorama
{

/**
 * @brief Computes numerical jacobians for any factor using central differences
 * 
 * This generic template function computes jacobians by perturbing each variable
 * parameter and observing the change in residual. It works with any factor type
 * that provides the required interface.
 * 
 * @param factor The factor to compute jacobians for (will be temporarily modified)
 * @param jacobians_out Output vector containing one jacobian per variable
 * @param epsilon Perturbation step size for numerical differentiation
 * 
 * Behavior:
 * - Uses central differences: (f(x+ε) - f(x-ε)) / (2ε) for better accuracy
 * - Handles constant variables by returning empty (0x0) matrices as signals
 * - Always returns same number of jacobians as factor.variables().size()
 * - Jacobians are ordered to match the factor's variables() ordering
 * - Restores all variables to original values after computation ("soft const")
 * 
 * Requirements:
 * - FactorT must provide: variables(), compute_residual(), residual_size()
 * - Variables must support: is_constant(), apply_increment(), set_value_from_vector()
 */
template <typename FactorT>
void ComputeNumericalJacobians(FactorT& factor,
                               std::vector<Eigen::MatrixXd>& jacobians_out,
                               double epsilon = 1e-6)
{
    const auto& vars = factor.variables();
    const int num_vars = vars.size();
    const int residual_dim = factor.residual_size();

    Eigen::VectorXd residual0 = factor.compute_residual();
    assert(residual0.size() == residual_dim);

    jacobians_out.resize(num_vars);

    for (int vi = 0; vi < num_vars; ++vi)
    {
        auto var = vars[vi];
        
        if (var->is_constant())
        {
            // Empty matrix signals constant variable
            jacobians_out[vi] = Eigen::MatrixXd();
        }
        else
        {
            const int var_dim = var->size();
            jacobians_out[vi] = Eigen::MatrixXd(residual_dim, var_dim);

            for (int k = 0; k < var_dim; ++k)
            {
                // Store original value
                Eigen::VectorXd x0 = var->value();

                // Perturbation
                Eigen::VectorXd dx = Eigen::VectorXd::Zero(var_dim);
                dx[k] = epsilon;

                // Apply +epsilon
                var->apply_increment(dx);
                Eigen::VectorXd res_plus = factor.compute_residual();

                // Reset
                var->set_value_from_vector(x0);

                // Apply -epsilon
                var->apply_increment(-dx);
                Eigen::VectorXd res_minus = factor.compute_residual();

                // Reset again
                var->set_value_from_vector(x0);

                // Central difference
                jacobians_out[vi].col(k) = (res_plus - res_minus) / (2.0 * epsilon);
            }
        }
    }
}

}  // namespace factorama

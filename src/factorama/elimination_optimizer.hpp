#pragma once
#include <cstddef>
#include <optional>
#include <unordered_map>

#include "factorama/factor_graph.hpp"
#include "factorama/sparse_optimizer.hpp"

namespace factorama
{

    /**
     * @brief Represents a linearized factor's contribution in information form
     *
     * Used during variable elimination to represent the information contribution
     * from a single factor or a group of factors after Schur complement computation.
     */
    struct InfoFactor
    {
        std::vector<Variable*> variables;         // Active (non-constant) variables
        Eigen::MatrixXd information_matrix_H;     // Block information matrix (sum of J_i^T * J_j)
        Eigen::VectorXd gradient_vector_g;        // Block gradient (sum of J_i^T * (-residual))

        /**
         * @brief Get offset for variable at given index within this factor's blocks
         * @param var_index Index into the variables vector
         * @return Offset within information_matrix_H and gradient_vector_g
         */
        int variable_offset(int var_index) const
        {
            int offset = 0;
            for (int i = 0; i < var_index; ++i) {
                offset += variables[i]->size();
            }
            return offset;
        }

        /**
         * @brief Get total dimension of all variables in this factor
         * @return Sum of variable dimensions
         */
        int total_dim() const
        {
            int dim = 0;
            for (const auto* var : variables) {
                dim += var->size();
            }
            return dim;
        }
    };

    /**
     * @brief Stores the result of eliminating a variable (a Gaussian conditional)
     *
     * The conditional represents: upper_triangular_R * x_frontal + cross_term_S * x_separator = rhs_d
     * This can be solved for x_frontal given the separator variable values.
     */
    struct GaussianConditional
    {
        Variable* frontal_variable = nullptr;              // Variable that was eliminated
        std::vector<Variable*> separator_variables;        // Parent variables in the Bayes net
        Eigen::MatrixXd upper_triangular_R;                // Upper triangular (frontal_dim x frontal_dim)
        Eigen::MatrixXd cross_term_S;                      // Cross term (frontal_dim x separator_total_dim)
        Eigen::VectorXd rhs_d;                             // Right-hand side

        /**
         * @brief Solve for frontal variable given separator values
         * @param separator_values Map from variable ID to its current delta value
         * @return Delta for the frontal variable
         */
        Eigen::VectorXd solve(const std::unordered_map<int, Eigen::VectorXd>& separator_values) const
        {
            // Compute rhs_d - cross_term_S * x_separator
            Eigen::VectorXd adjusted_rhs = rhs_d;

            if (!separator_variables.empty() && cross_term_S.cols() > 0) {
                int sep_offset = 0;
                for (const auto* sep_var : separator_variables) {
                    int sep_dim = sep_var->size();
                    auto it = separator_values.find(sep_var->id());
                    if (it != separator_values.end()) {
                        adjusted_rhs -= cross_term_S.middleCols(sep_offset, sep_dim) * it->second;
                    }
                    sep_offset += sep_dim;
                }
            }

            // Solve R * x_frontal = adjusted_rhs via back-substitution
            return upper_triangular_R.triangularView<Eigen::Upper>().solve(adjusted_rhs);
        }
    };

    /**
     * @brief Variable elimination-based optimizer for factor graphs
     *
     * Implements optimization via variable elimination (sparse Cholesky factorization
     * on the fly), avoiding explicit formation of the full Hessian matrix.
     * Instead, variables are eliminated one-by-one using partial Cholesky factorization,
     * producing a Bayes net of conditionals, then back-substitution gives the solution.
     *
     * This approach can be more efficient for problems with sparse structure, especially
     * when using a good elimination ordering.
     *
     * @code
     * EliminationOptimizer optimizer;
     * OptimizerSettings settings;
     * settings.method = OptimizerMethod::GaussNewton;
     * optimizer.setup(graph_ptr, settings);
     * optimizer.optimize();
     * @endcode
     */
    class EliminationOptimizer
    {
    public:
        EliminationOptimizer() = default;

        /**
         * @brief Configure optimizer with factor graph and settings
         * @param graph_ptr Shared pointer to finalized FactorGraph
         * @param settings Optimization configuration (algorithm, tolerances, etc.)
         */
        void setup(std::shared_ptr<FactorGraph> graph_ptr, const OptimizerSettings& settings);

        /**
         * @brief Run optimization until convergence or max iterations
         * Updates variable values in the associated FactorGraph
         */
        void optimize();

        /**
         * @brief Get current optimizer settings
         * @return Reference to optimization settings
         */
        const OptimizerSettings& settings() const { return settings_; }

        OptimizerStats initial_stats_;
        OptimizerStats current_stats_;

    private:
        // ---- Core Settings and Graph ----
        OptimizerSettings settings_;
        std::shared_ptr<FactorGraph> graph_;

        bool optimization_complete_ = false;

        // ---- Elimination Order ----
        std::vector<int> elimination_order_;  // Variable IDs in elimination order

        // Cached residuals for convergence checking
        Eigen::VectorXd residual1_;
        Eigen::VectorXd residual2_;

        // ---- Core Algorithm Methods ----

        /**
         * @brief Linearize a single factor into information form
         * @param factor Pointer to the factor to linearize
         * @return InfoFactor containing information matrix and gradient
         */
        InfoFactor linearize_factor(Factor* factor) const;

        /**
         * @brief Compute elimination order for variables
         * Currently uses natural order (order variables were added)
         */
        void compute_elimination_order();

        /**
         * @brief Eliminate a single variable from the active factors
         * @param active_factors List of active InfoFactors (modified in place)
         * @param variable_to_eliminate Variable to eliminate
         * @param damping_lambda LM damping parameter (0 for Gauss-Newton)
         * @return Pair of (conditional, optional separator_factor)
         */
        std::pair<GaussianConditional, std::optional<InfoFactor>> eliminate_variable(
            std::vector<InfoFactor>& active_factors,
            Variable* variable_to_eliminate,
            double damping_lambda);

        /**
         * @brief Back-substitute through Bayes net to get solution
         * @param bayes_net Vector of conditionals in elimination order
         * @return Delta vector for all variables
         */
        Eigen::VectorXd back_substitute(const std::vector<GaussianConditional>& bayes_net) const;

        /**
         * @brief Apply increment to all variables
         * @param dx Delta vector
         */
        void apply_increment(const Eigen::VectorXd& dx);

        // ---- Single Optimization Step ----
        void single_step_gauss_newton();
        void single_step_levenberg_marquardt();
    };

} // namespace factorama

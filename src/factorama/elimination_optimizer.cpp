#include "factorama/elimination_optimizer.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <set>

namespace factorama
{

    void EliminationOptimizer::setup(std::shared_ptr<FactorGraph> graph_ptr, const OptimizerSettings& settings)
    {
        settings_ = settings;
        graph_ = graph_ptr;
        optimization_complete_ = false;
        compute_elimination_order();
    }

    void EliminationOptimizer::compute_elimination_order()
    {
        // Simple natural order: eliminate variables in the order they were added
        // Future enhancement: use AMD or other heuristics for better fill-in reduction
        elimination_order_.clear();

        for (const auto& var : graph_->get_all_variables()) {
            if (!var->is_constant()) {
                bool valid = false;
                VariablePlacement placement = graph_->variable_placement(var->id(), valid);
                // Only include variables that have non-zero dimension (not constant)
                if (valid && placement.dim > 0) {
                    elimination_order_.push_back(var->id());
                }
            }
        }
    }

    InfoFactor EliminationOptimizer::linearize_factor(Factor* factor) const
    {
        InfoFactor info;

        // Get jacobians
        std::vector<Eigen::MatrixXd> jacobians;
        factor->compute_jacobians(jacobians);

        // Get residual
        Eigen::VectorXd residual = factor->compute_residual();

        // Get variables and filter out constant ones (those with empty jacobians)
        std::vector<Variable*> all_vars = factor->variables();

        // Collect active (non-constant) variables with their jacobians
        std::vector<Variable*> active_vars;
        std::vector<Eigen::MatrixXd> active_jacobians;

        for (size_t i = 0; i < all_vars.size(); ++i) {
            // A variable is considered constant if:
            // 1. Its jacobian is empty (0x0 matrix), or
            // 2. It's marked as constant
            if (jacobians[i].rows() > 0 && jacobians[i].cols() > 0 && !all_vars[i]->is_constant()) {
                active_vars.push_back(all_vars[i]);
                active_jacobians.push_back(jacobians[i]);
            }
        }

        if (active_vars.empty()) {
            // All variables are constant - return empty info factor
            info.information_matrix_H = Eigen::MatrixXd::Zero(0, 0);
            info.gradient_vector_g = Eigen::VectorXd::Zero(0);
            return info;
        }

        info.variables = active_vars;

        // Compute total dimension
        int total_dim = 0;
        for (const auto* var : active_vars) {
            total_dim += var->size();
        }

        // Build information matrix H = sum of J_i^T * J_j for all variable pairs
        info.information_matrix_H = Eigen::MatrixXd::Zero(total_dim, total_dim);
        info.gradient_vector_g = Eigen::VectorXd::Zero(total_dim);

        int row_offset = 0;
        for (size_t i = 0; i < active_vars.size(); ++i) {
            int dim_i = active_vars[i]->size();
            const Eigen::MatrixXd& Ji = active_jacobians[i];

            // Diagonal block: J_i^T * J_i
            info.information_matrix_H.block(row_offset, row_offset, dim_i, dim_i) = Ji.transpose() * Ji;

            // Gradient: J_i^T * (-residual)
            info.gradient_vector_g.segment(row_offset, dim_i) = Ji.transpose() * (-residual);

            // Off-diagonal blocks: J_i^T * J_j
            int col_offset = row_offset + dim_i;
            for (size_t j = i + 1; j < active_vars.size(); ++j) {
                int dim_j = active_vars[j]->size();
                const Eigen::MatrixXd& Jj = active_jacobians[j];

                Eigen::MatrixXd cross = Ji.transpose() * Jj;
                info.information_matrix_H.block(row_offset, col_offset, dim_i, dim_j) = cross;
                info.information_matrix_H.block(col_offset, row_offset, dim_j, dim_i) = cross.transpose();

                col_offset += dim_j;
            }

            row_offset += dim_i;
        }

        return info;
    }

    std::pair<GaussianConditional, std::optional<InfoFactor>> EliminationOptimizer::eliminate_variable(
        std::vector<InfoFactor>& active_factors,
        Variable* variable_to_eliminate,
        double damping_lambda)
    {
        GaussianConditional conditional;
        conditional.frontal_variable = variable_to_eliminate;

        int frontal_id = variable_to_eliminate->id();
        int frontal_dim = variable_to_eliminate->size();

        // Step 1: Gather all InfoFactors containing this variable
        std::vector<size_t> involved_indices;
        for (size_t i = 0; i < active_factors.size(); ++i) {
            for (const auto* var : active_factors[i].variables) {
                if (var->id() == frontal_id) {
                    involved_indices.push_back(i);
                    break;
                }
            }
        }

        if (involved_indices.empty()) {
            // Variable has no factors - create identity conditional
            conditional.upper_triangular_R = Eigen::MatrixXd::Identity(frontal_dim, frontal_dim);
            conditional.cross_term_S = Eigen::MatrixXd::Zero(frontal_dim, 0);
            conditional.rhs_d = Eigen::VectorXd::Zero(frontal_dim);
            return {conditional, std::nullopt};
        }

        // Step 2: Collect all separator variables (variables that appear in involved factors, excluding frontal)
        std::set<int> separator_ids_set;
        std::vector<Variable*> separator_vars;

        for (size_t idx : involved_indices) {
            for (auto* var : active_factors[idx].variables) {
                if (var->id() != frontal_id && separator_ids_set.find(var->id()) == separator_ids_set.end()) {
                    separator_ids_set.insert(var->id());
                    separator_vars.push_back(var);
                }
            }
        }

        conditional.separator_variables = separator_vars;

        // Compute separator total dimension
        int separator_dim = 0;
        for (const auto* var : separator_vars) {
            separator_dim += var->size();
        }

        // Step 3: Build joint information matrix over [frontal | separator]
        int joint_dim = frontal_dim + separator_dim;
        Eigen::MatrixXd joint_H = Eigen::MatrixXd::Zero(joint_dim, joint_dim);
        Eigen::VectorXd joint_g = Eigen::VectorXd::Zero(joint_dim);

        // Helper to find offset of a variable in [frontal | separator] ordering
        auto get_joint_offset = [&](int var_id) -> int {
            if (var_id == frontal_id) {
                return 0;
            }
            int offset = frontal_dim;
            for (const auto* var : separator_vars) {
                if (var->id() == var_id) {
                    return offset;
                }
                offset += var->size();
            }
            return -1; // Not found
        };

        // Accumulate contributions from all involved factors
        for (size_t idx : involved_indices) {
            const InfoFactor& factor = active_factors[idx];

            for (size_t i = 0; i < factor.variables.size(); ++i) {
                int var_i_id = factor.variables[i]->id();
                int dim_i = factor.variables[i]->size();
                int joint_offset_i = get_joint_offset(var_i_id);

                if (joint_offset_i < 0) continue;

                int factor_offset_i = factor.variable_offset(i);

                // Gradient contribution
                joint_g.segment(joint_offset_i, dim_i) +=
                    factor.gradient_vector_g.segment(factor_offset_i, dim_i);

                // Information matrix contributions
                for (size_t j = 0; j < factor.variables.size(); ++j) {
                    int var_j_id = factor.variables[j]->id();
                    int dim_j = factor.variables[j]->size();
                    int joint_offset_j = get_joint_offset(var_j_id);

                    if (joint_offset_j < 0) continue;

                    int factor_offset_j = factor.variable_offset(j);

                    joint_H.block(joint_offset_i, joint_offset_j, dim_i, dim_j) +=
                        factor.information_matrix_H.block(factor_offset_i, factor_offset_j, dim_i, dim_j);
                }
            }
        }

        // Step 4: Add LM damping to frontal block if needed
        if (damping_lambda > 0) {
            for (int i = 0; i < frontal_dim; ++i) {
                joint_H(i, i) += damping_lambda;
            }
        }

        // Step 5: Perform partial Cholesky factorization
        // Extract blocks:
        //   A = joint_H[0:frontal_dim, 0:frontal_dim]           (frontal block)
        //   B = joint_H[0:frontal_dim, frontal_dim:joint_dim]   (cross block)
        //   C = joint_H[frontal_dim:joint_dim, frontal_dim:joint_dim] (separator block)

        Eigen::MatrixXd A = joint_H.topLeftCorner(frontal_dim, frontal_dim);
        Eigen::VectorXd g_frontal = joint_g.head(frontal_dim);

        // Cholesky decomposition of A: A = R^T * R where R is upper triangular
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        if (llt.info() != Eigen::Success) {
            // Factorization failed - matrix may be singular
            throw std::runtime_error("[EliminationOptimizer] Cholesky factorization failed for variable " +
                                     variable_to_eliminate->name());
        }

        // R = Cholesky factor (upper triangular)
        Eigen::MatrixXd R = llt.matrixL().transpose();
        conditional.upper_triangular_R = R;

        // Compute d = R^{-T} * g_frontal
        Eigen::VectorXd d = R.transpose().triangularView<Eigen::Lower>().solve(g_frontal);
        conditional.rhs_d = d;

        // Handle separator (if any)
        if (separator_dim > 0) {
            Eigen::MatrixXd B = joint_H.topRightCorner(frontal_dim, separator_dim);
            Eigen::MatrixXd C = joint_H.bottomRightCorner(separator_dim, separator_dim);
            Eigen::VectorXd g_separator = joint_g.tail(separator_dim);

            // Compute S = R^{-T} * B
            Eigen::MatrixXd S = R.transpose().triangularView<Eigen::Lower>().solve(B);
            conditional.cross_term_S = S;

            // Compute Schur complement: schur_H = C - S^T * S
            Eigen::MatrixXd schur_H = C - S.transpose() * S;

            // Compute schur_g = g_separator - S^T * d
            Eigen::VectorXd schur_g = g_separator - S.transpose() * d;

            // Create new InfoFactor on separator variables
            InfoFactor separator_factor;
            separator_factor.variables = separator_vars;
            separator_factor.information_matrix_H = schur_H;
            separator_factor.gradient_vector_g = schur_g;

            // Remove consumed factors from active list (in reverse order to preserve indices)
            std::sort(involved_indices.begin(), involved_indices.end(), std::greater<size_t>());
            for (size_t idx : involved_indices) {
                active_factors.erase(active_factors.begin() + idx);
            }

            return {conditional, separator_factor};
        } else {
            // No separator - just create empty cross term
            conditional.cross_term_S = Eigen::MatrixXd::Zero(frontal_dim, 0);

            // Remove consumed factors
            std::sort(involved_indices.begin(), involved_indices.end(), std::greater<size_t>());
            for (size_t idx : involved_indices) {
                active_factors.erase(active_factors.begin() + idx);
            }

            return {conditional, std::nullopt};
        }
    }

    Eigen::VectorXd EliminationOptimizer::back_substitute(const std::vector<GaussianConditional>& bayes_net) const
    {
        // Build solution map: variable_id -> delta
        std::unordered_map<int, Eigen::VectorXd> solution;

        // Total dimension for output vector
        int total_dim = graph_->num_values();
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(total_dim);

        // Iterate through conditionals in reverse order (back-substitute)
        for (auto it = bayes_net.rbegin(); it != bayes_net.rend(); ++it) {
            const GaussianConditional& cond = *it;

            if (cond.frontal_variable == nullptr) {
                continue;
            }

            // Solve for frontal variable
            Eigen::VectorXd frontal_delta = cond.solve(solution);

            // Store in solution map
            solution[cond.frontal_variable->id()] = frontal_delta;

            // Also place in output dx vector at the correct position
            bool valid = false;
            VariablePlacement placement = graph_->variable_placement(cond.frontal_variable->id(), valid);
            if (valid && placement.dim > 0) {
                dx.segment(placement.index, placement.dim) = frontal_delta;
            }
        }

        return dx;
    }

    void EliminationOptimizer::apply_increment(const Eigen::VectorXd& dx)
    {
        const auto& variables = graph_->get_all_variables();
        for (const auto& var : variables) {
            if (var->is_constant()) {
                continue;
            }

            bool valid = false;
            VariablePlacement placement = graph_->variable_placement(var->id(), valid);
            if (!valid || placement.dim == 0) {
                continue;
            }

            const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }
    }

    void EliminationOptimizer::single_step_gauss_newton()
    {
        if (!graph_) {
            throw std::runtime_error("[EliminationOptimizer] No factor graph set.");
        }

        // Phase 1: Linearize all factors
        std::vector<InfoFactor> active_factors;
        const auto& factors = graph_->get_all_factors();

        for (const auto& factor_ptr : factors) {
            InfoFactor info = linearize_factor(factor_ptr.get());
            if (!info.variables.empty()) {
                active_factors.push_back(std::move(info));
            }
        }

        // Phase 2: Elimination - eliminate variables in order
        std::vector<GaussianConditional> bayes_net;

        for (int var_id : elimination_order_) {
            Variable* var = graph_->get_variable(var_id);
            if (!var || var->is_constant()) {
                continue;
            }

            auto [conditional, separator_factor] = eliminate_variable(active_factors, var, 0.0);
            bayes_net.push_back(std::move(conditional));

            if (separator_factor.has_value() && !separator_factor->variables.empty()) {
                active_factors.push_back(std::move(*separator_factor));
            }
        }

        // Phase 3: Back-substitute to get solution
        Eigen::VectorXd dx = back_substitute(bayes_net);

        // Apply learning rate
        dx *= settings_.learning_rate;

        // Phase 4: Apply increments
        apply_increment(dx);

        // Compute updated residual and update statistics
        graph_->compute_full_residual_vector(residual2_);
        current_stats_.delta_norm = dx.norm();
        current_stats_.chi2 = residual2_.squaredNorm();
        current_stats_.residual_norm = residual2_.norm();
        current_stats_.current_iteration++;

        if (settings_.verbose) {
            std::cout << "\n[ElimGN] Iteration " << current_stats_.current_iteration << " Summary:\n";
            std::cout << "  Residual norm:   " << current_stats_.residual_norm << "\n";
            std::cout << "  Step norm:       " << current_stats_.delta_norm << "\n";
            std::cout << "  chi²:            " << current_stats_.chi2 << "\n";
        }
    }

    void EliminationOptimizer::single_step_levenberg_marquardt()
    {
        if (!graph_) {
            throw std::runtime_error("[EliminationOptimizer] No factor graph set.");
        }

        // Compute current residual
        graph_->compute_full_residual_vector(residual1_);
        double current_cost = 0.5 * residual1_.squaredNorm();

        // Initialize damping parameter if this is the first iteration
        if (current_stats_.current_iteration == 0) {
            current_stats_.damping_parameter = settings_.initial_lambda;
        }

        bool step_accepted = false;
        Eigen::VectorXd dx;
        double new_cost = current_cost;
        int lm_attempts = 0;
        const int max_lm_attempts = 5;

        // Store original values for potential rollback
        std::vector<Eigen::VectorXd> original_values;
        const auto& variables = graph_->get_all_variables();
        for (const auto& var : variables) {
            original_values.push_back(var->value());
        }

        while (!step_accepted && lm_attempts < max_lm_attempts) {
            try {
                // Phase 1: Linearize all factors
                std::vector<InfoFactor> active_factors;
                const auto& factors = graph_->get_all_factors();

                for (const auto& factor_ptr : factors) {
                    InfoFactor info = linearize_factor(factor_ptr.get());
                    if (!info.variables.empty()) {
                        active_factors.push_back(std::move(info));
                    }
                }

                // Phase 2: Elimination with damping
                std::vector<GaussianConditional> bayes_net;

                for (int var_id : elimination_order_) {
                    Variable* var = graph_->get_variable(var_id);
                    if (!var || var->is_constant()) {
                        continue;
                    }

                    auto [conditional, separator_factor] =
                        eliminate_variable(active_factors, var, current_stats_.damping_parameter);
                    bayes_net.push_back(std::move(conditional));

                    if (separator_factor.has_value() && !separator_factor->variables.empty()) {
                        active_factors.push_back(std::move(*separator_factor));
                    }
                }

                // Phase 3: Back-substitute
                dx = back_substitute(bayes_net);
                dx *= settings_.learning_rate;

                // Phase 4: Apply increments temporarily
                apply_increment(dx);

                // Evaluate new cost
                graph_->compute_full_residual_vector(residual2_);
                new_cost = 0.5 * residual2_.squaredNorm();

                // Check if step should be accepted
                if (new_cost < current_cost) {
                    step_accepted = true;
                    current_stats_.damping_parameter *= settings_.lambda_down_factor;
                    current_stats_.damping_parameter = std::max(current_stats_.damping_parameter, 1e-12);

                    if (settings_.verbose) {
                        std::cout << "[ElimLM] Step accepted, cost: " << current_cost << " -> " << new_cost
                                  << ", decreasing lambda to " << current_stats_.damping_parameter << "\n";
                    }
                } else {
                    // Reject step: restore original values and increase damping
                    for (size_t i = 0; i < variables.size(); ++i) {
                        variables[i]->set_value_from_vector(original_values[i]);
                    }

                    current_stats_.damping_parameter *= settings_.lambda_up_factor;
                    current_stats_.damping_parameter = std::min(current_stats_.damping_parameter, settings_.max_lambda);
                    lm_attempts++;

                    if (settings_.verbose) {
                        std::cout << "[ElimLM] Step rejected, cost: " << current_cost << " -> " << new_cost
                                  << ", increasing lambda to " << current_stats_.damping_parameter << "\n";
                    }

                    if (current_stats_.damping_parameter >= settings_.max_lambda) {
                        if (settings_.verbose) {
                            std::cout << "[ElimLM] Lambda too large, stopping\n";
                        }
                        break;
                    }
                }
            } catch (const std::exception& e) {
                // Factorization failed - increase damping and try again
                for (size_t i = 0; i < variables.size(); ++i) {
                    variables[i]->set_value_from_vector(original_values[i]);
                }

                current_stats_.damping_parameter *= settings_.lambda_up_factor;
                lm_attempts++;

                if (settings_.verbose) {
                    std::cout << "[ElimLM] Factorization failed: " << e.what()
                              << ", increasing lambda to " << current_stats_.damping_parameter << "\n";
                }
            }
        }

        // If no step was accepted, we're done (convergence or failure)
        if (!step_accepted) {
            double initial_cost = initial_stats_.chi2 * 0.5;
            if (current_cost < initial_cost) {
                current_stats_.valid = true;
                current_stats_.status = OptimizerStatus::SUCCESS;
            } else {
                current_stats_.valid = false;
                current_stats_.status = (current_stats_.damping_parameter >= settings_.max_lambda)
                                            ? OptimizerStatus::ILL_CONDITIONED
                                            : OptimizerStatus::FAILED;
            }

            if (settings_.verbose) {
                std::cout << "[ElimLM] No acceptable step found after " << max_lm_attempts << " attempts\n";
            }
            dx = Eigen::VectorXd::Zero(graph_->num_values());
            new_cost = current_cost;
        }

        // Update statistics
        Eigen::VectorXd& r_final = step_accepted ? residual2_ : residual1_;
        current_stats_.delta_norm = dx.norm();
        current_stats_.chi2 = r_final.squaredNorm();
        current_stats_.residual_norm = r_final.norm();
        current_stats_.current_iteration++;

        if (settings_.verbose) {
            std::cout << "\n[ElimLM] Iteration " << current_stats_.current_iteration << " Summary:\n";
            std::cout << "  Residual norm:   " << current_stats_.residual_norm << "\n";
            std::cout << "  Step norm:       " << current_stats_.delta_norm << "\n";
            std::cout << "  chi²:            " << current_stats_.chi2 << "\n";
            std::cout << "  Damping (λ):     " << current_stats_.damping_parameter << "\n";
        }
    }

    void EliminationOptimizer::optimize()
    {
        if (!graph_) {
            throw std::runtime_error("[EliminationOptimizer] No factor graph set.");
        }

        // Initialize stats
        current_stats_ = OptimizerStats{};
        current_stats_.valid = true;
        current_stats_.status = OptimizerStatus::RUNNING;
        current_stats_.current_iteration = 0;

        // Store initial residual for convergence checking
        const Eigen::VectorXd initial_residual = graph_->compute_full_residual_vector();
        current_stats_.residual_norm = initial_residual.norm();
        current_stats_.chi2 = initial_residual.squaredNorm();

        initial_stats_ = current_stats_;

        if (settings_.verbose) {
            std::cout << "\n=== EliminationOptimizer Starting ===\n";
            std::cout << "Algorithm: "
                      << (settings_.method == OptimizerMethod::GaussNewton ? "Gauss-Newton" : "Levenberg-Marquardt")
                      << "\n";
            std::cout << "Max iterations: " << settings_.max_num_iterations << "\n";
            std::cout << "Initial residual norm: " << initial_stats_.residual_norm << "\n";
            std::cout << "Initial chi²: " << initial_stats_.chi2 << "\n";
            std::cout << "Variables to eliminate: " << elimination_order_.size() << "\n";
            std::cout << "=================================\n";
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Main optimization loop
        for (size_t i = 0; i < settings_.max_num_iterations; ++i) {
            // Store previous stats for convergence checking
            auto prev_stats = current_stats_;

            // Perform one optimization step
            try {
                if (settings_.method == OptimizerMethod::GaussNewton) {
                    single_step_gauss_newton();
                } else {
                    single_step_levenberg_marquardt();
                }
            } catch (const std::exception& e) {
                std::cerr << "[EliminationOptimizer] Error in iteration " << i << ": " << e.what() << std::endl;
                current_stats_.valid = false;
                current_stats_.status = OptimizerStatus::FAILED;
                break;
            }

            // Check if step failed due to matrix issues
            if (current_stats_.status == OptimizerStatus::SINGULAR_HESSIAN ||
                current_stats_.status == OptimizerStatus::ILL_CONDITIONED ||
                current_stats_.status == OptimizerStatus::FAILED) {
                if (settings_.verbose) {
                    std::cout << "[EliminationOptimizer] Stopping due to error status.\n";
                }
                break;
            }

            // Check convergence based on step norm
            if (current_stats_.delta_norm < settings_.step_tolerance) {
                if (settings_.verbose) {
                    std::cout << "[EliminationOptimizer] Converged: Step norm (" << current_stats_.delta_norm
                              << ") below tolerance (" << settings_.step_tolerance << ")\n";
                }
                current_stats_.status = OptimizerStatus::SUCCESS;
                break;
            }

            // Check convergence based on residual improvement
            double residual_improvement = prev_stats.residual_norm - current_stats_.residual_norm;

            if (residual_improvement < 0.0) {
                if (settings_.verbose) {
                    std::cout << "[EliminationOptimizer] Aborted: residual went up - improvement: "
                              << residual_improvement << std::endl;
                }
                if (current_stats_.residual_norm >= initial_stats_.residual_norm) {
                    current_stats_.status = OptimizerStatus::DIVERGED;
                } else {
                    current_stats_.status = OptimizerStatus::SUCCESS;
                }
                break;
            } else if (residual_improvement < settings_.residual_tolerance) {
                if (settings_.verbose) {
                    std::cout << "[EliminationOptimizer] Converged: Residual improvement (" << residual_improvement
                              << ") below tolerance (" << settings_.residual_tolerance << ")\n";
                }
                current_stats_.status = OptimizerStatus::SUCCESS;
                break;
            }

            // Check for divergence
            if (current_stats_.residual_norm > 10.0 * initial_stats_.residual_norm) {
                if (settings_.verbose) {
                    std::cout << "[EliminationOptimizer] Diverged: Residual norm too large\n";
                }
                current_stats_.valid = false;
                current_stats_.status = OptimizerStatus::DIVERGED;
                break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end_time - start_time;

        // Set final status if still RUNNING (max iterations reached)
        if (current_stats_.status == OptimizerStatus::RUNNING) {
            current_stats_.status = OptimizerStatus::SUCCESS;
        }

        if (settings_.verbose) {
            std::cout << "\n=== EliminationOptimizer Summary ===\n";
            std::cout << "Total iterations: " << current_stats_.current_iteration << "\n";
            std::cout << "Final residual norm: " << current_stats_.residual_norm << "\n";
            std::cout << "Final chi²: " << current_stats_.chi2 << "\n";
            std::cout << "Final step norm: " << current_stats_.delta_norm << "\n";
            std::cout << "Residual improvement: " << (initial_stats_.residual_norm - current_stats_.residual_norm)
                      << "\n";
            std::cout << "Total time: " << elapsed_sec.count() << " seconds\n";
            std::cout << "Optimization " << (current_stats_.valid ? "SUCCESS" : "FAILED") << "\n";
            std::cout << "===============================\n";
        }

        optimization_complete_ = true;
    }

} // namespace factorama

#include "factorama/sparse_optimizer.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>

namespace factorama
{

    void SparseOptimizer::single_step_gauss_newton()
    {
        if (!graph_)
        {
            throw std::runtime_error("[SparseOptimizer] No factor graph set.");
        }

        const auto t_start = std::chrono::steady_clock::now();

        

        // Step 1: Compute residual and Jacobian
        residual_stopwatch_.start();
        graph_->compute_full_residual_vector(residual1_);
        residual_stopwatch_.stop();
        jacobian_stopwatch_.start();
        const Eigen::SparseMatrix<double> &J = graph_->compute_sparse_jacobian_matrix();
        jacobian_stopwatch_.stop();

        // Step 2: Form normal equations
        hessian_stopwatch_.start();
        H_= J.transpose() * J;
        b_.noalias() = J.transpose() * residual1_;
        b_ *= -1.0;
        hessian_stopwatch_.stop();


        // Step 3: Solve H dx = b using sparse LDLT
        //solve_stopwatch_.start();
        solve_stopwatch_.start();
        sparse_solver_.compute(H_);
        if (sparse_solver_.info() != Eigen::Success)
        {
            current_stats_.valid = false;
            current_stats_.status = OptimizerStatus::FAILED;
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Sparse Cholesky factorization failed.\n";
            }
            return;
        }

        // Check for singular/ill-conditioned matrix via diagonal elements
        
        D_ = sparse_solver_.vectorD().cwiseAbs();
        solve_stopwatch_.stop();
        double min_diag = D_.minCoeff();
        double max_diag = D_.maxCoeff();

        if (min_diag < 1e-14)
        {
            current_stats_.valid = false;
            current_stats_.status = OptimizerStatus::SINGULAR_HESSIAN;
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Hessian is singular (min diagonal = " << min_diag << ")\n";
            }
            return;
        }
        
        // Check condition number estimate: max/min diagonal ratio
        double cond_estimate = max_diag / min_diag;
        if (cond_estimate > 1e12)
        {
            current_stats_.valid = false;
            current_stats_.status = OptimizerStatus::ILL_CONDITIONED;
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Hessian is ill-conditioned (condition estimate = "
                          << cond_estimate << ")\n";
            }
            return;
        }

        Eigen::VectorXd dx = sparse_solver_.solve(b_);
        if (sparse_solver_.info() != Eigen::Success)
        {
            current_stats_.valid = false;
            current_stats_.status = OptimizerStatus::FAILED;
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Failed to solve linear system.\n";
            }
            return;
        }

        dx *= settings_.learning_rate;

        // Step 4: Apply increment (todo - push this into the factor graph)
        update_vars_stopwatch_.start();
        const auto &variables = graph_->get_all_variables();
        for (const auto &var : variables)
        {
            if (var->is_constant())
            {
                continue;
            }

            bool valid = false;
            VariablePlacement placement = graph_->variable_placement(var->id(), valid);
            if (!valid)
            {
                std::cerr << "[SparseGN] Skipping variable id=" << var->id() << " (no placement found)\n";
                continue;
            }

            const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }
        update_vars_stopwatch_.stop();

        // Step 5: Compute updated residual and update statistics
        graph_->compute_full_residual_vector(residual2_);
        current_stats_.delta_norm = dx.norm();
        current_stats_.chi2 = residual2_.squaredNorm();
        current_stats_.residual_norm = residual2_.norm();
        current_stats_.current_iteration++;

        // Step 6: Optional verbose logging
        if (settings_.verbose)
        {
            const auto t_end = std::chrono::steady_clock::now();
            auto micros = [](auto duration)
            {
                return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            };

            std::cout << "\n[SparseGN] Iteration " << current_stats_.current_iteration << " Summary:\n";
            std::cout << "  Initial residual norm: " << residual2_.norm() << "\n";
            std::cout << "  Final residual norm:   " << current_stats_.residual_norm << "\n";
            std::cout << "  Step norm (||dx||):    " << current_stats_.delta_norm << "\n";
            std::cout << "  chi²:                  " << current_stats_.chi2 << "\n";
            std::cout << "  J: " << J.rows() << " x " << J.cols()
                      << " | nnz = " << J.nonZeros()
                      << " | density = "
                      << 100.0 * J.nonZeros() / double(J.rows() * J.cols())
                      << " %\n";
            std::cout << "  Total time:            " << micros(t_end - t_start) << " us\n";
        }
    }

    void SparseOptimizer::setup(std::shared_ptr<FactorGraph> graph_ptr,
                                const OptimizerSettings &settings)
    {
        settings_ = settings;
        graph_ = graph_ptr;
        optimization_complete_ = false;
        cached_hessian_valid_for_covariance_est_ = false;
    }

    void SparseOptimizer::optimize()
    {
        if (!graph_)
        {
            throw std::runtime_error("[SparseOptimizer] No factor graph set.");
        }

        // Ensure the graph structure is finalized
        // Note: We assume finalize_structure() was called externally
        // graph_->finalize_structure(); // Uncomment if needed

        // Initialize stats
        current_stats_ = OptimizerStats{};
        current_stats_.valid = true;
        current_stats_.status = OptimizerStatus::RUNNING;
        current_stats_.current_iteration = 0;

        residual_stopwatch_.reset();
        residual_stopwatch_.reset();
        hessian_stopwatch_.reset();
        solve_stopwatch_.reset();
        update_vars_stopwatch_.reset();

        // Store initial residual for convergence checking
        const Eigen::VectorXd initial_residual = graph_->compute_full_residual_vector();
        current_stats_.residual_norm = initial_residual.norm();
        current_stats_.chi2 = initial_residual.squaredNorm();

        initial_stats_ = current_stats_;

        if (settings_.verbose)
        {
            std::cout << "\n=== SparseOptimizer Starting ===\n";
            std::cout << "Algorithm: " << (settings_.method == OptimizerMethod::GaussNewton ? "Gauss-Newton" : "Levenberg-Marquardt") << "\n";
            std::cout << "Max iterations: " << settings_.max_num_iterations << "\n";
            std::cout << "Initial residual norm: " << initial_stats_.residual_norm << "\n";
            std::cout << "Initial chi²: " << initial_stats_.chi2 << "\n";
            std::cout << "Step tolerance: " << settings_.step_tolerance << "\n";
            std::cout << "Residual tolerance: " << settings_.residual_tolerance << "\n";
            std::cout << "Learning rate: " << settings_.learning_rate << "\n";
            std::cout << "=================================\n";
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Main optimization loop (following SPARSE METHOD 2 pattern)
        for (size_t i = 0; i < settings_.max_num_iterations; ++i)
        {
            bool verbose_iteration = settings_.verbose && (i < 2 || i == settings_.max_num_iterations - 1);

            if (verbose_iteration)
            {
                std::cout << "\n#### Iteration " << i << " ####\n";
            }

            // Store previous stats for convergence checking
            auto prev_stats = current_stats_;

            // Perform one optimization step based on selected method
            try
            {
                if (settings_.method == OptimizerMethod::GaussNewton)
                {
                    single_step_gauss_newton();
                }
                else
                {
                    single_step_levenberg_marquardt();
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[SparseOptimizer] Error in iteration " << i << ": " << e.what() << std::endl;
                current_stats_.valid = false;
                current_stats_.status = OptimizerStatus::FAILED;
                break;
            }

            // Check if step failed due to matrix issues
            if (current_stats_.status == OptimizerStatus::SINGULAR_HESSIAN ||
                current_stats_.status == OptimizerStatus::ILL_CONDITIONED ||
                current_stats_.status == OptimizerStatus::FAILED)
            {
                if (settings_.verbose)
                {
                    std::cout << "[SparseOptimizer] Stopping optimization due to error status.\n";
                }
                break;
            }

            // Check convergence based on step norm
            if (current_stats_.delta_norm < settings_.step_tolerance)
            {
                if (settings_.verbose)
                {
                    std::cout << "[SparseOptimizer] Converged: Step norm (" << current_stats_.delta_norm
                              << ") below tolerance (" << settings_.step_tolerance << ")\n";
                }
                current_stats_.status = OptimizerStatus::SUCCESS;
                break;
            }

            // Check convergence based on residual improvement
            double residual_improvement = prev_stats.residual_norm - current_stats_.residual_norm;

            if (residual_improvement < 0.0)
            {
                // TODO: revert the update if the residual went up
                if (settings_.verbose)
                {
                    std::cout << "[SparseOptimizer] Aborted: residual went up - residual improvement: " << residual_improvement << std::endl;
                }
                // Check if we've diverged beyond the initial state
                if (current_stats_.residual_norm >= initial_stats_.residual_norm)
                {
                    current_stats_.status = OptimizerStatus::DIVERGED;
                }
                else
                {
                    current_stats_.status = OptimizerStatus::SUCCESS;
                }
                break;
            }
            else if (residual_improvement < settings_.residual_tolerance)
            {
                if (settings_.verbose)
                {
                    std::cout << "[SparseOptimizer] Converged: Residual improvement (" << residual_improvement
                              << ") below tolerance (" << settings_.residual_tolerance << ")\n";
                }
                current_stats_.status = OptimizerStatus::SUCCESS;
                break;
            }

            // Check for divergence
            if (current_stats_.residual_norm > 10.0 * initial_stats_.residual_norm)
            {
                if (settings_.verbose)
                {
                    std::cout << "[SparseOptimizer] Diverged: Residual norm too large (" << current_stats_.residual_norm
                              << " > 10 * " << initial_stats_.residual_norm << ")\n";
                }
                current_stats_.valid = false;
                current_stats_.status = OptimizerStatus::DIVERGED;
                break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end_time - start_time;

        // Set final status if still RUNNING (max iterations reached)
        if (current_stats_.status == OptimizerStatus::RUNNING)
        {
            current_stats_.status = OptimizerStatus::SUCCESS;
        }

        if (settings_.verbose)
        {
            std::vector<std::string> timer_names = {
                "residual",
                "jacobian",
                "hessian",
                "solve",
                "update_vars"
            };

            std::vector<double> timer_vals = {
                residual_stopwatch_.elapsed(),
                jacobian_stopwatch_.elapsed(),
                hessian_stopwatch_.elapsed(),
                solve_stopwatch_.elapsed(),
                update_vars_stopwatch_.elapsed()
            };


            std::cout << "\n=== SparseOptimizer Summary ===\n";
            std::cout << "Total iterations: " << current_stats_.current_iteration << "\n";
            std::cout << "Final residual norm: " << current_stats_.residual_norm << "\n";
            std::cout << "Final chi²: " << current_stats_.chi2 << "\n";
            std::cout << "Final step norm: " << current_stats_.delta_norm << "\n";
            std::cout << "Residual improvement: " << (initial_stats_.residual_norm - current_stats_.residual_norm) << "\n";
            std::cout << "Relative improvement: " << (initial_stats_.residual_norm > 0 ? (initial_stats_.residual_norm - current_stats_.residual_norm) / initial_stats_.residual_norm * 100.0 : 0.0)
                      << " %\n";
            std::cout << "Total time: " << elapsed_sec.count() << " seconds\n";
            std::cout << "Detailed times:" << std::endl;
            for(size_t i = 0; i < timer_names.size(); i++) {
                std::cout << "  " << timer_names[i] << ": " << timer_vals[i] * 1e6 << "us" << std::endl;
            }
            std::cout << "Optimization " << (current_stats_.valid ? "SUCCESS" : "FAILED") << "\n";
            std::cout << "Status: " << static_cast<int>(current_stats_.status) << "\n";
            std::cout << "===============================\n";
        }

        optimization_complete_ = true;
    }

    void SparseOptimizer::single_step_levenberg_marquardt()
    {
        if (!graph_)
        {
            throw std::runtime_error("[SparseOptimizer] No factor graph set.");
        }

        const auto t_start = std::chrono::steady_clock::now();

        // Step 1: Compute residual and Jacobian
        residual_stopwatch_.start();
        graph_->compute_full_residual_vector(residual1_);
        residual_stopwatch_.stop();

        jacobian_stopwatch_.start();
        const Eigen::SparseMatrix<double> &J = graph_->compute_sparse_jacobian_matrix();
        jacobian_stopwatch_.stop();

        double current_cost = 0.5 * residual1_.squaredNorm();

        // Initialize damping parameter if this is the first iteration
        if (current_stats_.current_iteration == 0)
        {
            current_stats_.damping_parameter = settings_.initial_lambda;
        }

        bool step_accepted = false;
        Eigen::VectorXd dx;
        double new_cost = current_cost;
        int lm_attempts = 0;
        const int max_lm_attempts = 5;

        // Levenberg-Marquardt loop: try different damping values until we get improvement
        while (!step_accepted && lm_attempts < max_lm_attempts)
        {
            // Step 2: Form damped normal equations: (J^T J + λI) dx = -J^T r
            hessian_stopwatch_.start();
            Eigen::SparseMatrix<double> H = J.transpose() * J;

            // Add damping: H = J^T J + λI
            for (int i = 0; i < H.outerSize(); ++i)
            {
                H.coeffRef(i, i) += current_stats_.damping_parameter;
            }

            Eigen::VectorXd b = -J.transpose() * residual1_;
            hessian_stopwatch_.stop();

            // Step 3: Solve damped system
            solve_stopwatch_.start();
            sparse_solver_.compute(H);
            if (sparse_solver_.info() != Eigen::Success)
            {
                // If factorization fails, increase damping and try again
                current_stats_.damping_parameter *= settings_.lambda_up_factor;
                lm_attempts++;
                if (settings_.verbose)
                {
                    std::cout << "[LM] Factorization failed, increasing lambda to " << current_stats_.damping_parameter << "\n";
                }
                continue;
            }

            // Check for singular/ill-conditioned matrix via diagonal elements
            Eigen::VectorXd D = sparse_solver_.vectorD().cwiseAbs();
            double min_diag = D.minCoeff();
            double max_diag = D.maxCoeff();

            if (min_diag < 1e-14 || (max_diag / min_diag) > 1e12)
            {
                // Hessian is ill-conditioned, increase damping and try again
                current_stats_.damping_parameter *= settings_.lambda_up_factor;
                lm_attempts++;
                if (settings_.verbose)
                {
                    std::cout << "[LM] Hessian is "
                              << (min_diag < 1e-14 ? "singular" : "ill-conditioned")
                              << " (min diag=" << min_diag << ", cond est=" << (max_diag/min_diag)
                              << "), increasing lambda to " << current_stats_.damping_parameter << "\n";
                }
                continue;
            }

            dx = sparse_solver_.solve(b);
            if (sparse_solver_.info() != Eigen::Success)
            {
                current_stats_.damping_parameter *= settings_.lambda_up_factor;
                lm_attempts++;
                if (settings_.verbose)
                {
                    std::cout << "[LM] Solve failed, increasing lambda to " << current_stats_.damping_parameter << "\n";
                }
                continue;
            }
            solve_stopwatch_.stop();

            dx *= settings_.learning_rate;

            // Step 4: Apply increment to variables (temporarily)
            update_vars_stopwatch_.start();
            std::vector<Eigen::VectorXd> original_values;
            const auto &variables = graph_->get_all_variables();

            // Store original values for potential rollback
            for (const auto &var : variables)
            {
                original_values.push_back(var->value());
            }

            // Apply increments
            for (const auto &var : variables)
            {
                if (var->is_constant())
                {
                    continue;
                }

                bool valid = false;
                VariablePlacement placement = graph_->variable_placement(var->id(), valid);
                if (!valid)
                {
                    std::cerr << "[LM] Skipping variable id=" << var->id() << " (no placement found)\n";
                    continue;
                }

                const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
                var->apply_increment(local_dx);
            }
            update_vars_stopwatch_.stop();

            // Step 5: Evaluate new cost
            residual_stopwatch_.start();
            graph_->compute_full_residual_vector(residual2_);
            residual_stopwatch_.stop();
            new_cost = 0.5 * residual2_.squaredNorm();

            // Step 6: Check if step should be accepted
            if (new_cost < current_cost)
            {
                // Accept step: decrease damping for next iteration
                step_accepted = true;
                current_stats_.damping_parameter *= settings_.lambda_down_factor;
                current_stats_.damping_parameter = std::max(current_stats_.damping_parameter, 1e-12);

                if (settings_.verbose)
                {
                    std::cout << "[LM] Step accepted, cost: " << current_cost << " -> " << new_cost
                              << ", decreasing lambda to " << current_stats_.damping_parameter << "\n";
                }
            }
            else
            {
                update_vars_stopwatch_.start();
                // Reject step: restore original values and increase damping
                for (size_t i = 0; i < variables.size(); ++i)
                {
                    variables[i]->set_value_from_vector(original_values[i]);
                }

                current_stats_.damping_parameter *= settings_.lambda_up_factor;
                current_stats_.damping_parameter = std::min(current_stats_.damping_parameter, settings_.max_lambda);
                lm_attempts++;

                if (settings_.verbose)
                {
                    std::cout << "[LM] Step rejected, cost: " << current_cost << " -> " << new_cost
                              << ", increasing lambda to " << current_stats_.damping_parameter << "\n";
                }

                // If lambda becomes too large, we're stuck
                if (current_stats_.damping_parameter >= settings_.max_lambda)
                {
                    if (settings_.verbose)
                    {
                        std::cout << "[LM] Lambda too large (" << current_stats_.damping_parameter
                                  << " >= " << settings_.max_lambda << "), stopping\n";
                    }
                    break;
                }
                update_vars_stopwatch_.stop();
            }
        }

        // If no step was accepted, we're done (convergence or failure)
        if (!step_accepted)
        {   

            double initial_cost = initial_stats_.chi2 * 0.5;
            if(current_cost < initial_cost) {
                // cost has gone down since the beginning, so we will call this a win
                current_stats_.valid = true;
                current_stats_.status = OptimizerStatus::SUCCESS;

            }
            else {
                current_stats_.valid = false;
                current_stats_.status = (current_stats_.damping_parameter >= settings_.max_lambda)
                                        ? OptimizerStatus::ILL_CONDITIONED
                                        : OptimizerStatus::FAILED;
            }
            
            if (settings_.verbose)
            {
                std::cout << "[LM] No acceptable step found after " << max_lm_attempts << " attempts\n";
            }
            // Set minimal step for convergence detection
            dx = Eigen::VectorXd::Zero(dx.size());
            new_cost = current_cost;
        }

        // Step 7: Update statistics (use final accepted state)
        Eigen::VectorXd& r_final = residual2_;
        if(!step_accepted) {
            r_final = residual1_;
        }
        current_stats_.delta_norm = dx.norm();
        current_stats_.chi2 = r_final.squaredNorm();
        current_stats_.residual_norm = r_final.norm();
        current_stats_.current_iteration++;

        // Step 8: Optional verbose logging
        if (settings_.verbose)
        {
            const auto t_end = std::chrono::steady_clock::now();
            auto micros = [](auto duration)
            {
                return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            };

            std::cout << "\n[LM] Iteration " << current_stats_.current_iteration << " Summary:\n";
            std::cout << "  Initial residual norm: " << residual1_.norm() << "\n";
            std::cout << "  Final residual norm:   " << current_stats_.residual_norm << "\n";
            std::cout << "  Step norm (||dx||):    " << current_stats_.delta_norm << "\n";
            std::cout << "  chi²:                  " << current_stats_.chi2 << "\n";
            std::cout << "  Damping (λ):           " << current_stats_.damping_parameter << "\n";
            std::cout << "  LM attempts:           " << lm_attempts + 1 << "\n";
            std::cout << "  Step accepted:         " << (step_accepted ? "YES" : "NO") << "\n";
            std::cout << "  J: " << J.rows() << " x " << J.cols()
                      << " | nnz = " << J.nonZeros()
                      << " | density = "
                      << 100.0 * J.nonZeros() / double(J.rows() * J.cols())
                      << " %\n";
            std::cout << "  Total time:            " << micros(t_end - t_start) << " us\n";
        }
    }

    void SparseOptimizer::prepare_to_estimate_covariances()
    {
        // Assume the graph has already been optimized - estimate the covariance of each variable

        if (cached_hessian_valid_for_covariance_est_)
        {
            // early end, if it was already valid
            return;
        }

        if(!optimization_complete_) {

            return;
        }

        const auto t_start = std::chrono::steady_clock::now();

        // Step 1: Get the sparse jacobian
        const Eigen::SparseMatrix<double> &J = graph_->sparse_jacobian();

        // Step 2: compute Hessian
        cached_hessian_ = J.transpose() * J;

        sparse_solver_.compute(cached_hessian_);

        if (sparse_solver_.info() != Eigen::Success)
        {
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Sparse Cholesky factorization failed for covariance estimation.\n";
            }
            return;
        }

        // Check for singular/ill-conditioned matrix via diagonal elements
        Eigen::VectorXd D = sparse_solver_.vectorD().cwiseAbs();
        double min_diag = D.minCoeff();
        double max_diag = D.maxCoeff();
        double cond_estimate = max_diag / min_diag;

        if (min_diag < 1e-14 || cond_estimate > 1e12)
        {
            if (settings_.verbose)
            {
                std::cerr << "[SparseOptimizer] Hessian is "
                          << (min_diag < 1e-14 ? "singular" : "ill-conditioned")
                          << " (min diag=" << min_diag << ", cond est=" << cond_estimate
                          << ") - covariance estimation may be unreliable.\n";
            }
            // Note: We still allow covariance estimation to proceed, but warn the user
        }

        cached_hessian_valid_for_covariance_est_ = true;

        if (settings_.verbose)
        {

            const auto t_end = std::chrono::steady_clock::now();
            auto micros = [](auto duration)
            {
                return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            };
            std::cout << "estimate covariance" << std::endl;
            std::cout << "  Total time:            " << micros(t_end - t_start) << " us\n";
        }
    }

    Eigen::MatrixXd SparseOptimizer::estimate_covariance(const Variable *variable, bool &covariance_valid)
    {
        covariance_valid = false;
        if (variable == nullptr)
        {
            if (settings_.verbose)
            {
                std::cerr << "get covariance got an empty variable" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        if (variable->is_constant() || variable->size() == 0)
        {
            if (settings_.verbose)
            {
                std::cerr << "cannot get a covariance from a constant variable or one with no size" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        if(!optimization_complete_) {
            if (settings_.verbose)
            {
                std::cerr << "optimization was not complete prior to est" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        if (!cached_hessian_valid_for_covariance_est_ || cached_hessian_.rows() == 0 || cached_hessian_.cols() == 0)
        {
            if (settings_.verbose)
            {
                std::cerr << "cached hessian was invalid" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        int var_id = variable->id();

        bool var_valid;
        auto placement = graph_->variable_placement(var_id, var_valid);

        if (!var_valid)
        {
            if (settings_.verbose)
            {
                std::cerr << "variable placement not found" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        // Given the placement, grab the indices into the hessian
        int total_hessian_size = cached_hessian_.rows();
        int variable_size = placement.dim;
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(total_hessian_size, variable_size);

        for (int i = 0; i < variable_size; i++)
        {
            B(placement.index + i, i) = 1.0;
        }
        Eigen::MatrixXd X = sparse_solver_.solve(B);

        if (sparse_solver_.info() != Eigen::Success)
        {
            if (settings_.verbose)
            {
                std::cerr << "get covariance for variable " + variable->name() + ", solve failed" << std::endl;
            }
            return Eigen::MatrixXd();
        }

        Eigen::MatrixXd P = X.block(placement.index, 0, variable_size, variable_size);

        if (!P.allFinite())
        {
            return Eigen::MatrixXd();
        }

        double max_diag = P.diagonal().maxCoeff();

        if (max_diag > 1e12)
        {
            return Eigen::MatrixXd();
        }

        // TODO: Check P for nan, inf, and outlandishly large values.

        covariance_valid = true;
        // Normalize P to be symmetric
        P = 0.5 * (P + P.transpose());
        return P;
    }

    void SparseOptimizer::print_all_covariances()
    {
        if (!cached_hessian_valid_for_covariance_est_)
        {
            prepare_to_estimate_covariances();
        }

        std::cout << "\n\nPrinting out all covariances for the factor graph" << std::endl;

        for (auto var : graph_->get_all_variables())
        {
            std::cout << "variable: " + var->name() << std::endl;
            if (var->is_constant())
            {
                std::cout << "constant" << std::endl;
                continue;
            }
            bool cov_valid;
            auto cov = estimate_covariance(var.get(), cov_valid);

            if (cov_valid)
            {
                std::cout << cov << std::endl;
            }
            else
            {
                std::cout << "invalid" << std::endl;
            }
        }
    }

} // namespace factorama
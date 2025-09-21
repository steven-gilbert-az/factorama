#pragma once
#include <cstddef>

#include "factorama/factor_graph.hpp"

namespace factorama
{

  /**
   * @brief Optimization algorithm selection
   */
  enum class OptimizerMethod
  {
    GaussNewton,        ///< Gauss-Newton method (faster, requires good initialization)
    LevenbergMarquardt  ///< Levenberg-Marquardt method (more robust, handles poor initialization)
  };

  /**
   * @brief Configuration settings for sparse optimization
   *
   * Controls convergence criteria, algorithm selection, and damping parameters.
   * See README.md for detailed usage examples.
   */
  struct OptimizerSettings
  {
    // Algorithm to use (GN or LM)
    OptimizerMethod method = OptimizerMethod::GaussNewton;

    // Maximum number of optimization iterations
    size_t max_num_iterations = 100;

    // Convergence threshold on the step norm (||dx||)
    double step_tolerance = 1e-6;

    // Convergence threshold on residual norm improvement
    double residual_tolerance = 1e-6;

    // Initial damping parameter for LM
    double initial_lambda = 1e-3;

    // Maximum allowed lambda for LM (prevents runaway)
    double max_lambda = 1e5;

    // Damping adjustment multiplier
    double lambda_up_factor = 10.0;
    double lambda_down_factor = 0.1;

    // Optional learning rate (for GN) — 1.0 = full step
    double learning_rate = 1.0;

    // Enable verbose logging
    bool verbose = false;

    // Enable rank deficiency check (may cost extra time)
    bool check_rank_deficiency = false;
  };

  /**
   * @brief Runtime statistics from optimization iterations
   *
   * Tracks convergence metrics and algorithm state during optimization.
   */
  struct OptimizerStats
  {
    bool valid = false;                  ///< Whether these statistics are valid
    double chi2 = -1.0;                  ///< Chi-squared cost (||residual||^2)
    double delta_norm = -1.0;            ///< Norm of optimization step
    double residual_norm = -1.0;         ///< Norm of residual vector
    int current_iteration = 0;           ///< Current iteration number
    int rank = -1;                       ///< Matrix rank (if computed)

    double damping_parameter = 1e-3;     ///< Levenberg-Marquardt damping parameter (lambda)
  };

  /**
   * @brief Sparse non-linear least squares optimizer
   *
   * Implements Gauss-Newton and Levenberg-Marquardt algorithms using Eigen's sparse
   * linear algebra. Designed to work with FactorGraph for bundle adjustment and SLAM problems.
   *
   * @code
   * SparseOptimizer optimizer;
   * OptimizerSettings settings;
   * settings.method = OptimizerMethod::LevenbergMarquardt;
   * optimizer.setup(graph_ptr, settings);
   * optimizer.optimize();
   * @endcode
   */
  class SparseOptimizer
  {
  public:
    SparseOptimizer() = default;

    /**
     * @brief Configure optimizer with factor graph and settings
     * @param graph_ptr Shared pointer to finalized FactorGraph
     * @param settings Optimization configuration (algorithm, tolerances, etc.)
     */
    void setup(std::shared_ptr<FactorGraph> graph_ptr,
               const OptimizerSettings &settings);

    /**
     * @brief Run optimization until convergence or max iterations
     * Updates variable values in the associated FactorGraph
     */
    void optimize();

    /**
     * @brief Get current optimizer settings
     * @return Reference to optimization settings
     */
    const OptimizerSettings &settings() const { return settings_; }

    OptimizerStats initial_stats_;
    OptimizerStats current_stats_;

  private:
    // ---- Core Settings and Graph ----
    OptimizerSettings settings_;
    std::shared_ptr<FactorGraph> graph_;

    // we could also save off a std::vector<OptimizerStats> for logging / telemetry

    // ---- Cached Jacobian Pattern ----
    bool jacobian_pattern_initialized_ = false;
    Eigen::SparseMatrix<double> J_pattern_; // Only structure (non-zero pattern)

    // Optional reuse of solver object (may reduce overhead)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparse_solver_;

    // ---- Optimization Step Functions ----
    void single_step_gauss_newton();
    void single_step_levenberg_marquardt();

    // Internal utilities (e.g., convergence check, logging, damping updates)
    bool check_convergence();
    void print_iteration_summary();
  };

}
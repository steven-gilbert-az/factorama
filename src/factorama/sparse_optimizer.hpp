#pragma once
#include <cstddef>

#include "factorama/factor_graph.hpp"
#include "factorama/stopwatch.hpp"

namespace factorama
{

  /**
   * @brief Optimization algorithm selection
   */
  enum class OptimizerMethod
  {
    GaussNewton,       ///< Gauss-Newton method (faster, requires good initialization)
    LevenbergMarquardt ///< Levenberg-Marquardt method (more robust, handles poor initialization)
  };

  /**
   * @brief Optimization status codes
   */
  enum class OptimizerStatus
  {
    SUCCESS,           ///< Optimization completed successfully (converged or max iterations)
    RUNNING,           ///< Optimization is in progress
    SINGULAR_HESSIAN,  ///< Hessian matrix is singular (not invertible)
    ILL_CONDITIONED,   ///< Hessian matrix is ill-conditioned (nearly singular)
    DIVERGED,          ///< Optimization diverged (residual growing)
    FAILED             ///< Generic failure (e.g., factorization or solve failed)
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
    bool valid = false;                           ///< Whether these statistics are valid
    OptimizerStatus status = OptimizerStatus::RUNNING; ///< Current optimization status
    double chi2 = -1.0;                           ///< Chi-squared cost (||residual||^2)
    double delta_norm = -1.0;                     ///< Norm of optimization step
    double residual_norm = -1.0;                  ///< Norm of residual vector
    int current_iteration = 0;                    ///< Current iteration number
    int rank = -1;                                ///< Matrix rank (if computed)

    double damping_parameter = 1e-3;              ///< Levenberg-Marquardt damping parameter (lambda)
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
     * @brief Compute and cache Hessian for covariance estimation
     */
    void prepare_to_estimate_covariances();

    /**
     * @brief Estimate covariance matrix for a specific variable
     * @param variable Variable to compute covariance for
     * @param valid_out Set to true if computation succeeded
     * @return Covariance matrix (empty if invalid)
     */
    Eigen::MatrixXd estimate_covariance(const Variable *variable, bool &valid_out);

    /**
     * @brief Print covariance matrices for all variables in the graph
     */
    void print_all_covariances();

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

    bool optimization_complete_ = false;

    // ---- Cached Jacobian Pattern ----
    bool jacobian_pattern_initialized_ = false;
    Eigen::SparseMatrix<double> J_pattern_; // Only structure (non-zero pattern)

    // Cached Hessian (for covariance estimates)
    bool cached_hessian_valid_for_covariance_est_ = false;
    Eigen::SparseMatrix<double> cached_hessian_;

    // Cached Residuals
    Eigen::VectorXd residual1_;
    Eigen::VectorXd residual2_;
    Eigen::SparseMatrix<double> H_;
    Eigen::VectorXd b_;
    Eigen::VectorXd D_;


    // Optional reuse of solver object (may reduce overhead)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparse_solver_;
    //Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> sparse_solver2_;

    // ---- Optimization Step Functions ----
    void single_step_gauss_newton();
    void single_step_levenberg_marquardt();

    // Internal utilities (e.g., convergence check, logging, damping updates)
    bool check_convergence();
    void print_iteration_summary();


    // Stopwatches for gathering runtime profiling data
    Stopwatch residual_stopwatch_;
    Stopwatch jacobian_stopwatch_;
    Stopwatch hessian_stopwatch_;
    Stopwatch solve_stopwatch_;
    Stopwatch update_vars_stopwatch_;



  };

}
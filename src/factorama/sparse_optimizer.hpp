#pragma once
#include <cstddef>

#include "factorama/factor_graph.hpp"

namespace factorama
{

  enum class OptimizerMethod
  {
    GaussNewton,
    LevenbergMarquardt
  };

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

  struct OptimizerStats
  {
    bool valid = false;
    double chi2 = -1.0;
    double delta_norm = -1.0;
    double residual_norm = -1.0;
    int current_iteration = 0;
    int rank = -1;

    // Levenberg-Marquardt only
    double damping_parameter = 1e-3; // aka lambda
  };

  class SparseOptimizer
  {
  public:
    SparseOptimizer() = default;

    // Setup with factor graph and settings
    void setup(std::shared_ptr<FactorGraph> graph_ptr,
               const OptimizerSettings &settings);

    // Run full optimization
    void optimize();

    // Access the current settings
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
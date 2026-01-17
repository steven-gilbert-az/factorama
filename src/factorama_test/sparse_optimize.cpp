// === Includes ===
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky> // For SimplicialLDLT
#include <iostream>
#include <iomanip>
#include <chrono>
#include "factorama/factor_graph.hpp"
#include "factorama/base_types.hpp"
#include "factorama_test/test_utils.hpp"


namespace factorama
{

    struct GaussNewtonStats
    {
        double initial_chi2 = -1.0;
        double dx_norm = -1.0;
        int rank = -1;
    };

    GaussNewtonStats run_sparse_gauss_newton_iteration(FactorGraph& graph, bool verbose = false,
                                                       double learning_rate = 1.0)
    {
        graph.compute_full_jacobian_and_residual();
        const Eigen::MatrixXd& J_dense = graph.jacobian();
        const Eigen::VectorXd& r = graph.residual();

        const double initial_residual_norm = r.norm();
        const double chi2 = r.squaredNorm();

        // === Convert to sparse ===
        Eigen::SparseMatrix<double> J_sparse = J_dense.sparseView();

        // === Normal equations ===
        Eigen::SparseMatrix<double> H = J_sparse.transpose() * J_sparse;
        Eigen::VectorXd b = -J_sparse.transpose() * r;

        // === Solve H * dx = b using Eigen’s sparse LDLT ===
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        Eigen::VectorXd dx = solver.solve(b) * learning_rate;

        if (solver.info() != Eigen::Success) {
            std::cerr << "[SparseGaussNewton] Warning: Solver failed or matrix may be singular.\n";
        }

        // const int rank = solver.rank(); // Note: Only available in Cholmod-based solvers
        // const int rank = 0;
        const double dx_norm = dx.norm();

        // === Apply increments ===
        const auto& variables = graph.get_all_variables();
        for (const auto& var : variables) {
            if (var->is_constant())
                continue;

            bool valid = false;
            auto placement = graph.variable_placement(var->id(), valid);
            if (!valid) {
                std::cerr << "[SparseGaussNewton] Skipping variable id=" << var->id() << " (no placement)\n";
                continue;
            }

            Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }

        // === Final diagnostics ===
        auto r_opt = graph.compute_full_residual_vector();
        const double final_residual_norm = r_opt.norm();

        if (verbose) {
            std::cout << "[SparseGaussNewton] chi²: " << chi2 << " | ||dx||: " << dx_norm
                      << " | Initial residual norm: " << initial_residual_norm
                      << " | Final residual norm: " << final_residual_norm
                      << " | Δnorm: " << (initial_residual_norm - final_residual_norm) << "\n";
        }

        return GaussNewtonStats{
            chi2,    // initial_chi2
            dx_norm, // dx_norm
            -1       // rank - Eigen::SimplicialLDLT doesn't expose rank
        };
    }

    GaussNewtonStats run_gauss_newton_iteration(FactorGraph& graph, bool verbose = false, double learning_rate = 1.0)
    {
        // Step 1: Compute residual and Jacobian
        graph.compute_full_jacobian_and_residual();
        const Eigen::MatrixXd& J = graph.jacobian();
        const Eigen::VectorXd& r = graph.residual();

        double initial_residual_norm = r.norm();

        // Step 2: Form normal equations
        const Eigen::MatrixXd H = J.transpose() * J;  // Hessian approximation
        const Eigen::VectorXd b = -J.transpose() * r; // Right-hand side

        // Step 3: Solve H dx = b
        Eigen::VectorXd dx;
        Eigen::FullPivLU<Eigen::MatrixXd> solver(H);
        dx = solver.solve(b) * learning_rate;

        if (!solver.isInvertible()) {
            std::cerr << "[GaussNewton] Warning: Hessian is rank deficient!\n";
        }

        int rank = solver.rank();
        double chi2 = r.squaredNorm();
        double dx_norm = dx.norm();

        // Step 4: Apply increment to each variable
        const auto& variables = graph.get_all_variables();
        for (const auto& var : variables) {
            if (var->is_constant())
                continue;

            bool valid = false;
            VariablePlacement placement = graph.variable_placement(var->id(), valid);
            if (!valid) {
                std::cerr << "[GaussNewton] Skipping variable id=" << var->id() << " (no placement found)\n";
                continue;
            }

            const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }
        auto r_opt = graph.compute_full_residual_vector();
        double final_residual_norm = r_opt.norm();
        if (verbose) {
            std::cout << "[GaussNewton] chi²: " << chi2 << " | ||dx||: " << dx_norm << " | rank: " << rank << "\n";

            std::cout << "[GaussNewton] Initial residual norm: " << initial_residual_norm
                      << " | Final residual norm: " << final_residual_norm << "\n";
            std::cout << "Delta norm: " << (initial_residual_norm - final_residual_norm) << "\n";
        }

        return GaussNewtonStats{
            chi2,    // initial_chi2
            dx_norm, // dx_norm
            rank     // rank
        };
    }

    GaussNewtonStats run_sparser_gauss_newton_iteration(FactorGraph& graph, bool verbose = false,
                                                        double learning_rate = 1.0, bool check_rank_deficiency = false)
    {
        // Step 1: Compute residual and sparse Jacobian
        auto t0 = GetMonotonicSeconds();
        const Eigen::VectorXd& r = graph.compute_full_residual_vector();
        const Eigen::SparseMatrix<double>& J = graph.compute_sparse_jacobian_matrix();
        auto t1 = GetMonotonicSeconds();

        if (J.rows() < J.cols()) {
            throw std::runtime_error("Sparse Gauss-Newton: Jacobian is not tall (rows < cols), which may lead to "
                                     "ill-posed normal equations.");
        }

        double initial_residual_norm = r.norm();

        // Step 2: Form normal equations
        auto t2 = GetMonotonicSeconds();
        const Eigen::SparseMatrix<double> H = J.transpose() * J; // Hessian approximation
        auto t3 = GetMonotonicSeconds();
        const Eigen::VectorXd b = -J.transpose() * r; // RHS
        auto t4 = GetMonotonicSeconds();

        if (verbose) {

            bool verboser = false;


            std::cout << "\n====== [Verbose] Sparse Jacobian Debug ======\n";

            std::cout << "[J] dimensions: " << J.rows() << " x " << J.cols() << "\n";
            std::cout << "[J] Nonzeros: " << J.nonZeros() << "\n";
            std::cout << "[J] Density: " << 100.0 * J.nonZeros() / (double(J.rows()) * J.cols()) << " %\n";


            if (verboser) {
                Eigen::MatrixXd J_dense = Eigen::MatrixXd(J);
                std::cout << "[J] Dense contents:\n" << J_dense << "\n";

                // Build the Hessian manually in verbose mode to get it printed
                // Eigen::SparseMatrix<double> H(sparse_jacobian_.cols(), sparse_jacobian_.cols());
                // H = sparse_jacobian_.transpose() * sparse_jacobian_;

                std::cout << "\n[H] dimensions: " << H.rows() << " x " << H.cols() << "\n";
                std::cout << "[H] Nonzeros: " << H.nonZeros() << "\n";
                std::cout << "[H] Density: " << 100.0 * H.nonZeros() / (double(H.rows()) * H.cols()) << " %\n";

                Eigen::MatrixXd H_dense = Eigen::MatrixXd(H);
                std::cout << "[H] Dense contents:\n" << H_dense << "\n";

                std::cout << "=============================================\n\n";
            }
        }

        // Step 3: Solve H dx = b using sparse Cholesky
        Eigen::VectorXd dx;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        auto t5 = GetMonotonicSeconds();

        if (check_rank_deficiency) {
            throw std::runtime_error("check_rank_deficiency=true not yet implemented for sparse solver. You could use "
                                     "eigenvalue thresholding or fallback rank estimation.");
        }

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Gauss-Newton: SimplicialLLT failed to factorize the Hessian.");
        }

        dx = solver.solve(b) * learning_rate;
        auto t6 = GetMonotonicSeconds();

        double dx_norm = dx.norm();
        double chi2 = r.squaredNorm();

        // Step 4: Apply dx to variables
        const auto& variables = graph.get_all_variables();
        for (const auto& var : variables) {
            if (var->is_constant())
                continue;

            bool valid = false;
            VariablePlacement placement = graph.variable_placement(var->id(), valid);
            if (!valid) {
                std::cerr << "[SparseGN] Skipping variable id=" << var->id() << " (no placement found)\n";
                continue;
            }

            const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }

        const Eigen::VectorXd r_opt = graph.compute_full_residual_vector();
        double final_residual_norm = r_opt.norm();

        auto t7 = GetMonotonicSeconds();

        if (verbose) {
            auto us = [](double seconds) { return static_cast<long long>(seconds * 1e6); };
            std::cout << "[SparseGN] Time breakdown (microseconds):\n";
            std::cout << "  Residual/Jacobian:   " << std::setw(8) << us(t1 - t0) << " us\n"
                      << "  Hessian:            " << std::setw(8) << us(t3 - t2) << " us\n"
                      << "  RHS:                " << std::setw(8) << us(t4 - t3) << " us\n"
                      << "  Compute:            " << std::setw(8) << us(t5 - t4) << " us\n"
                      << "  Solve:              " << std::setw(8) << us(t6 - t5) << " us\n"
                      << "  Apply increments:   " << std::setw(8) << us(t7 - t6) << " us\n"
                      << "  Total:              " << std::setw(8) << us(t7 - t0) << " us\n";
        }

        if (verbose) {
            std::cout << "[SparseGN] chi²: " << chi2 << " | ||dx||: " << dx_norm << "\n";
            std::cout << "[SparseGN] Initial residual norm: " << initial_residual_norm
                      << " | Final residual norm: " << final_residual_norm << "\n";
            std::cout << "Delta norm: " << (initial_residual_norm - final_residual_norm) << "\n";
        }

        return GaussNewtonStats{
            chi2,    // initial_chi2
            dx_norm, // dx_norm
            -1       // rank - unknown in sparse mode
        };
    }

} // namespace factorama

int main(int argc, char *argv[])
{
    using namespace factorama;

    // Default values
    int num_iterations = 10;
    int num_landmarks = 20;

    double learning_rate = 1.00;
    double sparsity = 0.5; // 1.0 - fully sparse (no connections)

    bool inv_range_vars = false;

    // Parse arguments
    if (argc > 1)
        num_iterations = std::atoi(argv[1]);
    if (argc > 2)
        num_landmarks = std::atoi(argv[2]);

    std::cout << " Using " << num_iterations << " iterations and " << num_landmarks << " landmarks \n";

    std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
    std::vector<Eigen::Vector3d> gt_landmark_positions;

    {
        std::cout << std::endl << std::endl << "##############      DENSE     ##############" << std::endl;
        // CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
        CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);

        auto extra_landmarks =
            CreateLandmarksInVolume(Eigen::Vector3d(10.0, -5.0, -5.0), Eigen::Vector3d(10.0, 5.0, 5.0), num_landmarks);

        gt_landmark_positions.insert(gt_landmark_positions.end(), extra_landmarks.begin(), extra_landmarks.end());
        FactorGraph graph;
        if (!inv_range_vars) {
            graph = CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.04, sparsity);
        } else {
            graph = CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, true);
        }

        graph.finalize_structure();

        std::cout << "Num variables: " << graph.num_variables() << "\n";
        std::cout << "Num values: " << graph.num_values() << "\n";
        std::cout << "Num residuals: " << graph.num_residuals() << "\n";

        std::cout << "Running " << num_iterations << " Dense Gauss-Newton Iterations...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            bool verbose = i == 0;
            run_gauss_newton_iteration(graph, verbose, learning_rate);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end_time - start_time;

        std::cout << "⏱️ Total time for " << num_iterations << " iterations: " << elapsed_sec.count()
                  << " seconds\n";
    }

    {
        std::cout << std::endl << std::endl << "############## SPARSE METHOD 1 ##############" << std::endl;
        // CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
        CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
        auto extra_landmarks =
            CreateLandmarksInVolume(Eigen::Vector3d(10.0, -5.0, -5.0), Eigen::Vector3d(10.0, 5.0, 5.0), num_landmarks);

        gt_landmark_positions.insert(gt_landmark_positions.end(), extra_landmarks.begin(), extra_landmarks.end());

        FactorGraph graph;
        if (!inv_range_vars) {
            graph = CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.04,
                                             1.0 - sparsity);
        } else {
            graph = CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, true);
        }
        graph.finalize_structure();

        std::cout << "Num variables: " << graph.num_variables() << "\n";
        std::cout << "Num values: " << graph.num_values() << "\n";
        std::cout << "Num residuals: " << graph.num_residuals() << "\n";

        std::cout << "Running " << num_iterations << " Sparse Gauss-Newton Iterations...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            bool verbose = i == 0;
            run_sparse_gauss_newton_iteration(graph, verbose, learning_rate);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end_time - start_time;

        std::cout << "⏱️ Total time for " << num_iterations << " iterations: " << elapsed_sec.count()
                  << " seconds\n";
    }

    {
        std::cout << std::endl << std::endl << "############## SPARSE METHOD 2 ##############" << std::endl;
        CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
        auto extra_landmarks =
            CreateLandmarksInVolume(Eigen::Vector3d(10.0, -5.0, -5.0), Eigen::Vector3d(10.0, 5.0, 5.0), num_landmarks);

        gt_landmark_positions.insert(gt_landmark_positions.end(), extra_landmarks.begin(), extra_landmarks.end());

        FactorGraph graph;
        if (!inv_range_vars) {
            graph = CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.04,
                                             1.0 - sparsity);
        } else {
            graph = CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, true);
        }
        graph.finalize_structure();

        std::cout << "Running " << num_iterations << " Sparsiest Gauss-Newton Iterations...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            bool verbose = i < 2;

            if (verbose) {
                std::cout << "####   Iteration " << i << "  #####" << std::endl;

                compare_dense_and_sparse_jacobians(graph, 1e-8);
            }
            run_sparser_gauss_newton_iteration(graph, verbose, learning_rate);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sec = end_time - start_time;

        std::cout << "⏱️ Total time for " << num_iterations << " iterations: " << elapsed_sec.count()
                  << " seconds\n";
    }

    return 0;
}
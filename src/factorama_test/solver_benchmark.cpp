#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "factorama/sparse_optimizer.hpp"
#include "factorama/factor_graph.hpp"
#include "factorama/bearing_projection_factor_2d.hpp"
#include "factorama_test/test_utils.hpp"

using namespace factorama;

// Create a more challenging scenario with many cameras and landmarks
void CreateLargeScenario(std::vector<Eigen::Matrix<double, 6, 1>>& camera_poses, 
                        std::vector<Eigen::Vector3d>& landmark_positions)
{
    // Create a circular trajectory of cameras looking inward at landmarks
    const int num_cameras = 8;
    const int num_landmarks = 25;
    const double radius = 10.0;
    const double height = 2.0;
    
    camera_poses.clear();
    landmark_positions.clear();
    
    // Generate cameras in a circle
    for (int i = 0; i < num_cameras; ++i) {
        double angle = 2.0 * M_PI * i / num_cameras;
        Eigen::Matrix<double, 6, 1> pose;
        
        // Position: circular trajectory at height
        pose[0] = radius * cos(angle);  // x
        pose[1] = radius * sin(angle);  // y 
        pose[2] = height;               // z
        
        // Orientation: look toward center with slight variations
        pose[3] = 0.1 * sin(angle);     // roll
        pose[4] = 0.1 * cos(angle);     // pitch
        pose[5] = angle + M_PI;         // yaw (face inward)
        
        camera_poses.push_back(pose);
    }
    
    // Generate landmarks in a 5x5 grid at center
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            Eigen::Vector3d lm;
            lm[0] = -4.0 + 2.0 * x;  // x: -4 to +4
            lm[1] = -4.0 + 2.0 * y;  // y: -4 to +4  
            lm[2] = 0.5 * sin(x) + 0.5 * cos(y);  // z: slight terrain variation
            landmark_positions.push_back(lm);
        }
    }
    
    std::cout << "Created scenario with " << num_cameras << " cameras and " 
              << num_landmarks << " landmarks\n";
}

void RunSolverBenchmark(bool use_2d_bearing_factors)
{
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "           FACTORAMA SOLVER BENCHMARK\n"; 
    std::cout << std::string(60, '=') << "\n\n";
    
    // Create test scenario
    std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
    std::vector<Eigen::Vector3d> gt_landmark_positions;
    CreateLargeScenario(gt_camera_poses, gt_landmark_positions);
    
    // Optimization methods to test
    const char* method_names[] = {"GaussNewton", "LevenbergMarquardt"};
    OptimizerMethod method_types[] = {
        OptimizerMethod::GaussNewton,
        OptimizerMethod::LevenbergMarquardt
    };
    
    // Store results for comparison
    struct BenchmarkResult {
        std::string name;
        bool converged;
        int iterations;
        double initial_residual;
        double final_residual;
        double total_time_ms;
        double avg_time_per_iter_ms;
    };
    
    std::vector<BenchmarkResult> results;
    
    // Test each optimization method
    for (int method_idx = 0; method_idx < 2; ++method_idx) {
        std::cout << "\n" << std::string(50, '-') << "\n";
        std::cout << "Testing Method: " << method_names[method_idx] << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        // Create fresh factor graph for each test
        auto graph = std::make_shared<FactorGraph>();
        if (use_2d_bearing_factors) {  // Use BearingProjectionFactor2D for LevenbergMarquardt
            std::cout << "Using 2d bearing factor" << std::endl;
            *graph = CreateGraphWithBearingProjection2D(
                gt_camera_poses, 
                gt_landmark_positions, 
                true,    // random_noise
                false,   // constant_pose (let poses optimize)
                true,    // prior_factors
                0.08,    // noise_sigma (moderate noise)
                0.1      // sparsity (10% missing observations)
            );
        } else {
            std::cout << "Using 3d bearing factor" << std::endl;
            *graph = CreateGraphWithLandmarks(
                gt_camera_poses, 
                gt_landmark_positions, 
                true,    // random_noise
                false,   // constant_pose (let poses optimize)
                true,    // prior_factors
                0.08,    // noise_sigma (moderate noise)
                0.1      // sparsity (10% missing observations)
            );
        }
        
        graph->set_sparse_jacobians(true);
        graph->finalize_structure();

        // Run the jacobian test!
        if(!graph->detailed_factor_test(1e-6, false)) {
            // If it failed, re run it with verbose true
            graph->detailed_factor_test(1e-6, true);
        }
        
        // Print Jacobian size
        const Eigen::SparseMatrix<double>& J = graph->compute_sparse_jacobian_matrix();
        std::cout << "Jacobian size: " << J.rows() << " x " << J.cols() << " (density: " 
                  << std::fixed << std::setprecision(2) 
                  << 100.0 * J.nonZeros() / double(J.rows() * J.cols()) << "%)\n";
        
        // Configure optimizer settings
        OptimizerSettings settings;
        settings.method = method_types[method_idx];
        settings.max_num_iterations = 50;
        settings.step_tolerance = 0.0;  // Set to zero to force more iterations
        settings.residual_tolerance = 0.0;  // Set to zero to force more iterations
        settings.learning_rate = 0.1;  // Low learning rate to get more iterations
        settings.verbose = false;      // Keep quiet during benchmark
        
        // Run optimization with timing
        SparseOptimizer optimizer;
        optimizer.setup(graph, settings);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        optimizer.optimize();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time_ms = duration.count() / 1000.0;
        
        // Store results
        BenchmarkResult result;
        result.name = method_names[method_idx];
        result.converged = optimizer.current_stats_.valid;
        result.iterations = optimizer.current_stats_.current_iteration;
        result.initial_residual = optimizer.initial_stats_.residual_norm;
        result.final_residual = optimizer.current_stats_.residual_norm;
        result.total_time_ms = total_time_ms;
        result.avg_time_per_iter_ms = result.iterations > 0 ? total_time_ms / result.iterations : 0.0;
        
        results.push_back(result);
        
        // Print individual results
        std::cout << "  Status: " << (result.converged ? "CONVERGED" : "FAILED") << "\n";
        std::cout << "  Iterations: " << result.iterations << "\n";
        std::cout << "  Initial residual: " << std::scientific << std::setprecision(3) 
                  << result.initial_residual << "\n";
        std::cout << "  Final residual: " << std::scientific << std::setprecision(3) 
                  << result.final_residual << "\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
                  << result.total_time_ms << " ms\n";
        std::cout << "  Avg time/iter: " << std::fixed << std::setprecision(2) 
                  << result.avg_time_per_iter_ms << " ms\n";

        // Rerun the detailed factor test again, after optimization
                // Run the jacobian test!
        if(!graph->detailed_factor_test(1e-6, false)) {
            // If it failed, re run it with verbose true
            graph->detailed_factor_test(1e-6, true);
        }
    }
    
    // Print comparison table
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "                    COMPARISON SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::cout << std::left 
              << std::setw(15) << "Method"
              << std::setw(8) << "Status" 
              << std::setw(6) << "Iters"
              << std::setw(12) << "Final Res"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "Avg/Iter(ms)" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::left 
                  << std::setw(15) << result.name
                  << std::setw(8) << (result.converged ? "OK" : "FAIL")
                  << std::setw(6) << result.iterations
                  << std::setw(12) << std::scientific << std::setprecision(2) << result.final_residual
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.total_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_time_per_iter_ms << "\n";
    }
    
    // Find fastest solver
    auto fastest = std::min_element(results.begin(), results.end(), 
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            if (!a.converged && b.converged) return false;
            if (a.converged && !b.converged) return true;
            return a.total_time_ms < b.total_time_ms;
        });
        
    if (fastest != results.end() && fastest->converged) {
        std::cout << "\n🏆 Fastest converged method: " << fastest->name 
                  << " (" << std::fixed << std::setprecision(1) << fastest->total_time_ms << " ms)\n";
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n\n";
}

int main()
{
    try {
        RunSolverBenchmark(false);
        RunSolverBenchmark(true);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
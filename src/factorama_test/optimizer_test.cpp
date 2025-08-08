#include <catch2/catch_test_macros.hpp>
#include "factorama/sparse_optimizer.hpp"
#include "factorama/factor_graph.hpp"
#include "factorama_test/test_utils.hpp"

using namespace factorama;

constexpr double kTol = 1e-6;



TEST_CASE("SparseOptimizer basic GN convergence", "[optimizer]") 
{
    // Step 1: Construct a factor graph (you’ll fill this in)
    auto graph = std::make_shared<FactorGraph>();

    float sparsity = 0.0f;
    std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
    std::vector<Eigen::Vector3d> gt_landmark_positions;
    CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
    *graph = CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.04, 1.0 - sparsity);
    graph->set_sparse_jacobians(true);
    graph->finalize_structure();

    // Step 2: Configure optimizer settings
    OptimizerSettings settings;
    settings.method = OptimizerMethod::GaussNewton;
    settings.max_num_iterations = 10;
    settings.step_tolerance = 1e-8;
    settings.residual_tolerance = 1e-8;
    settings.learning_rate = 1.0;
    settings.verbose = true;

    // Step 3: Construct and run optimizer
    SparseOptimizer optimizer;
    optimizer.setup(graph, settings);
    optimizer.optimize();

    // Step 4: Validate result (fill in real expectations)
    const auto &variables = graph->get_all_variables();

    // Ensure that the optimization went for at least 1 round
    CAPTURE(optimizer.current_stats_.valid);
    CAPTURE(optimizer.current_stats_.current_iteration);
    REQUIRE(optimizer.current_stats_.valid);
    REQUIRE(optimizer.current_stats_.current_iteration > 1);

    // Ensure that the residual decreased

    CAPTURE(optimizer.current_stats_.residual_norm);
    CAPTURE(optimizer.initial_stats_.residual_norm);
    REQUIRE(optimizer.current_stats_.residual_norm < optimizer.initial_stats_.residual_norm);

}



TEST_CASE("SparseOptimizer LM convergence", "[optimizer]") 
{
    // Step 1: Construct a factor graph (you’ll fill this in)
    auto graph = std::make_shared<FactorGraph>();

    float sparsity = 0.0f;
    std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
    std::vector<Eigen::Vector3d> gt_landmark_positions;
    CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
    *graph = CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.04, 1.0 - sparsity);
    graph->set_sparse_jacobians(true);
    graph->finalize_structure();

    // Step 2: Configure optimizer settings
    OptimizerSettings settings;
    settings.method = OptimizerMethod::LevenbergMarquardt;
    settings.max_num_iterations = 10;
    settings.step_tolerance = 1e-8;
    settings.residual_tolerance = 1e-8;
    settings.learning_rate = 1.0;
    settings.verbose = true;

    // Step 3: Construct and run optimizer
    SparseOptimizer optimizer;
    optimizer.setup(graph, settings);
    optimizer.optimize();

    // Step 4: Validate result (fill in real expectations)
    const auto &variables = graph->get_all_variables();

    // Ensure that the optimization went for at least 1 round
    CAPTURE(optimizer.current_stats_.valid);
    CAPTURE(optimizer.current_stats_.current_iteration);
    REQUIRE(optimizer.current_stats_.valid);
    REQUIRE(optimizer.current_stats_.current_iteration > 1);

    // Ensure that the residual decreased

    CAPTURE(optimizer.current_stats_.residual_norm);
    CAPTURE(optimizer.initial_stats_.residual_norm);
    REQUIRE(optimizer.current_stats_.residual_norm < optimizer.initial_stats_.residual_norm);

}
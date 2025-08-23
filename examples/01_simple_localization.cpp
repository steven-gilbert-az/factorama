#include <iostream>
#include <factorama/factor_graph.hpp>
#include <factorama/generic_variable.hpp>
#include <factorama/generic_prior_factor.hpp>
#include <factorama/sparse_optimizer.hpp>

using namespace factorama;


void run_example_simple() {

    // Create a 3D position variable (unknown camera position)
    Eigen::Vector3d initial_position = Eigen::Vector3d::Zero();
    auto position_variable = std::make_shared<GenericVariable>(1, initial_position);

    // Create factor graph
    FactorGraph graph;
    graph.add_variable(position_variable);

    // Add GPS measurements as prior factors
    int factor_id = 0;
    
    // GPS measurement 1: [2, 1, 0] with some uncertainty
    auto gps_factor_1 = std::make_shared<GenericPriorFactor>(
        factor_id++, // id
        position_variable, // variable
        Eigen::Vector3d(2.0, 1.0, 0.0), // estimate value 
        0.5); // sigma
    graph.add_factor(gps_factor_1);
    
    // GPS measurement 2: Another measurement with different uncertainty
    Eigen::Vector3d gps_measurement_2(2.2, 0.8, 0.1);
    auto gps_factor_2 = std::make_shared<GenericPriorFactor>(
        factor_id++, 
        position_variable, 
        Eigen::Vector3d(2.2, 0.8, 0.1), 
        0.8);
    graph.add_factor(gps_factor_2);

    // Finalize and optimize
    graph.set_sparse_jacobians(true);
    graph.finalize_structure();

    // Configure optimizer
    OptimizerSettings settings;
    settings.method = OptimizerMethod::LevenbergMarquardt;
    settings.max_num_iterations = 50;
    settings.step_tolerance = 1e-8;
    settings.verbose = true;

    // Run optimization
    auto graph_ptr = std::make_shared<FactorGraph>(graph);
    SparseOptimizer optimizer;
    optimizer.setup(graph_ptr, settings);
    
    std::cout << std::endl << "Optimizing..." << std::endl;
    optimizer.optimize();

    std::cout << "Finished optimizing. Final Graph:" << std::endl;
    graph_ptr->print_variables();

}

void run_example() {
    std::cout << "=== Simple Camera Localization ===" << std::endl;
    std::cout << "Problem: Find camera position from GPS measurements" << std::endl;
    std::cout << "Demonstrates: Basic factor graph with position priors" << std::endl << std::endl;

    // Create a 3D position variable (unknown camera position)
    // Initial guess: camera is at origin
    Eigen::Vector3d initial_position = Eigen::Vector3d::Zero();
    auto camera_position = std::make_shared<GenericVariable>(1, initial_position);

    std::cout << "Initial camera position guess: " << camera_position->value().transpose() << std::endl;

    // Create factor graph
    FactorGraph graph;
    graph.add_variable(camera_position);

    // Add GPS measurements as prior factors
    int factor_id = 0;
    
    // GPS measurement 1: Camera is near [2, 1, 0] with some uncertainty
    double gps_sigma = 0.5;  // 0.5 meter uncertainty
    Eigen::Vector3d gps_measurement_1(2.0, 1.0, 0.0);
    auto gps_factor_1 = std::make_shared<GenericPriorFactor>(
        factor_id++, camera_position, gps_measurement_1, gps_sigma);
    graph.add_factor(gps_factor_1);
    
    // GPS measurement 2: Another measurement with different uncertainty
    double gps_sigma_2 = 0.8;  // Less precise measurement
    Eigen::Vector3d gps_measurement_2(2.2, 0.8, 0.1);
    auto gps_factor_2 = std::make_shared<GenericPriorFactor>(
        factor_id++, camera_position, gps_measurement_2, gps_sigma_2);
    graph.add_factor(gps_factor_2);

    std::cout << "Added GPS measurements:" << std::endl;
    std::cout << "  GPS 1: " << gps_measurement_1.transpose() << " (sigma: " << gps_sigma << ")" << std::endl;
    std::cout << "  GPS 2: " << gps_measurement_2.transpose() << " (sigma: " << gps_sigma_2 << ")" << std::endl;

    // Finalize and optimize
    graph.set_sparse_jacobians(true);
    graph.finalize_structure();

    // Configure optimizer
    OptimizerSettings settings;
    settings.method = OptimizerMethod::LevenbergMarquardt;
    settings.max_num_iterations = 50;
    settings.step_tolerance = 1e-8;
    settings.verbose = false;

    // Run optimization
    auto graph_ptr = std::make_shared<FactorGraph>(graph);
    SparseOptimizer optimizer;
    optimizer.setup(graph_ptr, settings);
    
    std::cout << std::endl << "Optimizing..." << std::endl;
    optimizer.optimize();

    // Show results
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Final camera position: " << camera_position->value().transpose() << std::endl;
    
    // The optimal position should be a weighted average of the GPS measurements
    // More precise measurements (smaller sigma) have higher weight
    double w1 = 1.0 / (gps_sigma * gps_sigma);
    double w2 = 1.0 / (gps_sigma_2 * gps_sigma_2);
    Eigen::Vector3d expected_pos = (w1 * gps_measurement_1 + w2 * gps_measurement_2) / (w1 + w2);
    std::cout << "Expected position (weighted average): " << expected_pos.transpose() << std::endl;
    
    std::cout << std::endl << "Key insight: The factor graph automatically combines" << std::endl;
    std::cout << "multiple measurements, weighting more precise ones higher." << std::endl;
}

int main() {
    //run_example();
    run_example_simple();
    return 0;
}
#include <iostream>
#include <factorama/factor_graph.hpp>
#include <factorama/pose_variable.hpp>
#include <factorama/landmark_variable.hpp>
#include <factorama/bearing_observation_factor.hpp>
#include <factorama/sparse_optimizer.hpp>

using namespace factorama;

void run_example() {
    std::cout << "=== Landmark Triangulation ===" << std::endl;
    std::cout << "Problem: Find 3D landmark positions from multiple camera views" << std::endl;
    std::cout << "Demonstrates: How multiple observations constrain 3D structure" << std::endl << std::endl;

    // Create known camera poses at different positions
    // Camera 1: At origin, looking forward
    Eigen::Matrix<double, 6, 1> pose1_vec = Eigen::Matrix<double, 6, 1>::Zero();
    auto camera1 = std::make_shared<PoseVariable>(1, pose1_vec);
    camera1->set_constant(true);  // Known camera positions
    
    // Camera 2: Moved to the right, still looking forward
    Eigen::Matrix<double, 6, 1> pose2_vec = Eigen::Matrix<double, 6, 1>::Zero();
    pose2_vec[0] = 2.0;  // 2 meters to the right
    auto camera2 = std::make_shared<PoseVariable>(2, pose2_vec);
    camera2->set_constant(true);  // Known camera positions
    
    // Camera 3: Moved up and back
    Eigen::Matrix<double, 6, 1> pose3_vec = Eigen::Matrix<double, 6, 1>::Zero();
    pose3_vec[1] = 1.0;  // 1 meter up
    pose3_vec[2] = -1.0; // 1 meter back
    auto camera3 = std::make_shared<PoseVariable>(3, pose3_vec);
    camera3->set_constant(true);  // Known camera positions

    std::cout << "Known camera positions:" << std::endl;
    std::cout << "  Camera 1: " << camera1->pos_W().transpose() << std::endl;
    std::cout << "  Camera 2: " << camera2->pos_W().transpose() << std::endl;
    std::cout << "  Camera 3: " << camera3->pos_W().transpose() << std::endl << std::endl;

    // Create unknown landmark positions (initial guesses)
    Eigen::Vector3d landmark1_guess(0.0, 0.0, 5.0);  // Roughly where we expect it
    Eigen::Vector3d landmark2_guess(1.0, -1.0, 4.0);
    auto landmark1 = std::make_shared<LandmarkVariable>(4, landmark1_guess);
    auto landmark2 = std::make_shared<LandmarkVariable>(5, landmark2_guess);

    std::cout << "Initial landmark guesses:" << std::endl;
    std::cout << "  Landmark 1: " << landmark1->pos_W().transpose() << std::endl;
    std::cout << "  Landmark 2: " << landmark2->pos_W().transpose() << std::endl << std::endl;

    // Create factor graph
    FactorGraph graph;
    graph.add_variable(camera1);
    graph.add_variable(camera2);
    graph.add_variable(camera3);
    graph.add_variable(landmark1);
    graph.add_variable(landmark2);

    // Add bearing observations (simulated measurements)
    int factor_id = 0;
    double bearing_sigma = 0.02;  // 0.02 radian uncertainty (~1 degree)

    // Landmark 1 observations
    // True landmark 1 position: [0.5, 0.2, 5.0]
    Eigen::Vector3d true_landmark1(0.5, 0.2, 5.0);
    
    // Bearing from camera 1 to landmark 1
    Eigen::Vector3d bearing_c1_l1 = (true_landmark1 - camera1->pos_W()).normalized();
    auto factor_c1_l1 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera1, landmark1, bearing_c1_l1, bearing_sigma);
    graph.add_factor(factor_c1_l1);
    
    // Bearing from camera 2 to landmark 1
    Eigen::Vector3d bearing_c2_l1 = (true_landmark1 - camera2->pos_W()).normalized();
    auto factor_c2_l1 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera2, landmark1, bearing_c2_l1, bearing_sigma);
    graph.add_factor(factor_c2_l1);
    
    // Bearing from camera 3 to landmark 1
    Eigen::Vector3d bearing_c3_l1 = (true_landmark1 - camera3->pos_W()).normalized();
    auto factor_c3_l1 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera3, landmark1, bearing_c3_l1, bearing_sigma);
    graph.add_factor(factor_c3_l1);

    // Landmark 2 observations
    // True landmark 2 position: [1.2, -0.8, 4.5]
    Eigen::Vector3d true_landmark2(1.2, -0.8, 4.5);
    
    // Bearing from camera 1 to landmark 2
    Eigen::Vector3d bearing_c1_l2 = (true_landmark2 - camera1->pos_W()).normalized();
    auto factor_c1_l2 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera1, landmark2, bearing_c1_l2, bearing_sigma);
    graph.add_factor(factor_c1_l2);
    
    // Bearing from camera 2 to landmark 2
    Eigen::Vector3d bearing_c2_l2 = (true_landmark2 - camera2->pos_W()).normalized();
    auto factor_c2_l2 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera2, landmark2, bearing_c2_l2, bearing_sigma);
    graph.add_factor(factor_c2_l2);

    std::cout << "Added bearing observations from cameras to landmarks" << std::endl;
    std::cout << "Each observation constrains the landmark to lie along a ray" << std::endl;
    std::cout << "Multiple rays intersect at the true landmark position" << std::endl << std::endl;

    // Finalize and optimize
    graph.set_sparse_jacobians(true);
    graph.finalize_structure();

    // Configure optimizer
    OptimizerSettings settings;
    settings.method = OptimizerMethod::LevenbergMarquardt;
    settings.max_num_iterations = 100;
    settings.step_tolerance = 1e-8;
    settings.verbose = false;

    // Run optimization
    auto graph_ptr = std::make_shared<FactorGraph>(graph);
    SparseOptimizer optimizer;
    optimizer.setup(graph_ptr, settings);
    
    std::cout << "Optimizing..." << std::endl;
    optimizer.optimize();

    // Show results
    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Estimated landmark positions:" << std::endl;
    std::cout << "  Landmark 1: " << landmark1->pos_W().transpose() << std::endl;
    std::cout << "  Landmark 2: " << landmark2->pos_W().transpose() << std::endl << std::endl;
    
    std::cout << "True landmark positions:" << std::endl;
    std::cout << "  Landmark 1: " << true_landmark1.transpose() << std::endl;
    std::cout << "  Landmark 2: " << true_landmark2.transpose() << std::endl << std::endl;
    
    // Calculate errors
    double error1 = (landmark1->pos_W() - true_landmark1).norm();
    double error2 = (landmark2->pos_W() - true_landmark2).norm();
    std::cout << "Estimation errors:" << std::endl;
    std::cout << "  Landmark 1: " << error1 << " meters" << std::endl;
    std::cout << "  Landmark 2: " << error2 << " meters" << std::endl << std::endl;
    
    std::cout << "Key insight: Triangulation works by intersecting rays from" << std::endl;
    std::cout << "multiple viewpoints. More cameras = better constraint." << std::endl;
}

int main() {
    run_example();
    return 0;
}
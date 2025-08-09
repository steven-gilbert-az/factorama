# Factorama Examples

## Bundle Adjustment Example

```cpp
#include <iostream>
#include <factorama/factor_graph.hpp>
#include <factorama/pose_variable.hpp>
#include <factorama/landmark_variable.hpp>
#include <factorama/bearing_observation_factor.hpp>
#include <factorama/sparse_optimizer.hpp>
#include <factorama/pose_prior_factors.hpp>
#include <factorama/generic_prior_factor.hpp>

using namespace factorama;

int main() {
    std::cout << "*** FACTORAMA TEST ***" << std::endl;

    // Create variables
    Eigen::Matrix<double, 6, 1> pose_vec = Eigen::Matrix<double, 6, 1>::Zero();
    auto camera_pose = std::make_shared<PoseVariable>(1, pose_vec);

    Eigen::Vector3d landmark_pos(0.0, 1.0, 6.0);
    auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

    Eigen::Vector3d landmark_pos2(0.0, -1.0, 6.0);  // the second landmark will be constant
    auto landmark2 = std::make_shared<LandmarkVariable>(3, landmark_pos);
    landmark2->set_is_constant(true);


    // Create factors
    int factor_id = 0;

    // The bearing vector is a unit vector pointing from camera to the landmark
    // Note - this is in the CAMERA reference frame (not world). but since the camera orientation
    // starts out at 0,0,0 then these are the same. 
    double bearing_sigma = 0.1;
    Eigen::Vector3d bearing_vector(0.0, 0.0, 1.0); // bearing 1 is noisy
    auto bearing_factor = std::make_shared<BearingObservationFactor>(
        factor_id++, camera_pose, landmark, bearing_vector, bearing_sigma);


    Eigen::Vector3d bearing_vector2 = landmark_pos2.normalized(); // bearing 2 can be perfect.
    auto bearing_factor2 = std::make_shared<BearingObservationFactor>(
        factor_id++, camera_pose, landmark2, bearing_vector2, bearing_sigma);
        

    // Create landmark prior
    double landmark_sigma = 1.0;
    auto landmark_prior = std::make_shared<GenericPriorFactor>(
        factor_id++, landmark,  Eigen::Vector3d(0.0, 0.0, 5.0), landmark_sigma);

    
    // Create Pose Position Prior
    double position_sigma = 0.5;
    auto pose_position_prior_factor = std::make_shared<PosePositionPriorFactor>(
        factor_id++, camera_pose, Eigen::Vector3d::Zero(), position_sigma);
    
    // Create pose orientation prior
    double orientation_sigma = 0.1;
    auto pose_orientation_prior_factor = std::make_shared<PoseOrientationPriorFactor>(
        factor_id++, camera_pose, Eigen::Vector3d::Zero(), orientation_sigma);


    // Build factor graph
    FactorGraph graph;
    graph.add_variable(camera_pose);
    graph.add_variable(landmark);
    graph.add_variable(landmark2);
    graph.add_factor(bearing_factor);
    graph.add_factor(bearing_factor2);
    graph.add_factor(pose_position_prior_factor);
    graph.add_factor(pose_orientation_prior_factor);
    graph.add_factor(landmark_prior);
    graph.set_sparse_jacobians(true);
    graph.finalize_structure();
    

    // Configure optimizer
    OptimizerSettings settings;
    settings.method = OptimizerMethod::LevenbergMarquardt;
    settings.max_num_iterations = 100;
    settings.step_tolerance = 1e-6;
    settings.verbose = true;

    // Run optimization
    auto graph_ptr = std::make_shared<FactorGraph>(graph);
    SparseOptimizer optimizer;
    optimizer.setup(graph_ptr, settings);
    optimizer.optimize();

    // Print final variable states
    graph.print_variables();

    // Run jacobian test
    graph.detailed_factor_test(1e-6, true);
}
```
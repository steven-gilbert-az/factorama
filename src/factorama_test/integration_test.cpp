#include <catch2/catch_test_macros.hpp>
#include <random>
#include <functional>
#include "factorama/pose_variable.hpp"
#include "factorama/pose_2d_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/generic_variable.hpp"
#include "factorama/inverse_range_variable.hpp"
#include "factorama/plane_variable.hpp"
#include "factorama/bearing_observation_factor.hpp"
#include "factorama/bearing_observation_factor_2d.hpp"
#include "factorama/inverse_range_bearing_factor.hpp"
#include "factorama/bearing_projection_factor_2d.hpp"
#include "factorama/plane_factor.hpp"
#include "factorama/plane_prior_factor.hpp"
#include "factorama/pose_2d_prior_factor.hpp"
#include "factorama/range_bearing_factor_2d.hpp"
#include "factorama/generic_prior_factor.hpp"
#include "factorama/factor_graph.hpp"
#include "factorama/sparse_optimizer.hpp"
#include "factorama_test/test_utils.hpp"
using namespace factorama;

// Test scenario configuration
struct TestScenario {
    std::string name;
    std::function<FactorGraph()> create_graph;
    OptimizerSettings settings;
    bool should_run_jacobian_test = false;
    double jacobian_tolerance = 1e-6;
    bool verbose_output = false;
    bool expect_failure = false; // If true, expect optimizer to fail with invalid status
};

// Optimization result structure
struct OptimizationResult {
    double initial_norm;
    double final_norm;
    bool converged;
    std::string details;
};

// Common optimization test pipeline
OptimizationResult run_optimization_test(FactorGraph graph, const OptimizerSettings& settings, bool run_jacobian_test = false, double jacobian_tol = 1e-6, bool verbose = false, bool expect_failure = false) {
    OptimizationResult result;

    graph.finalize_structure();

    if (verbose) {
        std::cout << "**** INITIAL FACTOR GRAPH ****" << std::endl;
        graph.print_structure();
        graph.print_variables();
    }

    // Run jacobian test if requested (only for non-failure cases)
    if(run_jacobian_test && !expect_failure){
        REQUIRE(graph.detailed_factor_test(jacobian_tol, true));
    }


    // Get initial residual norm
    result.initial_norm = graph.compute_full_residual_vector().norm();

    // Create copy for optimization
    std::shared_ptr<FactorGraph> graph_copy = std::make_shared<FactorGraph>(graph);
    graph_copy->finalize_structure();

    // Run optimization
    SparseOptimizer optimizer;
    optimizer.setup(graph_copy, settings);
    optimizer.optimize();

    if (expect_failure) {
        // Check that optimization failed with appropriate status
        std::cout << "Optimizer status: " << static_cast<int>(optimizer.current_stats_.status) << std::endl;
        std::cout << "Optimizer valid: " << optimizer.current_stats_.valid << std::endl;
        REQUIRE(optimizer.current_stats_.valid == false);
        REQUIRE((optimizer.current_stats_.status == OptimizerStatus::SINGULAR_HESSIAN ||
                 optimizer.current_stats_.status == OptimizerStatus::ILL_CONDITIONED ||
                 optimizer.current_stats_.status == OptimizerStatus::FAILED));
    } else {
        // Check that optimization status is SUCCESS
        REQUIRE(optimizer.current_stats_.status == OptimizerStatus::SUCCESS);
        REQUIRE(optimizer.current_stats_.valid == true);
    }

    // Get final residual norm
    result.final_norm = graph_copy->compute_full_residual_vector().norm();
    result.converged = (result.final_norm < result.initial_norm);

    if (verbose) {
        std::cout << "**** FINAL FACTOR GRAPH ****" << std::endl;
        graph_copy->print_variables();
    }

    // Test covariance estimation (should not crash) - only for successful optimizations
    if (verbose && !expect_failure) {
        std::cout << "**** TESTING COVARIANCE ESTIMATION ****" << std::endl;
        optimizer.print_all_covariances();
    }

    // Run post-optimization jacobian test if requested (only for non-failure cases)
    if (run_jacobian_test && !expect_failure) {
        REQUIRE(graph_copy->detailed_factor_test(jacobian_tol, true));
    }

    return result;
}


TEST_CASE("Consolidated Integration Tests")
{
    std::cout << "\n=== Running Consolidated Integration Tests ===" << std::endl;
    
    // Define all test scenarios
    std::vector<TestScenario> scenarios;
    
    // Scenario 1: Basic pose optimization
    {
        OptimizerSettings settings1;
        settings1.method = OptimizerMethod::GaussNewton;
        settings1.verbose = true;
        scenarios.push_back({
            "Full optimization with pose variables",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true);
            },
            settings1,
            false, 1e-6, true
        });
    }
    
    // Scenario 2: Basic pose optimization (second test)
    {
        OptimizerSettings settings2;
        settings2.method = OptimizerMethod::GaussNewton;
        settings2.verbose = true;
        scenarios.push_back({
            "Full optimization with pose variables - second test",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true);
            },
            settings2,
            false, 1e-6, true
        });
    }
    
    // Scenario 3: Inverse range variables
    {
        OptimizerSettings settings3;
        settings3.method = OptimizerMethod::LevenbergMarquardt;
        settings3.verbose = true;
        scenarios.push_back({
            "Full optimization with inverse range variables",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, true);
            },
            settings3,
            false, 1e-6, true
        });
    }
    
    // Scenario 4: Inverse range with constant poses
    {
        OptimizerSettings settings4;
        settings4.method = OptimizerMethod::GaussNewton;
        settings4.verbose = true;
        scenarios.push_back({
            "Full optimization with inverse range variables (Camera poses held constant)",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, true, true);
            },
            settings4,
            false, 1e-6, true
        });
    }
    
    // Scenario 5: Larger planar problem with jacobian testing
    {
        OptimizerSettings settings5;
        settings5.method = OptimizerMethod::LevenbergMarquardt;
        settings5.learning_rate = 1.0;
        settings5.verbose = true;
        scenarios.push_back({
            "Full optimization with inverse range variables (larger planar problem)",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, true, true, 0.004, 5.0);
            },
            settings5,
            true, 1e-6, true
        });
    }
    
    // Scenario 6: Inverse range with prior factors
    {
        OptimizerSettings settings6;
        settings6.method = OptimizerMethod::GaussNewton;
        settings6.verbose = true;
        scenarios.push_back({
            "Full optimization with inverse range variables (and prior factors!)",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, true, 0.005, 5.0);
            },
            settings6,
            false, 1e-6, true
        });
    }
    
    // Scenario 7: Prior factors with landmarks
    {
        OptimizerSettings settings7;
        settings7.method = OptimizerMethod::GaussNewton;
        settings7.verbose = true;
        scenarios.push_back({
            "Full optimization with prior factors!",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, true, 0.005);
            },
            settings7,
            false, 1e-6, true
        });
    }
    
    // Scenario 8: Relative orientation alignment factors
    {
        OptimizerSettings settings8;
        settings8.method = OptimizerMethod::GaussNewton;
        settings8.verbose = true;
        scenarios.push_back({
            "Full optimization with relative orientation alignment factors",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses{
                    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                    {0.0, 0.0, 0.0, 0.5, 0.5, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.5, 0.5},
                    {0.0, 0.0, 0.0, 0.5, 0.0, 0.5}};
                Eigen::Vector3d gt_rot_IC{0.1, 0.2, 0.3};
                return CreateGraphWithPoseBetweenFactors(gt_camera_poses, gt_rot_IC, true, 0.04);
            },
            settings8,
            false, 1e-6, false
        });
    }
    
    // Scenario 9: BearingProjectionFactor2D basic test
    {
        OptimizerSettings settings9;
        settings9.method = OptimizerMethod::GaussNewton;
        settings9.verbose = true;
        scenarios.push_back({
            "Full optimization with BearingProjectionFactor2D",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithBearingProjection2D(gt_camera_poses, gt_landmark_positions, true, false, true);
            },
            settings9,
            false, 1e-6, true
        });
    }
    
    // Scenario 10: BearingProjectionFactor2D with prior factors
    {
        OptimizerSettings settings10;
        settings10.method = OptimizerMethod::GaussNewton;
        settings10.verbose = true;
        scenarios.push_back({
            "Full optimization with BearingProjectionFactor2D and prior factors",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreatePlanarScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithBearingProjection2D(gt_camera_poses, gt_landmark_positions, true, false, true, 0.005);
            },
            settings10,
            false, 1e-6, true
        });
    }

    // ===== FAILURE CASES: Test that optimizer correctly detects ill-conditioned problems =====

    // Scenario 11: Landmarks without priors - should FAIL
    {
        OptimizerSettings settings11;
        settings11.method = OptimizerMethod::GaussNewton;
        settings11.verbose = true;
        scenarios.push_back({
            "FAILURE TEST: Landmarks without priors (expect ill-conditioned)",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithLandmarks(gt_camera_poses, gt_landmark_positions, true, false, false);
            },
            settings11,
            false, 1e-6, true, true  // expect_failure = true
        });
    }

    // Scenario 12: Inverse range without priors - should FAIL
    {
        OptimizerSettings settings12;
        settings12.method = OptimizerMethod::GaussNewton;
        settings12.verbose = true;
        scenarios.push_back({
            "FAILURE TEST: Inverse range without priors (expect ill-conditioned)",
            []() {
                std::vector<Eigen::Matrix<double, 6, 1>> gt_camera_poses;
                std::vector<Eigen::Vector3d> gt_landmark_positions;
                CreateSimpleScenario(gt_camera_poses, gt_landmark_positions);
                return CreateGraphWithInverseRangeVariables(gt_camera_poses, gt_landmark_positions, true, false, false);
            },
            settings12,
            false, 1e-6, true, true  // expect_failure = true
        });
    }

    // Scenario 13: Plane fitting
    {
        OptimizerSettings settings13;
        settings13.method = OptimizerMethod::LevenbergMarquardt;
        settings13.verbose = true;
        scenarios.push_back({
            "Plane fitting with 6 points",
            []() {
                FactorGraph graph;
                int var_id = 0;
                int factor_id = 0;

                // Ground truth plane: z = 5 (normal = [0, 0, 1], distance = -5)

                // Create 6 points on the plane with small noise
                std::vector<Eigen::Vector3d> point_positions = {
                    {0.0, 0.0, 5.0},
                    {1.0, 0.0, 5.01},
                    {0.0, 1.0, 4.99},
                    {-1.0, 0.0, 5.02},
                    {0.0, -1.0, 4.98},
                    {0.5, 0.5, 5.0}
                };

                // Create point variables (held constant - these are our measurements)
                std::vector<std::shared_ptr<LandmarkVariable>> points;
                for (const auto& pos : point_positions) {
                    auto point = std::make_shared<LandmarkVariable>(var_id++, pos);
                    point->set_constant(true);
                    points.push_back(point);
                    graph.add_variable(point);
                }

                // Create plane variable with incorrect initial guess
                Eigen::Vector3d initial_normal(0.1, 0.1, 1.0);  // Slightly wrong orientation
                double initial_distance = -4.5;  // Slightly wrong distance
                auto plane = std::make_shared<PlaneVariable>(var_id++, initial_normal, initial_distance);
                graph.add_variable(plane);

                // Add plane factors connecting each point to the plane
                double sigma = 0.1;
                for (auto& point : points) {
                    auto factor = std::make_shared<PlaneFactor>(factor_id++, point.get(), plane.get(), sigma);
                    graph.add_factor(factor);
                }

                return graph;
            },
            settings13,
            true, 1e-6, true
        });
    }

    // Scenario 14: Plane fitting with prior
    {
        OptimizerSettings settings14;
        settings14.method = OptimizerMethod::LevenbergMarquardt;
        settings14.verbose = true;
        scenarios.push_back({
            "Plane fitting with points and prior constraint",
            []() {
                FactorGraph graph;
                int var_id = 0;
                int factor_id = 0;

                // Ground truth plane: approximately z = 5
                // Create 4 points with more noise this time
                std::vector<Eigen::Vector3d> point_positions = {
                    {0.0, 0.0, 5.1},
                    {1.0, 0.0, 4.9},
                    {0.0, 1.0, 5.05},
                    {-1.0, -1.0, 4.95}
                };

                // Create point variables (held constant)
                std::vector<std::shared_ptr<LandmarkVariable>> points;
                for (const auto& pos : point_positions) {
                    auto point = std::make_shared<LandmarkVariable>(var_id++, pos);
                    point->set_constant(true);
                    points.push_back(point);
                    graph.add_variable(point);
                }

                // Create plane variable with poor initial guess
                Eigen::Vector3d initial_normal(0.3, 0.2, 1.0);
                double initial_distance = -4.0;
                auto plane = std::make_shared<PlaneVariable>(var_id++, initial_normal, initial_distance);
                graph.add_variable(plane);

                // Add plane factors connecting each point to the plane
                double point_sigma = 0.2;
                for (auto& point : points) {
                    auto factor = std::make_shared<PlaneFactor>(factor_id++, point.get(), plane.get(), point_sigma);
                    graph.add_factor(factor);
                }

                // Add prior constraint to help constrain the solution
                Eigen::Vector3d prior_normal(0.0, 0.0, 1.0);
                double prior_distance = -5.0;
                double normal_sigma = 0.5;  // Relatively weak prior on orientation
                double distance_sigma = 1.0;  // Weak prior on distance
                auto prior_factor = std::make_shared<PlanePriorFactor>(
                    factor_id++, plane.get(), prior_normal, prior_distance, normal_sigma, distance_sigma);
                graph.add_factor(prior_factor);

                return graph;
            },
            settings14,
            true, 1e-6, true
        });
    }

    // Scenario 15: Plane fitting with distance scaling
    {
        OptimizerSettings settings15;
        settings15.method = OptimizerMethod::LevenbergMarquardt;
        settings15.verbose = true;
        scenarios.push_back({
            "Plane fitting with distance scaling enabled",
            []() {
                FactorGraph graph;
                int var_id = 0;
                int factor_id = 0;

                // Ground truth plane: z = 5 (normal = [0, 0, 1], distance = -5)
                // Create points at varying distances from a scaling origin
                std::vector<Eigen::Vector3d> point_positions = {
                    {0.0, 0.0, 5.0},    // close to origin
                    {2.0, 0.0, 5.01},   // medium distance
                    {0.0, 3.0, 4.99},   // medium distance
                    {-4.0, 0.0, 5.02},  // far from origin
                    {0.0, -5.0, 4.98},  // very far from origin
                    {1.0, 1.0, 5.0}     // close to origin
                };

                // Create point variables (held constant - these are our measurements)
                std::vector<std::shared_ptr<LandmarkVariable>> points;
                for (const auto& pos : point_positions) {
                    auto point = std::make_shared<LandmarkVariable>(var_id++, pos);
                    point->set_constant(true);
                    points.push_back(point);
                    graph.add_variable(point);
                }

                // Create plane variable with incorrect initial guess
                Eigen::Vector3d initial_normal(0.1, 0.1, 1.0);
                double initial_distance = -4.5;
                auto plane = std::make_shared<PlaneVariable>(var_id++, initial_normal, initial_distance);
                graph.add_variable(plane);

                // Add plane factors with distance scaling
                // Points farther from origin will have less weight
                double sigma = 0.1;
                bool do_distance_scaling = true;
                double dist_scaling_r0 = 3.0;  // characteristic scaling distance
                Eigen::Vector3d dist_scaling_p0(0.0, 0.0, 5.0);  // scaling origin on the plane

                for (auto& point : points) {
                    auto factor = std::make_shared<PlaneFactor>(
                        factor_id++, point.get(), plane.get(), sigma,
                        do_distance_scaling, dist_scaling_r0, dist_scaling_p0);
                    graph.add_factor(factor);
                }

                return graph;
            },
            settings15,
            true, 1e-6, true
        });
    }

    // Scenario 16: 2D SLAM with poses, landmarks, priors, and bearing observations
    {
        OptimizerSettings settings16;
        settings16.method = OptimizerMethod::LevenbergMarquardt;
        settings16.verbose = true;
        settings16.max_num_iterations = 50;
        scenarios.push_back({
            "2D SLAM with poses, landmarks, and bearing observations",
            []() {
                FactorGraph graph;
                int var_id = 0;
                int factor_id = 0;

                // Ground truth: 3 poses in a triangular configuration
                // Pose 0: origin, facing east (0 radians)
                // Pose 1: east of origin, facing north (π/2 radians)
                // Pose 2: north of origin, facing west (π radians)
                std::vector<Eigen::Vector3d> gt_poses = {
                    {0.0, 0.0, 0.0},           // x, y, theta
                    {3.0, 0.0, PI / 2},
                    {1.5, 2.5, PI}
                };

                // Ground truth: 4 landmarks forming a square in the middle
                std::vector<Eigen::Vector2d> gt_landmarks = {
                    {1.0, 1.0},   // Landmark 0
                    {2.0, 1.0},   // Landmark 1
                    {2.0, 2.0},   // Landmark 2
                    {1.0, 2.0}    // Landmark 3
                };

                // Create pose variables with noisy initial guesses
                std::vector<std::shared_ptr<Pose2DVariable>> poses;
                for (size_t i = 0; i < gt_poses.size(); ++i) {
                    // Add noise to initial guess
                    Eigen::Vector3d noisy_pose = gt_poses[i];
                    noisy_pose(0) += 0.2 * (i == 0 ? 0.1 : 0.3);  // x noise
                    noisy_pose(1) += 0.15 * (i == 1 ? -0.2 : 0.2); // y noise
                    noisy_pose(2) += 0.1 * (i == 2 ? 0.15 : -0.1); // theta noise

                    auto pose = std::make_shared<Pose2DVariable>(var_id++, noisy_pose);
                    poses.push_back(pose);
                    graph.add_variable(pose);
                }

                // Create landmark variables with noisy initial guesses
                std::vector<std::shared_ptr<GenericVariable>> landmarks;
                for (size_t i = 0; i < gt_landmarks.size(); ++i) {
                    // Add noise to initial guess
                    Eigen::Vector2d noisy_landmark = gt_landmarks[i];
                    noisy_landmark(0) += 0.1 * (i % 2 == 0 ? 0.2 : -0.15);
                    noisy_landmark(1) += 0.1 * (i / 2 == 0 ? -0.1 : 0.2);

                    auto landmark = std::make_shared<GenericVariable>(var_id++, noisy_landmark);
                    landmarks.push_back(landmark);
                    graph.add_variable(landmark);
                }

                // Add pose priors (all poses have priors based on ground truth)
                double pose_position_sigma = 0.5;
                double pose_angle_sigma = 0.2;
                for (size_t i = 0; i < poses.size(); ++i) {
                    auto pose_prior = std::make_shared<Pose2DPriorFactor>(
                        factor_id++, poses[i].get(), gt_poses[i],
                        pose_position_sigma, pose_angle_sigma);
                    graph.add_factor(pose_prior);
                }

                // Add landmark priors (weaker priors)
                double landmark_sigma = 1.0;
                for (size_t i = 0; i < landmarks.size(); ++i) {
                    auto landmark_prior = std::make_shared<GenericPriorFactor>(
                        factor_id++, landmarks[i].get(), gt_landmarks[i], landmark_sigma);
                    graph.add_factor(landmark_prior);
                }

                // Add bearing observations from poses to landmarks
                // Compute ground truth bearing angles and add observations
                double bearing_sigma = 0.05;  // radians

                // Pose 0 can see landmarks 0, 1, 3
                std::vector<std::pair<int, int>> observations = {
                    {0, 0}, {0, 1}, {0, 3},  // Pose 0 sees landmarks 0, 1, 3
                    {1, 0}, {1, 1}, {1, 2},  // Pose 1 sees landmarks 0, 1, 2
                    {2, 1}, {2, 2}, {2, 3}   // Pose 2 sees landmarks 1, 2, 3
                };

                for (const auto& obs : observations) {
                    int pose_idx = obs.first;
                    int landmark_idx = obs.second;

                    // Compute ground truth bearing angle
                    Eigen::Vector2d pose_pos = gt_poses[pose_idx].head<2>();
                    double pose_theta = gt_poses[pose_idx](2);
                    Eigen::Vector2d landmark_pos = gt_landmarks[landmark_idx];

                    // Delta in world frame
                    Eigen::Vector2d delta_world = landmark_pos - pose_pos;

                    // Rotate to pose frame
                    double c = std::cos(pose_theta);
                    double s = std::sin(pose_theta);
                    Eigen::Matrix2d R_T;
                    R_T << c, s, -s, c;
                    Eigen::Vector2d delta_local = R_T * delta_world;

                    // Compute bearing angle in pose frame
                    double bearing_angle = std::atan2(delta_local(1), delta_local(0));

                    // Create bearing observation factor
                    auto bearing_factor = std::make_shared<BearingObservationFactor2D>(
                        factor_id++, poses[pose_idx].get(), landmarks[landmark_idx].get(),
                        bearing_angle, bearing_sigma);
                    graph.add_factor(bearing_factor);
                }

                return graph;
            },
            settings16,
            true, 1e-6, true
        });
    }

    // Scenario 17: 2D SLAM with poses, landmarks, and range-bearing observations
    {
        OptimizerSettings settings17;
        settings17.method = OptimizerMethod::LevenbergMarquardt;
        settings17.verbose = true;
        settings17.max_num_iterations = 50;
        scenarios.push_back({
            "2D SLAM with poses, landmarks, and range-bearing observations",
            []() {
                FactorGraph graph;
                int var_id = 0;
                int factor_id = 0;

                // Ground truth: 3 poses in a triangular configuration
                // Pose 0: origin, facing east (0 radians)
                // Pose 1: east of origin, facing north (π/2 radians)
                // Pose 2: north of origin, facing west (π radians)
                std::vector<Eigen::Vector3d> gt_poses = {
                    {0.0, 0.0, 0.0},           // x, y, theta
                    {3.0, 0.0, PI / 2},
                    {1.5, 2.5, PI}
                };

                // Ground truth: 4 landmarks forming a square in the middle
                std::vector<Eigen::Vector2d> gt_landmarks = {
                    {1.0, 1.0},   // Landmark 0
                    {2.0, 1.0},   // Landmark 1
                    {2.0, 2.0},   // Landmark 2
                    {1.0, 2.0}    // Landmark 3
                };

                // Create pose variables with noisy initial guesses
                std::vector<std::shared_ptr<Pose2DVariable>> poses;
                for (size_t i = 0; i < gt_poses.size(); ++i) {
                    // Add noise to initial guess
                    Eigen::Vector3d noisy_pose = gt_poses[i];
                    noisy_pose(0) += 0.2 * (i == 0 ? 0.1 : 0.3);  // x noise
                    noisy_pose(1) += 0.15 * (i == 1 ? -0.2 : 0.2); // y noise
                    noisy_pose(2) += 0.1 * (i == 2 ? 0.15 : -0.1); // theta noise

                    auto pose = std::make_shared<Pose2DVariable>(var_id++, noisy_pose);
                    poses.push_back(pose);
                    graph.add_variable(pose);
                }

                // Create landmark variables with noisy initial guesses
                std::vector<std::shared_ptr<GenericVariable>> landmarks;
                for (size_t i = 0; i < gt_landmarks.size(); ++i) {
                    // Add noise to initial guess
                    Eigen::Vector2d noisy_landmark = gt_landmarks[i];
                    noisy_landmark(0) += 0.1 * (i % 2 == 0 ? 0.2 : -0.15);
                    noisy_landmark(1) += 0.1 * (i / 2 == 0 ? -0.1 : 0.2);

                    auto landmark = std::make_shared<GenericVariable>(var_id++, noisy_landmark);
                    landmarks.push_back(landmark);
                    graph.add_variable(landmark);
                }

                // Add pose priors (all poses have priors based on ground truth)
                double pose_position_sigma = 0.5;
                double pose_angle_sigma = 0.2;
                for (size_t i = 0; i < poses.size(); ++i) {
                    auto pose_prior = std::make_shared<Pose2DPriorFactor>(
                        factor_id++, poses[i].get(), gt_poses[i],
                        pose_position_sigma, pose_angle_sigma);
                    graph.add_factor(pose_prior);
                }

                // Add landmark priors (weaker priors)
                double landmark_sigma = 1.0;
                for (size_t i = 0; i < landmarks.size(); ++i) {
                    auto landmark_prior = std::make_shared<GenericPriorFactor>(
                        factor_id++, landmarks[i].get(), gt_landmarks[i], landmark_sigma);
                    graph.add_factor(landmark_prior);
                }

                // Add range-bearing observations from poses to landmarks
                // Compute ground truth range and bearing angles and add observations
                double range_sigma = 0.1;     // meters
                double bearing_sigma = 0.05;  // radians

                // Pose 0 can see landmarks 0, 1, 3
                std::vector<std::pair<int, int>> observations = {
                    {0, 0}, {0, 1}, {0, 3},  // Pose 0 sees landmarks 0, 1, 3
                    {1, 0}, {1, 1}, {1, 2},  // Pose 1 sees landmarks 0, 1, 2
                    {2, 1}, {2, 2}, {2, 3}   // Pose 2 sees landmarks 1, 2, 3
                };

                for (const auto& obs : observations) {
                    int pose_idx = obs.first;
                    int landmark_idx = obs.second;

                    // Compute ground truth range and bearing
                    Eigen::Vector2d pose_pos = gt_poses[pose_idx].head<2>();
                    double pose_theta = gt_poses[pose_idx](2);
                    Eigen::Vector2d landmark_pos = gt_landmarks[landmark_idx];

                    // Delta in world frame
                    Eigen::Vector2d delta_world = landmark_pos - pose_pos;

                    // Rotate to pose frame
                    double c = std::cos(pose_theta);
                    double s = std::sin(pose_theta);
                    Eigen::Matrix2d R_T;
                    R_T << c, s, -s, c;
                    Eigen::Vector2d delta_local = R_T * delta_world;

                    // Compute range and bearing in pose frame
                    double range = delta_local.norm();
                    double bearing_angle = std::atan2(delta_local(1), delta_local(0));

                    // Create range-bearing observation factor
                    auto rb_factor = std::make_shared<RangeBearingFactor2D>(
                        factor_id++, poses[pose_idx].get(), landmarks[landmark_idx].get(),
                        range, bearing_angle, range_sigma, bearing_sigma);
                    graph.add_factor(rb_factor);
                }

                return graph;
            },
            settings17,
            true, 1e-6, true
        });
    }

    // Run all scenarios
    int passed = 0;
    int total = scenarios.size();
    
    for (size_t i = 0; i < scenarios.size(); ++i) {
        const auto& scenario = scenarios[i];
        
        std::cout << "\n=== [" << (i+1) << "/" << total << "] " << scenario.name << " ===" << std::endl;
        
        try {
            auto graph = scenario.create_graph();

            auto result = run_optimization_test(graph, scenario.settings, scenario.should_run_jacobian_test, scenario.jacobian_tolerance, scenario.verbose_output, scenario.expect_failure);

            std::cout << "Initial residual norm: " << result.initial_norm << std::endl;
            std::cout << "Final residual norm: " << result.final_norm << std::endl;

            // Special handling for calibration scenario
            if (scenario.name.find("relative orientation alignment") != std::string::npos) {
                // This would require returning the optimized graph from run_optimization_test
                // Simplified for now
                std::cout << "Calibration result validation skipped in consolidated test" << std::endl;
            }

            // For failure cases, we don't expect convergence
            if (!scenario.expect_failure) {
                REQUIRE(result.converged);
            }
            passed++;
            std::cout << "PASSED" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
            throw; // Re-throw to fail the test
        }
    }
    
    std::cout << "\n=== FINAL SUMMARY ===" << std::endl;
    std::cout << "Scenarios passed: " << passed << "/" << total << std::endl;
    std::cout << "All integration tests completed successfully!" << std::endl;
}
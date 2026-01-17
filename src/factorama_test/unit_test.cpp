#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
// Random things
#include "factorama_test/test_utils.hpp"
#include "factorama/factor_graph.hpp"
#include "factorama/random_utils.hpp"
#include "factorama/numerical_jacobian.hpp"

// Variables
#include "factorama/generic_variable.hpp"
#include "factorama/inverse_range_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/rotation_variable.hpp"


// Factors
#include "factorama/bearing_observation_factor.hpp"
#include "factorama/bearing_projection_factor_2d.hpp"
#include "factorama/generic_between_factor.hpp"
#include "factorama/generic_prior_factor.hpp"
#include "factorama/inverse_range_bearing_factor.hpp"
#include "factorama/pose_between_factors.hpp"
#include "factorama/pose_prior_factors.hpp"
#include "factorama/rotation_prior_factor.hpp"


using namespace factorama;


TEST_CASE("BasicResidualSanityTest", "[residual][factor]")
{

    // Setup: pose at origin with identity rotation
    Eigen::Matrix<double, 6, 1> pose_vec;
    pose_vec.setZero(); // [tx, ty, tz, rx, ry, rz]
    auto pose = std::make_shared<PoseVariable>(1, pose_vec);

    // Landmark directly in front of camera
    Eigen::Vector3d landmark_pos(0.0, 0.0, 5.0);
    auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

    // Perfect bearing vector
    Eigen::Vector3d expected_bearing_C(0.0, 0.0, 1.0);

    // Create factor and graph
    auto factor = std::make_shared<BearingObservationFactor>(0, pose.get(), landmark.get(), expected_bearing_C, 1.0);
    auto prior1 = std::make_shared<PoseOrientationPriorFactor>(1, pose.get(), Eigen::Matrix3d::Identity(), 1.0);
    auto prior2 = std::make_shared<PosePositionPriorFactor>(2, pose.get(), Eigen::Vector3d::Zero(), 1.0);
    auto prior3 = std::make_shared<GenericPriorFactor>(3, landmark.get(), landmark_pos, 1.0);

    FactorGraph graph;
    graph.add_variable(pose);
    graph.add_variable(landmark);
    graph.add_factor(factor);
    graph.add_factor(prior1);
    graph.add_factor(prior2);
    graph.add_factor(prior3);
    graph.finalize_structure();

    SECTION("Residual should be near zero for perfect measurement")
    {
        Eigen::VectorXd residual = graph.compute_full_residual_vector();
        CAPTURE(residual);
        REQUIRE(residual.size() ==
                12); // bearing (3) + pose orientation prior (3) + pose position prior (3) + landmark prior (3)
        REQUIRE(is_approx_equal(residual, Eigen::VectorXd::Zero(12)));
    }

    SECTION("Residual should increase after landmark perturbation")
    {
        // Perturb landmark slightly
        landmark->set_pos_W(Eigen::Vector3d(0.1, 0.0, 5.0));

        Eigen::VectorXd residual = graph.compute_full_residual_vector();

        REQUIRE(residual.size() == 12);

        double norm = residual.norm();
        INFO("Residual norm after perturbation: " << norm);

        // Should be small but non-zero
        REQUIRE(norm > 0.0);
        REQUIRE(norm < 0.15); // sanity bound, adjusted for additional prior residuals
    }
}


TEST_CASE("SO(3) exponential and logarithmic maps round trip", "[SO3]")
{
    const std::vector<Eigen::Vector3d> test_omegas = {
        Eigen::Vector3d::Zero(),       Eigen::Vector3d(0.01, 0, 0),     Eigen::Vector3d(0, 0.01, 0),
        Eigen::Vector3d(0, 0, 0.01),   Eigen::Vector3d(0.1, -0.2, 0.3), Eigen::Vector3d(PI / 2, 0, 0),
        Eigen::Vector3d(0, PI / 2, 0), Eigen::Vector3d(0, 0, PI / 2),
    };

    for (const auto& omega : test_omegas) {
        Eigen::Matrix3d R = ExpMapSO3(omega);
        Eigen::Vector3d recovered = LogMapSO3(R);

        REQUIRE((omega - recovered).norm() < 1e-8);
    }
}

TEST_CASE("SO(3) ExpMap produces orthonormal matrix", "[SO3]")
{
    Eigen::Vector3d omega(0.3, -0.5, 0.2);
    Eigen::Matrix3d R = ExpMapSO3(omega);
    Eigen::Matrix3d I = R.transpose() * R;
    REQUIRE((I - Eigen::Matrix3d::Identity()).norm() < 1e-10);
    REQUIRE(std::abs(R.determinant() - 1.0) < 1e-10);
}

TEST_CASE("SO(3) ExpMap/LogMap π-rotation edge cases", "[SO3][EdgeCase]")
{
    const std::vector<Eigen::Vector3d> axis_vectors = {
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d(1, 1, 1).normalized(),
    };

    for (const auto& axis : axis_vectors) {

        CAPTURE(axis);
        Eigen::Vector3d omega = PI * axis;
        Eigen::Matrix3d R = ExpMapSO3(omega);
        Eigen::Vector3d recovered = LogMapSO3(R);

        CAPTURE(omega);
        CAPTURE(R);
        CAPTURE(recovered);

        // NOTE: LogMap is only defined up to 2π, and may return the negative rotation
        // We'll check that the angle is correct, and the axis is in the right direction (up to sign)
        double original_angle = omega.norm();
        double recovered_angle = recovered.norm();

        bool negative_rotation = std::abs(recovered_angle - original_angle) < 1e-6;
        bool positive_rotation = std::abs(recovered_angle + original_angle) < 1e-6;
        bool ok = negative_rotation || positive_rotation;
        CAPTURE(original_angle);
        CAPTURE(recovered_angle);
        CAPTURE(std::abs(recovered_angle - original_angle));
        CAPTURE(std::abs(recovered_angle + original_angle));
        REQUIRE(ok);
        // REQUIRE(std::abs(recovered_angle - original_angle) < 1e-6 ||
        //         std::abs(recovered_angle + original_angle) < 1e-6);

        if (recovered_angle > 1e-8) {
            Eigen::Vector3d recovered_axis = recovered / recovered_angle;
            REQUIRE(std::abs(recovered_axis.dot(axis)) > 0.999); // cosine near 1
        }
    }
}

TEST_CASE("SO(3) near-zero rotation is stable", "[SO3][EdgeCase]")
{

    std::vector<double> test_cases{0.1, 0.01, 0.00345, 1e-5, 2e-7, 1e-10};

    for (double test_theta : test_cases) {
        Eigen::Vector3d omega(test_theta, -test_theta, test_theta);
        Eigen::Matrix3d R = ExpMapSO3(omega);
        Eigen::Vector3d recovered = LogMapSO3(R);
        CAPTURE(test_theta);
        CAPTURE(omega);
        CAPTURE(recovered);
        REQUIRE((omega - recovered).norm() < 1e-8);
    }
}

TEST_CASE("SO(3) LogMap behaves near 180 degrees", "[SO3][Singularity]")
{
    // R = rotation of 180 deg around X
    Eigen::AngleAxisd aa(PI, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d R = aa.toRotationMatrix();

    Eigen::Vector3d log_result = LogMapSO3(R);
    double angle = log_result.norm();

    // Should be π, axis may be +X or -X
    bool positive_pi = std::abs(angle - PI) < 1e-6;
    bool negative_pi = std::abs(angle + PI) < 1e-6;
    bool ok = positive_pi || negative_pi;
    REQUIRE(ok);
    // REQUIRE(std::abs(angle - PI) < 1e-6 || std::abs(angle + PI) < 1e-6);

    Eigen::Vector3d axis = log_result.normalized();
    REQUIRE(std::abs(std::abs(axis.dot(Eigen::Vector3d::UnitX())) - 1.0) < 1e-6);
}


TEST_CASE("FactorGraph apply_increment", "[FactorGraph][apply_increment]")
{
    SECTION("Basic functionality with landmarks")
    {
        // Create landmark variables (linear increment behavior)
        Eigen::Vector3d landmark1_init(1.0, 2.0, 3.0);
        auto landmark1 = std::make_shared<LandmarkVariable>(1, landmark1_init);

        Eigen::Vector3d landmark2_init(5.0, -1.0, 2.0);
        auto landmark2 = std::make_shared<LandmarkVariable>(2, landmark2_init);

        // Create factor graph
        FactorGraph graph;
        graph.add_variable(landmark1);
        graph.add_variable(landmark2);
        auto prior1 = std::make_shared<GenericPriorFactor>(1, landmark1.get(), landmark1_init, 1.0);
        auto prior2 = std::make_shared<GenericPriorFactor>(2, landmark2.get(), landmark2_init, 1.0);
        graph.add_factor(prior1);
        graph.add_factor(prior2);
        graph.finalize_structure();

        // Create increment vector (landmark1: 3D, landmark2: 3D = 6D total)
        Eigen::VectorXd dx(6);
        dx << 0.1, 0.2, 0.3, // landmark1 increment
            1.0, -0.5, 0.8;  // landmark2 increment

        // Store original values
        auto original_landmark1 = landmark1->value();
        auto original_landmark2 = landmark2->value();

        // Apply increment
        graph.apply_increment(dx);

        // Check that values changed correctly (linear addition for landmarks)
        Eigen::Vector3d expected_landmark1 = original_landmark1 + dx.segment(0, 3);
        REQUIRE(is_approx_equal(landmark1->value(), expected_landmark1));

        Eigen::Vector3d expected_landmark2 = original_landmark2 + dx.segment(3, 3);
        REQUIRE(is_approx_equal(landmark2->value(), expected_landmark2));
    }

    SECTION("Constant variables are skipped")
    {
        // Create variables
        auto var1 = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto var2 = std::make_shared<LandmarkVariable>(2, Eigen::Vector3d(4.0, 5.0, 6.0));

        // Make var1 constant
        var1->set_constant(true);

        FactorGraph graph;
        graph.add_variable(var1);
        graph.add_variable(var2);
        auto prior = std::make_shared<GenericPriorFactor>(1, var2.get(), Eigen::Vector3d(4.0, 5.0, 6.0), 1.0);
        graph.add_factor(prior);
        graph.finalize_structure();

        // Create increment vector (only var2 contributes to num_values)
        Eigen::VectorXd dx(3);
        dx << 10.0, 20.0, 30.0;

        auto original_var1 = var1->value();
        auto original_var2 = var2->value();

        // Apply increment
        graph.apply_increment(dx);

        // var1 should be unchanged, var2 should be updated
        REQUIRE(is_approx_equal(var1->value(), original_var1));
        REQUIRE(is_approx_equal(var2->value(), original_var2 + dx));
    }

    SECTION("Error handling")
    {
        auto var = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(1.0, 2.0, 3.0));
        FactorGraph graph;
        graph.add_variable(var);

        // Test error when structure not finalized
        Eigen::VectorXd dx(3);
        dx.setZero();
        REQUIRE_THROWS(graph.apply_increment(dx));

        auto prior = std::make_shared<GenericPriorFactor>(1, var.get(), Eigen::Vector3d(1.0, 2.0, 3.0), 1.0);
        graph.add_factor(prior);
        graph.finalize_structure();

        // Test error with wrong increment size
        Eigen::VectorXd wrong_size_dx(5);
        wrong_size_dx.setZero();
        REQUIRE_THROWS(graph.apply_increment(wrong_size_dx));

        // Correct size should work
        REQUIRE_NOTHROW(graph.apply_increment(dx));
    }

    SECTION("Integration with get_variable_vector")
    {
        // Test that apply_increment is consistent with get_variable_vector
        auto landmark1 = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto landmark2 = std::make_shared<LandmarkVariable>(2, Eigen::Vector3d(4.0, 5.0, 6.0));

        FactorGraph graph;
        graph.add_variable(landmark1);
        graph.add_variable(landmark2);
        auto prior1 = std::make_shared<GenericPriorFactor>(1, landmark1.get(), Eigen::Vector3d(1.0, 2.0, 3.0), 1.0);
        auto prior2 = std::make_shared<GenericPriorFactor>(2, landmark2.get(), Eigen::Vector3d(4.0, 5.0, 6.0), 1.0);
        graph.add_factor(prior1);
        graph.add_factor(prior2);
        graph.finalize_structure();

        // Get initial variable vector
        Eigen::VectorXd x0 = graph.get_variable_vector();

        // Apply increment
        Eigen::VectorXd dx(6);
        dx << 0.1, 0.2, 0.3, // landmark1
            0.4, 0.5, 0.6;   // landmark2

        graph.apply_increment(dx);

        // Get updated variable vector
        Eigen::VectorXd x1 = graph.get_variable_vector();

        // Should equal initial plus increment (for landmarks this is linear)
        REQUIRE(is_approx_equal(x1, x0 + dx));
    }
}

TEST_CASE("FactorGraph ID Collision Detection", "[factor_graph][error_handling]")
{
    SECTION("Variable ID collision should throw exception")
    {
        FactorGraph graph;

        // Add first variable with ID = 1
        auto var1 = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(1.0, 2.0, 3.0));
        REQUIRE_NOTHROW(graph.add_variable(var1));

        // Try to add second variable with same ID = 1
        auto var2 = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(4.0, 5.0, 6.0));
        REQUIRE_THROWS_AS(graph.add_variable(var2), std::runtime_error);

        // Different ID should work fine
        auto var3 = std::make_shared<LandmarkVariable>(2, Eigen::Vector3d(7.0, 8.0, 9.0));
        REQUIRE_NOTHROW(graph.add_variable(var3));
    }

    SECTION("Factor ID collision should throw exception")
    {
        FactorGraph graph;

        // Set up variables
        auto pose = std::make_shared<PoseVariable>(1, Eigen::Matrix<double, 6, 1>::Zero());
        auto landmark = std::make_shared<LandmarkVariable>(2, Eigen::Vector3d(0.0, 0.0, 5.0));
        graph.add_variable(pose);
        graph.add_variable(landmark);

        // Add first factor with ID = 10
        Eigen::Vector3d bearing(0.0, 0.0, 1.0);
        auto factor1 = std::make_shared<BearingObservationFactor>(10, pose.get(), landmark.get(), bearing, 1.0);
        REQUIRE_NOTHROW(graph.add_factor(factor1));

        // Try to add second factor with same ID = 10
        auto factor2 = std::make_shared<BearingObservationFactor>(10, pose.get(), landmark.get(), bearing, 1.0);
        REQUIRE_THROWS_AS(graph.add_factor(factor2), std::runtime_error);

        // Different ID should work fine
        auto factor3 = std::make_shared<BearingObservationFactor>(11, pose.get(), landmark.get(), bearing, 1.0);
        REQUIRE_NOTHROW(graph.add_factor(factor3));
    }
}

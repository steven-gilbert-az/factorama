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
#include "factorama/linear_velocity_factor.hpp"
#include "factorama/pose_between_factors.hpp"
#include "factorama/pose_prior_factors.hpp"
#include "factorama/rotation_prior_factor.hpp"

using namespace factorama;


TEST_CASE("InverseRangeFactorSanityTest", "[sfm]")
{
    // Ground truth inputs
    Eigen::Vector3d origin_W(0, 0, 0);
    Eigen::Vector3d bearing_W(1, 0, 0); // looking in x direction
    double true_range = 10.0;

    // Camera pose
    Eigen::Vector3d cam_pos_W(5, 1, 0);
    Eigen::Matrix3d dcm_CW = Eigen::Matrix3d::Identity(); // no rotation
    Eigen::Matrix<double, 6, 1> tmp = Eigen::Matrix<double, 6, 1>::Zero();
    std::shared_ptr<PoseVariable> cam_pose = std::make_shared<PoseVariable>(0, tmp);
    cam_pose->set_dcm_CW(dcm_CW);
    cam_pose->set_pos_W(cam_pos_W);

    // Variable with slightly wrong inverse range
    double initial_range = 9.5;
    std::shared_ptr<InverseRangeVariable> inv_range_var = std::make_shared<InverseRangeVariable>(1, origin_W, bearing_W, initial_range);

    // Ground truth landmark position in W
    Eigen::Vector3d landmark_pos_W = origin_W + bearing_W.normalized() * true_range;
    Eigen::Vector3d bearing_C_gt = dcm_CW * (landmark_pos_W - cam_pos_W).normalized();

    // Construct factor
    InverseRangeBearingFactor factor(0, cam_pose.get(), inv_range_var.get(), bearing_C_gt);

    SECTION("Residual is small when guess is close")
    {
        Eigen::VectorXd res = factor.compute_residual();
        REQUIRE(res.size() == 3);
        REQUIRE(res.norm() < 0.2); // sanity check: <~ 1 deg
    }

    SECTION("Jacobian dimensions match expectations")
    {
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 2);
        REQUIRE(jacobians[0].rows() == 3); // w.r.t pose (6)
        REQUIRE(jacobians[0].cols() == 6);
        REQUIRE(jacobians[1].rows() == 3); // w.r.t inv range (1)
        REQUIRE(jacobians[1].cols() == 1);
    }
}


TEST_CASE("PosePositionPriorFactor: dimensions and residual")
{
    Eigen::Matrix<double, 6, 1> pose_init;
    pose_init.head<3>() = Eigen::Vector3d(1.0, 2.0, 3.0); // pos_W
    pose_init.tail<3>() = Eigen::Vector3d(0.1, 0.2, 0.3); // rotvec
    auto pose = std::make_shared<PoseVariable>(0, pose_init);

    Eigen::Vector3d pos_prior(0.5, 2.0, 2.5);
    // double info_mag = 4.0;
    double sigma = 0.5;

    PosePositionPriorFactor factor(0, pose.get(), pos_prior, sigma);

    SECTION("Residual dimension is correct")
    {
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);
    }

    SECTION("Jacobian dimension is correct")
    {
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].rows() == 3);
        REQUIRE(jacobians[0].cols() == 6);
    }

    SECTION("Residual value is correct")
    {
        Eigen::Vector3d expected = (1.0 / sigma) * (pose->pos_W() - pos_prior);
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.isApprox(expected, 1e-9));
    }
}

TEST_CASE("GenericPriorFactor: dimensions and residual with LandmarkVariable")
{
    Eigen::Vector3d pos_init(1.0, 2.0, 3.0);
    auto landmark = std::make_shared<LandmarkVariable>(42, pos_init);

    Eigen::VectorXd prior(3);
    prior << 0.5, 2.0, 2.5;

    // double info_mag = 4.0;
    double sigma = 0.5;

    GenericPriorFactor factor(0, landmark.get(), prior, sigma);

    SECTION("Residual dimension is correct")
    {
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);
    }

    SECTION("Jacobian dimension is correct")
    {
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].rows() == 3);
        REQUIRE(jacobians[0].cols() == 3);
    }

    SECTION("Residual value is correct")
    {
        Eigen::VectorXd expected = (1.0 / sigma) * (landmark->value() - prior);
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.isApprox(expected, 1e-9));
    }
}

TEST_CASE("BearingObservationFactor: analytical Jacobian matches numerical", "[jacobian][bearing]")
{
    SECTION("Linear mode (existing test)")
    {
        // Setup
        Eigen::Vector3d pos_W(0, 0, 0);
        Eigen::Vector3d rot_W(0, 0, 0); // No rotation
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);
        auto landmark = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(1, 0, 0));

        Eigen::Vector3d bearing_C = (Eigen::Vector3d(1, 0, 0)).normalized();
        double sigma = 0.1;

        auto factor = std::make_shared<BearingObservationFactor>(0, pose.get(), landmark.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            CAPTURE(i);
            CAPTURE(J_analytic[i]);
            CAPTURE(J_numeric[i]);
            CAPTURE((J_analytic[i] - J_numeric[i]).norm());
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }

    SECTION("SO(3) manifold mode: Non-identity rotation with sigma=0.05")
    {
        // Setup with significant rotation
        Eigen::Vector3d pos_W(1.5, -0.8, 0.3);
        Eigen::Vector3d rot_W(0.3, -0.2, 0.15); // Non-trivial rotation
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);
        auto landmark = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(2.8, 1.2, 3.5));

        Eigen::Vector3d bearing_C = Eigen::Vector3d(0.6, -0.3, 0.8).normalized();
        double sigma = 0.05; // Non-unit sigma

        auto factor = std::make_shared<BearingObservationFactor>(0, pose.get(), landmark.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            CAPTURE(i);
            CAPTURE(J_analytic[i]);
            CAPTURE(J_numeric[i]);
            CAPTURE((J_analytic[i] - J_numeric[i]).norm());
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }

    SECTION("SO(3) manifold mode: Different rotation with sigma=0.2")
    {
        // Setup with different rotation and sigma
        Eigen::Vector3d pos_W(-0.5, 2.1, -1.0);
        Eigen::Vector3d rot_W(-0.1, 0.4, -0.25);
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);
        auto landmark = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(-1.2, 0.8, 4.2));

        Eigen::Vector3d bearing_C = Eigen::Vector3d(-0.4, 0.7, 0.6).normalized();
        double sigma = 0.2;

        auto factor = std::make_shared<BearingObservationFactor>(0, pose.get(), landmark.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }
}

TEST_CASE("InverseRangeBearingFactor: analytical Jacobian matches numerical", "[jacobian][inv_range]")
{
    SECTION("Linear mode (existing test)")
    {
        // Setup
        Eigen::Vector3d cam_pos_W(0, 0, 0);
        Eigen::Vector3d rot_W(0, 0, 0); // No rotation
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << cam_pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);

        Eigen::Vector3d bearing_W = Eigen::Vector3d(1, 1, 0).normalized();
        double range = 2.0;

        auto inv_range_var = std::make_shared<InverseRangeVariable>(
            1, cam_pos_W, bearing_W, range);

        Eigen::Vector3d lm_pos = cam_pos_W + range * bearing_W;
        Eigen::Matrix3d dcm_CW = pose->dcm_CW();
        Eigen::Vector3d bearing_C = dcm_CW * (lm_pos - cam_pos_W).normalized();

        double sigma = 0.1;
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose.get(), inv_range_var.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }

    SECTION("SO(3) manifold mode: Non-trivial rotation with sigma=0.08")
    {
        // Setup with significant rotation
        Eigen::Vector3d cam_pos_W(0.8, -1.2, 0.5);
        Eigen::Vector3d rot_W(0.25, -0.15, 0.3);
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << cam_pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);

        Eigen::Vector3d bearing_W = Eigen::Vector3d(0.6, 0.8, 0.1).normalized();
        double range = 3.5;

        auto inv_range_var = std::make_shared<InverseRangeVariable>(1, cam_pos_W, bearing_W, range);

        Eigen::Vector3d lm_pos = cam_pos_W + range * bearing_W;
        Eigen::Matrix3d dcm_CW = pose->dcm_CW();
        Eigen::Vector3d bearing_C = dcm_CW * (lm_pos - cam_pos_W).normalized();

        double sigma = 0.08;
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose.get(), inv_range_var.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            CAPTURE(i);
            CAPTURE((J_analytic[i] - J_numeric[i]).norm());
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }

    SECTION("SO(3) manifold mode: Different configuration with sigma=0.15")
    {
        // Different test configuration
        Eigen::Vector3d cam_pos_W(-0.3, 0.9, -0.6);
        Eigen::Vector3d rot_W(-0.2, 0.1, -0.35);
        Eigen::Matrix<double, 6, 1> cam_pose;
        cam_pose << cam_pos_W, rot_W;

        auto pose = std::make_shared<PoseVariable>(0, cam_pose);

        Eigen::Vector3d bearing_W = Eigen::Vector3d(-0.4, 0.5, 0.7).normalized();
        double range = 2.8;

        auto inv_range_var = std::make_shared<InverseRangeVariable>(1, cam_pos_W, bearing_W, range);

        Eigen::Vector3d lm_pos = cam_pos_W + range * bearing_W;
        Eigen::Matrix3d dcm_CW = pose->dcm_CW();
        Eigen::Vector3d bearing_C = dcm_CW * (lm_pos - cam_pos_W).normalized();

        double sigma = 0.15;
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose.get(), inv_range_var.get(), bearing_C, sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }
}


TEST_CASE("PoseOrientationBetweenFactor residual and jacobian numeric check", "[PoseOrientationBetweenFactor][numeric]")
{
    SECTION("Basic configuration (existing test)")
    {
        // Setup two camera poses with identity and 90deg rotation about Z
        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R2;
        R2 = Eigen::AngleAxisd(PI / 2, Eigen::Vector3d::UnitZ());

        auto pose1 = std::make_shared<PoseVariable>(1, Eigen::Vector3d::Zero(), R1);
        auto pose2 = std::make_shared<PoseVariable>(2, Eigen::Vector3d::Zero(), R2);

        // Initial extrinsic rotation guess - identity
        auto extrinsic = std::make_shared<RotationVariable>(3, Eigen::Matrix3d::Identity());

        PoseOrientationBetweenFactor factor(0, pose1.get(), pose2.get(), extrinsic.get());

        // Compute residual
        Eigen::VectorXd residual = factor.compute_residual();

        INFO("Residual vector: " << residual.transpose());

        // Residual norm should be non-negative and finite
        REQUIRE(std::isfinite(residual.norm()));
        REQUIRE(residual.size() == 3);

        // Numeric Jacobian check for each variable
        std::vector<Variable *> vars = factor.variables();
        std::vector<Eigen::MatrixXd> analytic_jacs(3);
        factor.compute_jacobians(analytic_jacs);

        REQUIRE(analytic_jacs.size() == vars.size());
        std::vector<Eigen::MatrixXd> numeric_jacs;
        ComputeNumericalJacobians(factor, numeric_jacs);

        for (size_t i = 0; i < vars.size(); ++i)
        {

            INFO("Variable " << i << " analytic jacobian:\n"
                             << analytic_jacs[i]);
            INFO("Variable " << i << " numeric jacobian:\n"
                             << numeric_jacs[i]);

            // Compare numeric and analytic Jacobians element-wise
            CAPTURE(analytic_jacs[i]);
            CAPTURE(numeric_jacs[i]);

            double total_diff = (analytic_jacs[i] - numeric_jacs[i]).array().abs().sum();
            CAPTURE(total_diff);

            REQUIRE(is_approx_equal(analytic_jacs[i], numeric_jacs[i], 1e-6));
        }
    }

    SECTION("SO(3) manifold mode: Non-trivial rotations with sigma=0.75")
    {
        // Setup two poses with different rotations
        Eigen::Vector3d pos1_W(1.0, 0.5, 0.2);
        Eigen::Vector3d rot1_W(0.1, -0.2, 0.15);
        Eigen::Matrix<double, 6, 1> pose1_vec;
        pose1_vec << pos1_W, rot1_W;
        auto pose1 = std::make_shared<PoseVariable>(0, pose1_vec);

        Eigen::Vector3d pos2_W(1.2, 0.8, 0.3);
        Eigen::Vector3d rot2_W(0.05, 0.25, 0.18);
        Eigen::Matrix<double, 6, 1> pose2_vec;
        pose2_vec << pos2_W, rot2_W;
        auto pose2 = std::make_shared<PoseVariable>(1, pose2_vec);

        // Extrinsic rotation between sensor frames
        Eigen::Vector3d extrinsic_rot_vec(0.02, 0.03, -0.01);
        Eigen::Matrix3d extrinsic_rot = ExpMapSO3(extrinsic_rot_vec);
        auto extrinsic_var = std::make_shared<RotationVariable>(2, extrinsic_rot);

        double sigma = 0.75;
        auto factor = std::make_shared<PoseOrientationBetweenFactor>(0, pose1.get(), pose2.get(), extrinsic_var.get(), sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        // This should pass since PoseOrientationBetweenFactor uses numerical jacobians
        for (size_t i = 0; i < J_analytic.size(); ++i)
        {

            CAPTURE(i);
            CAPTURE((J_analytic[i] - J_numeric[i]).norm());
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-4); // Looser tolerance for numerical vs numerical
        }
    }

    SECTION("SO(3) manifold mode: Different configuration with sigma=0.12")
    {
        // Different test configuration
        Eigen::Vector3d pos1_W(-0.5, 1.2, -0.3);
        Eigen::Vector3d rot1_W(-0.08, 0.12, -0.05);
        Eigen::Matrix<double, 6, 1> pose1_vec;
        pose1_vec << pos1_W, rot1_W;
        auto pose1 = std::make_shared<PoseVariable>(0, pose1_vec);

        Eigen::Vector3d pos2_W(-0.2, 1.5, -0.1);
        Eigen::Vector3d rot2_W(-0.1, 0.08, -0.02);
        Eigen::Matrix<double, 6, 1> pose2_vec;
        pose2_vec << pos2_W, rot2_W;
        auto pose2 = std::make_shared<PoseVariable>(1, pose2_vec);

        // Different extrinsic rotation
        Eigen::Vector3d extrinsic_rot_vec(-0.015, 0.025, 0.008);
        Eigen::Matrix3d extrinsic_rot = ExpMapSO3(extrinsic_rot_vec);
        auto extrinsic_var = std::make_shared<RotationVariable>(2, extrinsic_rot);

        double sigma = 0.12;
        auto factor = std::make_shared<PoseOrientationBetweenFactor>(0, pose1.get(), pose2.get(), extrinsic_var.get(), sigma);

        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;

        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);

        REQUIRE(J_analytic.size() == J_numeric.size());

        // This should pass since PoseOrientationBetweenFactor uses numerical jacobians
        for (size_t i = 0; i < J_analytic.size(); ++i)
        {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-4); // Looser tolerance for numerical vs numerical
        }
    }
}


TEST_CASE("GenericBetweenFactor residuals and jacobians with LandmarkVariable", "[GenericBetweenFactor]")
{
    using namespace factorama;

    auto make_landmark = [](int id, const Eigen::Vector3d &pos)
    {
        auto l = std::make_shared<LandmarkVariable>(id, pos);
        l->set_is_constant(false);
        return l;
    };

    SECTION("Exact match should yield zero residual")
    {
        auto a = make_landmark(0, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto b = make_landmark(1, Eigen::Vector3d(4.0, 6.0, 9.0));
        Eigen::Vector3d expected_vec = b->value() - a->value();
        auto expected = std::make_shared<GenericVariable>(42, expected_vec);
        expected->set_is_constant(true);

        GenericBetweenFactor factor(42, a.get(), b.get(), expected.get());

        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero(), precision_tol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Offset between landmarks with zero expected")
    {
        auto a = make_landmark(2, Eigen::Vector3d(0.0, 0.0, 0.0));
        auto b = make_landmark(3, Eigen::Vector3d(1.0, -2.0, 3.0));
        Eigen::Vector3d expected_vec(0.0, 0.0, 0.0); // Expecting zero offset
        auto expected = std::make_shared<GenericVariable>(43, expected_vec);
        expected->set_is_constant(true);
        double sigma = 1.0;

        GenericBetweenFactor factor(43, a.get(), b.get(), expected.get(), sigma);

        Eigen::VectorXd residual = factor.compute_residual();
        Eigen::Vector3d expected_residual = b->value() - a->value();
        REQUIRE(is_approx_equal(residual, expected_residual, precision_tol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Residual scaled by sigma")
    {
        auto a = make_landmark(4, Eigen::Vector3d(1.0, 1.0, 1.0));
        auto b = make_landmark(5, Eigen::Vector3d(2.0, 2.0, 2.0));
        Eigen::Vector3d expected_vec(0.0, 0.0, 0.0);
        auto expected = std::make_shared<GenericVariable>(44, expected_vec);
        expected->set_is_constant(true);

        double sigma = 0.5; // weight = 2.0
        GenericBetweenFactor factor(44, a.get(), b.get(), expected.get(), sigma);

        Eigen::VectorXd residual = factor.compute_residual();
        Eigen::Vector3d expected_residual = 2.0 * (b->value() - a->value());
        REQUIRE(is_approx_equal(residual, expected_residual, precision_tol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -2.0 * Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(is_approx_equal(J[1], 2.0 * Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Skip Jacobian for constant variable A")
    {
        auto a = make_landmark(6, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto b = make_landmark(7, Eigen::Vector3d(2.0, 3.0, 4.0));
        a->set_is_constant(true);
        auto expected = std::make_shared<GenericVariable>(45, Eigen::Vector3d::Zero());
        expected->set_is_constant(true);

        GenericBetweenFactor factor(45, a.get(), b.get(), expected.get());

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(J[0].size() == 0); // Should be skipped
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Skip Jacobian for constant variable B")
    {
        auto a = make_landmark(8, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto b = make_landmark(9, Eigen::Vector3d(4.0, 5.0, 6.0));
        b->set_is_constant(true);
        auto expected = std::make_shared<GenericVariable>(46, Eigen::Vector3d(1.0, 2.0, 3.0));
        expected->set_is_constant(true);

        GenericBetweenFactor factor(46, a.get(), b.get(), expected.get());

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), precision_tol));
        REQUIRE(J[1].size() == 0); // Should be skipped
        REQUIRE(J[2].size() == 0); // expected is constant
    }
}

TEST_CASE("PosePositionBetweenFactor computes correct residuals and jacobians")
{
    using namespace factorama;

    // Pose A at (0,0,0), identity rotation
    auto pose_a = std::make_shared<PoseVariable>(0, Eigen::Vector3d(0, 0, 0), Eigen::Matrix3d::Identity());

    // Pose B at (1,2,3), identity rotation
    auto pose_b = std::make_shared<PoseVariable>(1, Eigen::Vector3d(1, 2, 3), Eigen::Matrix3d::Identity());

    // Measured relative offset from A to B
    Eigen::Vector3d measured_vec(1, 2, 3);
    auto measured = std::make_shared<GenericVariable>(100, measured_vec);
    measured->set_is_constant(true);

    // --------------------------
    SECTION("Residual is zero when measurement matches")
    {
        PosePositionBetweenFactor factor(100, pose_a.get(), pose_b.get(), measured.get(), 1.0);
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero()));
    }

    // --------------------------
    SECTION("Residual is non-zero with offset mismatch")
    {
        Eigen::Vector3d measured_vec2(1, 2, 3); // expected delta
        auto measured2 = std::make_shared<GenericVariable>(101, measured_vec2);
        measured2->set_is_constant(true);

        // Perturb pose B
        pose_b->set_pos_W(Eigen::Vector3d(2, 1, 4)); // actual delta is (2,1,4)

        PosePositionBetweenFactor factor(101, pose_a.get(), pose_b.get(), measured2.get(), 1.0);
        Eigen::VectorXd residual = factor.compute_residual();

        Eigen::Vector3d expected_residual = Eigen::Vector3d(2, 1, 4) - measured_vec2; // (1,-1,1)
        REQUIRE(is_approx_equal(residual, expected_residual));
    }

    // --------------------------
    SECTION("Jacobian structure and sign are correct")
    {
        PosePositionBetweenFactor factor(102, pose_a.get(), pose_b.get(), measured.get(), 1.0);

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);

        REQUIRE(J.size() == 3);
        REQUIRE(J[0].rows() == 3);
        REQUIRE(J[0].cols() == 6);
        REQUIRE(J[1].rows() == 3);
        REQUIRE(J[1].cols() == 6);

        Eigen::MatrixXd expected_Ja = Eigen::MatrixXd::Zero(3, 6);
        expected_Ja.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        Eigen::MatrixXd expected_Jb = Eigen::MatrixXd::Zero(3, 6);
        expected_Jb.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        REQUIRE(is_approx_equal(J[0], expected_Ja));
        REQUIRE(is_approx_equal(J[1], expected_Jb));
        REQUIRE(J[2].size() == 0); // measured is constant
    }

    // --------------------------
    SECTION("Jacobian respects constant variables")
    {
        pose_a->set_is_constant(true);
        PosePositionBetweenFactor factor(103, pose_a.get(), pose_b.get(), measured.get(), 1.0);

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);

        REQUIRE(J.size() == 3);
        REQUIRE(J[0].size() == 0); // Empty matrix for constant
        REQUIRE(J[1].rows() == 3);
        REQUIRE(J[1].cols() == 6);
        REQUIRE(J[2].size() == 0); // measured is constant
    }

    // --------------------------
    SECTION("Residual and Jacobians respect sigma weight")
    {
        double sigma = 2.0;
        PosePositionBetweenFactor factor(104, pose_a.get(), pose_b.get(), measured.get(), sigma);

        Eigen::VectorXd residual = factor.compute_residual();
        double expected_scale = 1.0 / sigma;
        REQUIRE(is_approx_equal(residual, expected_scale * (pose_b->pos_W() - pose_a->pos_W() - measured->value())));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);

        Eigen::MatrixXd expected_Ja = Eigen::MatrixXd::Zero(3, 6);
        expected_Ja.block<3, 3>(0, 0) = -expected_scale * Eigen::Matrix3d::Identity();
        Eigen::MatrixXd expected_Jb = Eigen::MatrixXd::Zero(3, 6);
        expected_Jb.block<3, 3>(0, 0) = expected_scale * Eigen::Matrix3d::Identity();

        REQUIRE(is_approx_equal(J[0], expected_Ja));
        REQUIRE(is_approx_equal(J[1], expected_Jb));
        REQUIRE(J[2].size() == 0); // measured is constant
    }
}

TEST_CASE("PosePositionPriorFactor: comprehensive behavior tests", "[prior][position]")
{
    SECTION("Basic residual and jacobian computation")
    {
        // Setup pose with non-zero position and rotation
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 2.0, 1.5, -0.5, 0.1, 0.2, 0.3;
        auto pose = std::make_shared<PoseVariable>(0, pose_init);

        Eigen::Vector3d pos_prior(1.0, 2.0, 0.0);
        double sigma = 0.2;

        PosePositionPriorFactor factor(0, pose.get(), pos_prior, sigma);

        // Check residual
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);

        Eigen::Vector3d expected_residual = (1.0 / sigma) * (pose->pos_W() - pos_prior);
        REQUIRE(is_approx_equal(residual, expected_residual, precision_tol));

        // Check Jacobian
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].rows() == 3);
        REQUIRE(jacobians[0].cols() == 6);

        // Expected Jacobian: derivative of position residual w.r.t. pose
        Eigen::MatrixXd expected_J = Eigen::MatrixXd::Zero(3, 6);
        expected_J.block<3, 3>(0, 0) = (1.0 / sigma) * Eigen::Matrix3d::Identity();
        REQUIRE(is_approx_equal(jacobians[0], expected_J, precision_tol));
    }

    SECTION("Zero residual for exact position match")
    {
        Eigen::Vector3d pos_exact(3.0, -1.0, 2.5);
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << pos_exact, 0.0, 0.0, 0.0; // position matches exactly
        auto pose = std::make_shared<PoseVariable>(0, pose_init);

        PosePositionPriorFactor factor(0, pose.get(), pos_exact, 1.0);

        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero(), precision_tol));
    }

    SECTION("Different sigma values affect residual and jacobian scaling")
    {
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
        auto pose = std::make_shared<PoseVariable>(0, pose_init);

        Eigen::Vector3d pos_prior(0.0, 0.0, 0.0);

        // Test different sigma values
        std::vector<double> sigma_values = {0.1, 1.0, 2.0, 5.0};

        for (double sigma : sigma_values)
        {
            PosePositionPriorFactor factor(0, pose.get(), pos_prior, sigma);

            Eigen::VectorXd residual = factor.compute_residual();
            double expected_scale = 1.0 / sigma;
            Eigen::Vector3d expected_residual = expected_scale * (pose->pos_W() - pos_prior);

            REQUIRE(is_approx_equal(residual, expected_residual, precision_tol));

            // Check Jacobian scaling
            std::vector<Eigen::MatrixXd> jacobians;
            factor.compute_jacobians(jacobians);

            Eigen::MatrixXd expected_J = Eigen::MatrixXd::Zero(3, 6);
            expected_J.block<3, 3>(0, 0) = expected_scale * Eigen::Matrix3d::Identity();
            REQUIRE(is_approx_equal(jacobians[0], expected_J, precision_tol));
        }
    }

    SECTION("Constant pose variable handling")
    {
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;
        auto pose = std::make_shared<PoseVariable>(0, pose_init);
        pose->set_is_constant(true);

        Eigen::Vector3d pos_prior(0.0, 0.0, 0.0);
        PosePositionPriorFactor factor(0, pose.get(), pos_prior, 1.0);

        // Residual should still be computed
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);
        REQUIRE(is_approx_equal(residual, pose->pos_W() - pos_prior, precision_tol));

        // Jacobian should be empty for constant variables
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].size() == 0);
    }
}

TEST_CASE("PosePositionPriorFactor: analytical vs numerical Jacobians", "[prior][position][jacobian]")
{
    // Setup
    Eigen::Matrix<double, 6, 1> pose_init;
    pose_init << 2.0, -1.0, 0.5, 0.1, 0.2, 0.3;
    auto pose = std::make_shared<PoseVariable>(0, pose_init);

    Eigen::Vector3d pos_prior(1.0, 0.0, 1.0);
    double sigma = 0.5;

    auto factor = std::make_shared<PosePositionPriorFactor>(0, pose.get(), pos_prior, sigma);

    std::vector<Eigen::MatrixXd> J_analytic;
    std::vector<Eigen::MatrixXd> J_numeric;

    factor->compute_jacobians(J_analytic);
    ComputeNumericalJacobians(*factor, J_numeric);

    REQUIRE(J_analytic.size() == J_numeric.size());
    REQUIRE(J_analytic.size() == 1);

    REQUIRE(J_analytic[0].rows() == J_numeric[0].rows());
    REQUIRE(J_analytic[0].cols() == J_numeric[0].cols());
    REQUIRE(J_analytic[0].rows() == 3);
    REQUIRE(J_analytic[0].cols() == 6);

    // Compare analytical vs numerical
    REQUIRE((J_analytic[0] - J_numeric[0]).norm() < 1e-6);
}

TEST_CASE("PoseOrientationPriorFactor: analytical vs numerical Jacobians", "[prior][orientation][jacobian]")
{
    // Setup
    Eigen::Matrix<double, 6, 1> pose_init;
    pose_init << 1.0, 2.0, 3.0, 0.15, -0.1, 0.25;
    auto pose = std::make_shared<PoseVariable>(0, pose_init);

    Eigen::Vector3d rot_prior(0.1, 0.0, 0.2);
    Eigen::Matrix3d dcm_prior = ExpMapSO3(rot_prior);
    double sigma = 0.3;

    auto factor = std::make_shared<PoseOrientationPriorFactor>(0, pose.get(), dcm_prior, sigma);

    std::vector<Eigen::MatrixXd> J_analytic;
    std::vector<Eigen::MatrixXd> J_numeric;

    factor->compute_jacobians(J_analytic);
    ComputeNumericalJacobians(*factor, J_numeric);

    REQUIRE(J_analytic.size() == J_numeric.size());
    REQUIRE(J_analytic.size() == 1);

    REQUIRE(J_analytic[0].rows() == J_numeric[0].rows());
    REQUIRE(J_analytic[0].cols() == J_numeric[0].cols());
    REQUIRE(J_analytic[0].rows() == 3);
    REQUIRE(J_analytic[0].cols() == 6);

    // Compare analytical vs numerical
    Eigen::MatrixXd diff = J_analytic[0] - J_numeric[0];
    double diff_norm = diff.norm();

    if (diff_norm >= 1e-6)
    {
        std::cout << "\nPoseOrientationPriorFactor Jacobian diff (norm: " << diff_norm << "):\n"
                  << diff << std::endl;
        std::cout << "Analytical Jacobian:\n"
                  << J_analytic[0] << std::endl;
        std::cout << "Numerical Jacobian:\n"
                  << J_numeric[0] << std::endl;
        std::cout << "pose rotation: " << pose->rot_CW().transpose() << std::endl;
        std::cout << "rot_prior: " << rot_prior.transpose() << std::endl;
    }

    REQUIRE(diff_norm < 1e-6);
}


// ============================================================================
// BearingProjectionFactor2D Tests
// ============================================================================

TEST_CASE("BearingProjectionFactor2D: Basic functionality and residual computation", "[bearing_projection_2d][residual]")
{
    // Setup: camera at origin, looking down +Z axis
    Eigen::Matrix<double, 6, 1> pose_vec = Eigen::Matrix<double, 6, 1>::Zero();
    auto pose = std::make_shared<PoseVariable>(1, pose_vec);

    // Landmark at (1, 0, 5) - offset to the right
    Eigen::Vector3d landmark_pos(1.0, 0.0, 5.0);
    auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

    // Perfect bearing measurement pointing toward the landmark
    Eigen::Vector3d bearing_k(1.0, 0.0, 5.0);
    bearing_k.normalize();

    SECTION("Perfect measurement should give near-zero residual")
    {
        auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_k, 1.0);

        Eigen::VectorXd residual = factor->compute_residual();
        REQUIRE(residual.size() == 2);
        REQUIRE(residual.norm() < precision_tol);
    }

    SECTION("Misaligned measurement should give non-zero residual")
    {
        // Introduce bearing error
        Eigen::Vector3d wrong_bearing(0.0, 0.0, 1.0); // straight ahead instead of angled
        auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), wrong_bearing, 1.0);

        Eigen::VectorXd residual = factor->compute_residual();
        REQUIRE(residual.size() == 2);
        REQUIRE(residual.norm() > 1e-6); // Should be significantly non-zero
    }

    SECTION("Weight scaling works correctly")
    {
        double sigma = 0.5;
        double expected_weight = 1.0 / sigma;
        auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_k, sigma);

        REQUIRE(std::abs(factor->weight() - expected_weight) < precision_tol);

        // Residual should be scaled by weight
        Eigen::Vector3d wrong_bearing(0.0, 0.0, 1.0);
        auto factor_wrong = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), wrong_bearing, sigma);
        Eigen::VectorXd weighted_residual = factor_wrong->compute_residual();

        // Compare with unit weight version
        auto factor_unit = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), wrong_bearing, 1.0);
        Eigen::VectorXd unit_residual = factor_unit->compute_residual();

        Eigen::VectorXd expected_weighted = expected_weight * unit_residual;
        REQUIRE(is_approx_equal(weighted_residual, expected_weighted, precision_tol));
    }
}

TEST_CASE("BearingProjectionFactor2D: Jacobian accuracy against numerical differentiation", "[bearing_projection_2d][jacobian]")
{
    // Setup with non-trivial pose and landmark positions
    Eigen::Matrix<double, 6, 1> pose_vec;
    pose_vec << 0.5, -0.3, 0.2, 0.1, -0.05, 0.15; // [tx, ty, tz, rx, ry, rz]
    auto pose = std::make_shared<PoseVariable>(1, pose_vec);

    Eigen::Vector3d landmark_pos(2.0, 1.5, 4.0);
    auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

    // Bearing measurement with some reasonable direction
    Eigen::Vector3d bearing_k(0.3, 0.4, 0.8);
    bearing_k.normalize();

    auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_k, 1.0);

    SECTION("Analytic vs numerical Jacobians should match")
    {
        // Compute analytic Jacobians
        std::vector<Eigen::MatrixXd> analytic_jacobians;
        factor->compute_jacobians(analytic_jacobians);

        // Compute numerical Jacobians
        std::vector<Eigen::MatrixXd> numerical_jacobians;
        ComputeNumericalJacobians(*factor, numerical_jacobians, 1e-6);

        REQUIRE(analytic_jacobians.size() == 2);
        REQUIRE(numerical_jacobians.size() == 2);

        // Check pose Jacobian (2x6)
        REQUIRE(analytic_jacobians[0].rows() == 2);
        REQUIRE(analytic_jacobians[0].cols() == 6);
        REQUIRE(numerical_jacobians[0].rows() == 2);
        REQUIRE(numerical_jacobians[0].cols() == 6);

        for (int i = 0; i < analytic_jacobians[0].rows(); ++i)
        {
            for (int j = 0; j < analytic_jacobians[0].cols(); ++j)
            {
                REQUIRE(std::abs(analytic_jacobians[0](i, j) - numerical_jacobians[0](i, j)) < 1e-6);
            }
        }

        // Check landmark Jacobian (2x3)
        REQUIRE(analytic_jacobians[1].rows() == 2);
        REQUIRE(analytic_jacobians[1].cols() == 3);
        REQUIRE(numerical_jacobians[1].rows() == 2);
        REQUIRE(numerical_jacobians[1].cols() == 3);

        for (int i = 0; i < analytic_jacobians[1].rows(); ++i)
        {
            for (int j = 0; j < analytic_jacobians[1].cols(); ++j)
            {
                REQUIRE(std::abs(analytic_jacobians[1](i, j) - numerical_jacobians[1](i, j)) < 1e-6);
            }
        }
    }

    SECTION("Multiple random configurations should all pass Jacobian test")
    {
        // Test several random configurations to ensure robustness
        for (int test_idx = 0; test_idx < 5; ++test_idx)
        {
            // Random pose
            Eigen::Matrix<double, 6, 1> rand_pose = Eigen::Matrix<double, 6, 1>::Random() * 0.5;
            pose->set_value_from_vector(rand_pose);

            // Random landmark (keep reasonable distance from camera)
            Eigen::Vector3d rand_landmark = Eigen::Vector3d::Random() * 2.0 + Eigen::Vector3d(0, 0, 3);
            landmark->set_pos_W(rand_landmark);

            // Compute Jacobians
            std::vector<Eigen::MatrixXd> analytic_jac, numerical_jac;
            factor->compute_jacobians(analytic_jac);
            ComputeNumericalJacobians(*factor, numerical_jac, 1e-6);

            // Check agreement
            for (int i = 0; i < 2; ++i)
            {
                for (int r = 0; r < analytic_jac[i].rows(); ++r)
                {
                    for (int c = 0; c < analytic_jac[i].cols(); ++c)
                    {
                        double diff = std::abs(analytic_jac[i](r, c) - numerical_jac[i](r, c));
                        REQUIRE(diff < 1e-5);
                    }
                }
            }
        }
    }
}

TEST_CASE("BearingProjectionFactor2D: Edge case handling for small alpha", "[bearing_projection_2d][edge_cases]")
{
    // Setup where landmark is behind or very close to the camera
    Eigen::Matrix<double, 6, 1> pose_vec = Eigen::Matrix<double, 6, 1>::Zero();
    auto pose = std::make_shared<PoseVariable>(1, pose_vec);

    SECTION("Landmark behind camera (negative alpha)")
    {
        // Landmark behind the camera
        Eigen::Vector3d landmark_pos(0.0, 0.0, -2.0);
        auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

        // Bearing pointing forward (but landmark is behind)
        Eigen::Vector3d bearing_k(0.0, 0.0, 1.0);

        double eps = 1e-6;
        auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_k, 1.0, eps);

        // Should return large residual and not crash
        Eigen::VectorXd residual = factor->compute_residual();
        REQUIRE(residual.size() == 2);
        REQUIRE(residual.norm() > 1.0); // Should be large

        // Jacobians should be zero/small
        std::vector<Eigen::MatrixXd> jacobians;
        factor->compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 2);
        REQUIRE(jacobians[0].norm() < 1e-3); // Should be near zero
        REQUIRE(jacobians[1].norm() < 1e-3); // Should be near zero
    }

    SECTION("Landmark very close to camera (small positive alpha)")
    {
        // Landmark very close but slightly in front
        Eigen::Vector3d landmark_pos(0.0, 0.0, 1e-8);
        auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);

        Eigen::Vector3d bearing_k(0.0, 0.0, 1.0);

        double eps = 1e-6;
        auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_k, 1.0, eps);

        // Should handle gracefully
        Eigen::VectorXd residual = factor->compute_residual();
        REQUIRE(residual.size() == 2);
        REQUIRE(std::isfinite(residual.norm())); // Should not be NaN/inf

        std::vector<Eigen::MatrixXd> jacobians;
        REQUIRE_NOTHROW(factor->compute_jacobians(jacobians));
    }
}

TEST_CASE("BearingProjectionFactor2D: Integration test with optimization convergence", "[bearing_projection_2d][integration]")
{
    // Setup initial guess with some error
    Eigen::Matrix<double, 6, 1> initial_pose;
    initial_pose << 0.1, -0.05, 0.02, 0.03, -0.01, 0.04; // small errors
    auto pose = std::make_shared<PoseVariable>(1, initial_pose);

    // True landmark position
    Eigen::Vector3d true_landmark(1.5, 1.0, 4.0);
    // Initial landmark guess with error
    Eigen::Vector3d initial_landmark = true_landmark + Eigen::Vector3d(0.2, -0.1, 0.3);
    auto landmark = std::make_shared<LandmarkVariable>(2, initial_landmark);

    // Generate "measurement" from true values
    Eigen::Matrix<double, 6, 1> true_pose = Eigen::Matrix<double, 6, 1>::Zero();
    auto true_pose_var = std::make_shared<PoseVariable>(999, true_pose); // temporary for measurement generation

    // Compute expected bearing from true pose to true landmark
    Eigen::Vector3d delta_W = true_landmark - true_pose_var->pos_W();
    Eigen::Vector3d bearing_C_true = true_pose_var->dcm_CW() * delta_W;
    bearing_C_true.normalize();

    // Create factor with this "perfect" measurement
    auto factor = std::make_shared<BearingProjectionFactor2D>(0, pose.get(), landmark.get(), bearing_C_true, 0.1);

    SECTION("Single iteration should reduce residual")
    {
        // Compute initial residual
        Eigen::VectorXd initial_residual = factor->compute_residual();
        double initial_cost = 0.5 * initial_residual.squaredNorm();

        // Compute Jacobians for Gauss-Newton step
        std::vector<Eigen::MatrixXd> jacobians;
        factor->compute_jacobians(jacobians);

        // Simple Gauss-Newton step (assuming single factor)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(9, 9); // 6 (pose) + 3 (landmark)
        Eigen::VectorXd b = Eigen::VectorXd::Zero(9);

        // Accumulate pose contribution
        if (jacobians[0].size() > 0)
        {
            H.block<6, 6>(0, 0) += jacobians[0].transpose() * jacobians[0];
            b.segment<6>(0) += jacobians[0].transpose() * initial_residual;
        }

        // Accumulate landmark contribution
        if (jacobians[1].size() > 0)
        {
            H.block<3, 3>(6, 6) += jacobians[1].transpose() * jacobians[1];
            b.segment<3>(6) += jacobians[1].transpose() * initial_residual;
        }

        // Cross terms
        if (jacobians[0].size() > 0 && jacobians[1].size() > 0)
        {
            H.block<6, 3>(0, 6) += jacobians[0].transpose() * jacobians[1];
            H.block<3, 6>(6, 0) += jacobians[1].transpose() * jacobians[0];
        }

        // Solve for increment
        Eigen::VectorXd dx = H.ldlt().solve(-b);

        // Apply increments
        if (jacobians[0].size() > 0)
        {
            pose->apply_increment(dx.segment<6>(0));
        }
        if (jacobians[1].size() > 0)
        {
            landmark->apply_increment(dx.segment<3>(6));
        }

        // Check that cost decreased
        Eigen::VectorXd final_residual = factor->compute_residual();
        double final_cost = 0.5 * final_residual.squaredNorm();

        REQUIRE(final_cost < initial_cost);
        REQUIRE(final_residual.norm() < initial_residual.norm());
    }
}

// ============================================================================
// LinearVelocityFactor Tests
// ============================================================================

TEST_CASE("LinearVelocityFactor: residual computation", "[linear_velocity]")
{
    // var1 at origin, var2 at (2,4,6), dt=2, velocity should be (1,2,3)
    auto var1 = std::make_shared<GenericVariable>(0, Eigen::Vector3d(0, 0, 0));
    auto var2 = std::make_shared<GenericVariable>(1, Eigen::Vector3d(2, 4, 6));
    auto vel_var = std::make_shared<GenericVariable>(2, Eigen::Vector3d(0.5, 1.0, 1.5)); // wrong velocity

    double dt = 2.0;
    LinearVelocityFactor factor(0, var1.get(), var2.get(), vel_var.get(), dt, 1.0);

    Eigen::VectorXd residual = factor.compute_residual();
    // residual = (var2 - var1) - vel*dt = (2,4,6) - (1,2,3) = (1,2,3)
    REQUIRE(is_approx_equal(residual, Eigen::Vector3d(1, 2, 3), precision_tol));
}

TEST_CASE("LinearVelocityFactor: sigma/weight behavior", "[linear_velocity]")
{
    auto var1 = std::make_shared<GenericVariable>(0, Eigen::Vector3d(0, 0, 0));
    auto var2 = std::make_shared<GenericVariable>(1, Eigen::Vector3d(1, 1, 1));
    auto vel_var = std::make_shared<GenericVariable>(2, Eigen::Vector3d(0, 0, 0));

    double sigma = 0.5; // weight = 2.0
    LinearVelocityFactor factor(0, var1.get(), var2.get(), vel_var.get(), 1.0, sigma);

    Eigen::VectorXd residual = factor.compute_residual();
    REQUIRE(is_approx_equal(residual, 2.0 * Eigen::Vector3d(1, 1, 1), precision_tol));
}

TEST_CASE("LinearVelocityFactor: Jacobian structure", "[linear_velocity]")
{
    auto var1 = std::make_shared<GenericVariable>(0, Eigen::Vector3d(1, 2, 3));
    auto var2 = std::make_shared<GenericVariable>(1, Eigen::Vector3d(2, 3, 4));
    auto vel_var = std::make_shared<GenericVariable>(2, Eigen::Vector3d(1, 1, 1));

    double dt = 2.0;
    LinearVelocityFactor factor(0, var1.get(), var2.get(), vel_var.get(), dt, 1.0);

    std::vector<Eigen::MatrixXd> jacobians;
    factor.compute_jacobians(jacobians);

    REQUIRE(jacobians.size() == 3);
    REQUIRE(is_approx_equal(jacobians[0], -Eigen::Matrix3d::Identity(), precision_tol));
    REQUIRE(is_approx_equal(jacobians[1], Eigen::Matrix3d::Identity(), precision_tol));
    REQUIRE(is_approx_equal(jacobians[2], -dt * Eigen::Matrix3d::Identity(), precision_tol));
}

TEST_CASE("LinearVelocityFactor: analytical vs numerical Jacobians", "[linear_velocity]")
{
    auto var1 = std::make_shared<GenericVariable>(0, Eigen::Vector3d(1, 2, 3));
    auto var2 = std::make_shared<GenericVariable>(1, Eigen::Vector3d(2.5, 3.5, 4.5));
    auto vel_var = std::make_shared<GenericVariable>(2, Eigen::Vector3d(1.2, 1.8, 2.1));

    auto factor = std::make_shared<LinearVelocityFactor>(0, var1.get(), var2.get(), vel_var.get(), 2.5, 0.3);

    std::vector<Eigen::MatrixXd> J_analytic, J_numeric;
    factor->compute_jacobians(J_analytic);
    ComputeNumericalJacobians(*factor, J_numeric);

    for (size_t i = 0; i < J_analytic.size(); ++i)
    {
        REQUIRE(is_approx_equal(J_analytic[i], J_numeric[i], 1e-6));
    }
}

TEST_CASE("LinearVelocityFactor: initial_index constrains subset of state", "[linear_velocity]")
{
    // 6DOF variables (e.g., pose = position + rotation)
    Eigen::Matrix<double, 6, 1> var1_val, var2_val;
    var1_val << 0, 0, 0, 0.1, 0.2, 0.3;
    var2_val << 2, 4, 6, 0.1, 0.2, 0.3;

    auto var1 = std::make_shared<GenericVariable>(0, var1_val);
    auto var2 = std::make_shared<GenericVariable>(1, var2_val);
    auto vel_var = std::make_shared<GenericVariable>(2, Eigen::Vector3d(1, 2, 3));

    double dt = 2.0;
    int initial_index = 0;

    auto factor = std::make_shared<LinearVelocityFactor>(0, var1.get(), var2.get(), vel_var.get(), dt, 1.0, initial_index);

    Eigen::VectorXd residual = factor->compute_residual();
    REQUIRE(residual.size() == 3);
    REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero(), precision_tol));

    std::vector<Eigen::MatrixXd> J_analytic, J_numeric;
    factor->compute_jacobians(J_analytic);
    ComputeNumericalJacobians(*factor, J_numeric);

    for (size_t i = 0; i < J_analytic.size(); ++i)
    {
        REQUIRE(is_approx_equal(J_analytic[i], J_numeric[i], 1e-6));
    }
}

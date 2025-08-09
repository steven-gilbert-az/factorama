#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "factorama_test/test_utils.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/generic_variable.hpp"
#include "factorama/bearing_observation_factor.hpp"
#include "factorama/rotation_variable.hpp"
#include "factorama/factor_graph.hpp"
#include "factorama/generic_between_factor.hpp"
#include "factorama/inverse_range_variable.hpp"
#include "factorama/inverse_range_bearing_factor.hpp"
#include "factorama/generic_prior_factor.hpp"
#include "factorama/pose_between_factors.hpp"
#include "factorama/pose_prior_factors.hpp"
#include "factorama/random_utils.hpp"
#include "factorama/numerical_jacobian.hpp"

using namespace factorama;

constexpr double kTol = 1e-9;

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
    auto factor = std::make_shared<BearingObservationFactor>(0, pose, landmark, expected_bearing_C, 1.0);
    FactorGraph graph;
    graph.add_variable(pose);
    graph.add_variable(landmark);
    graph.add_factor(factor);
    graph.finalize_structure();

    SECTION("Residual should be near zero for perfect measurement")
    {
        Eigen::VectorXd residual = graph.compute_full_residual_vector();

        REQUIRE(residual.size() == 3); // Because bearing residual is a 3D reprojection error
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero()));
    }

    SECTION("Residual should increase after landmark perturbation")
    {
        // Perturb landmark slightly
        landmark->set_pos_W(Eigen::Vector3d(0.1, 0.0, 5.0));

        Eigen::VectorXd residual = graph.compute_full_residual_vector();

        REQUIRE(residual.size() == 3);

        double norm = residual.norm();
        INFO("Residual norm after perturbation: " << norm);

        // Should be small but non-zero
        REQUIRE(norm > 0.0);
        REQUIRE(norm < 0.05); // sanity bound, since 0.1m shift at 5m shouldn't cause huge error
    }
}

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
    InverseRangeBearingFactor factor(0, cam_pose, inv_range_var, bearing_C_gt);

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

    PosePositionPriorFactor factor(0, pose, pos_prior, sigma);

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

    GenericPriorFactor factor(0, landmark, prior, sigma);

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

        auto factor = std::make_shared<BearingObservationFactor>(0, pose, landmark, bearing_C, sigma);

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

        auto pose = std::make_shared<PoseVariable>(0, cam_pose, true); // do_so3_nudge=true
        auto landmark = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(2.8, 1.2, 3.5));
        
        Eigen::Vector3d bearing_C = Eigen::Vector3d(0.6, -0.3, 0.8).normalized();
        double sigma = 0.05; // Non-unit sigma
        
        auto factor = std::make_shared<BearingObservationFactor>(0, pose, landmark, bearing_C, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());
        
        for (size_t i = 0; i < J_analytic.size(); ++i) {
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

        auto pose = std::make_shared<PoseVariable>(0, cam_pose, true); // do_so3_nudge=true
        auto landmark = std::make_shared<LandmarkVariable>(1, Eigen::Vector3d(-1.2, 0.8, 4.2));
        
        Eigen::Vector3d bearing_C = Eigen::Vector3d(-0.4, 0.7, 0.6).normalized();
        double sigma = 0.2;
        
        auto factor = std::make_shared<BearingObservationFactor>(0, pose, landmark, bearing_C, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());
        
        for (size_t i = 0; i < J_analytic.size(); ++i) {
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
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose, inv_range_var, bearing_C, sigma);

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

        auto pose = std::make_shared<PoseVariable>(0, cam_pose, true); // do_so3_nudge=true
        
        Eigen::Vector3d bearing_W = Eigen::Vector3d(0.6, 0.8, 0.1).normalized();
        double range = 3.5;
        
        auto inv_range_var = std::make_shared<InverseRangeVariable>(1, cam_pos_W, bearing_W, range);
        
        Eigen::Vector3d lm_pos = cam_pos_W + range * bearing_W;
        Eigen::Matrix3d dcm_CW = pose->dcm_CW();
        Eigen::Vector3d bearing_C = dcm_CW * (lm_pos - cam_pos_W).normalized();
        
        double sigma = 0.08;
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose, inv_range_var, bearing_C, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());
        
        for (size_t i = 0; i < J_analytic.size(); ++i) {
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

        auto pose = std::make_shared<PoseVariable>(0, cam_pose, true); // do_so3_nudge=true
        
        Eigen::Vector3d bearing_W = Eigen::Vector3d(-0.4, 0.5, 0.7).normalized();
        double range = 2.8;
        
        auto inv_range_var = std::make_shared<InverseRangeVariable>(1, cam_pos_W, bearing_W, range);
        
        Eigen::Vector3d lm_pos = cam_pos_W + range * bearing_W;
        Eigen::Matrix3d dcm_CW = pose->dcm_CW();
        Eigen::Vector3d bearing_C = dcm_CW * (lm_pos - cam_pos_W).normalized();
        
        double sigma = 0.15;
        auto factor = std::make_shared<InverseRangeBearingFactor>(0, pose, inv_range_var, bearing_C, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());
        
        for (size_t i = 0; i < J_analytic.size(); ++i) {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-6);
        }
    }
}

TEST_CASE("SO(3) exponential and logarithmic maps round trip", "[SO3]")
{
    const std::vector<Eigen::Vector3d> test_omegas = {
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.01, 0, 0),
        Eigen::Vector3d(0, 0.01, 0),
        Eigen::Vector3d(0, 0, 0.01),
        Eigen::Vector3d(0.1, -0.2, 0.3),
        Eigen::Vector3d(M_PI / 2, 0, 0),
        Eigen::Vector3d(0, M_PI / 2, 0),
        Eigen::Vector3d(0, 0, M_PI / 2),
    };

    for (const auto &omega : test_omegas)
    {
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

    for (const auto &axis : axis_vectors)
    {

        CAPTURE(axis);
        Eigen::Vector3d omega = M_PI * axis;
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

        if (recovered_angle > 1e-8)
        {
            Eigen::Vector3d recovered_axis = recovered / recovered_angle;
            REQUIRE(std::abs(recovered_axis.dot(axis)) > 0.999); // cosine near 1
        }
    }
}

TEST_CASE("SO(3) near-zero rotation is stable", "[SO3][EdgeCase]")
{

    std::vector<double> test_cases{
        0.1,
        0.01,
        0.00345,
        1e-5,
        2e-7,
        1e-10};

    for (double test_theta : test_cases)
    {
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
    Eigen::AngleAxisd aa(M_PI, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d R = aa.toRotationMatrix();

    Eigen::Vector3d log_result = LogMapSO3(R);
    double angle = log_result.norm();

    // Should be π, axis may be +X or -X
    bool positive_pi = std::abs(angle - M_PI) < 1e-6;
    bool negative_pi = std::abs(angle + M_PI) < 1e-6;
    bool ok = positive_pi || negative_pi;
    REQUIRE(ok);
    // REQUIRE(std::abs(angle - M_PI) < 1e-6 || std::abs(angle + M_PI) < 1e-6);

    Eigen::Vector3d axis = log_result.normalized();
    REQUIRE(std::abs(std::abs(axis.dot(Eigen::Vector3d::UnitX())) - 1.0) < 1e-6);
}

TEST_CASE("PoseOrientationBetweenFactor residual and jacobian numeric check", "[PoseOrientationBetweenFactor][numeric]")
{
    SECTION("Basic configuration (existing test)")
    {
        // Setup two camera poses with identity and 90deg rotation about Z
        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R2;
        R2 = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());

        auto pose1 = std::make_shared<PoseVariable>(1, Eigen::Vector3d::Zero(), R1);
        auto pose2 = std::make_shared<PoseVariable>(2, Eigen::Vector3d::Zero(), R2);

        // Initial extrinsic rotation guess - identity
        auto extrinsic = std::make_shared<RotationVariable>(3, Eigen::Matrix3d::Identity());

        PoseOrientationBetweenFactor factor(0, pose1, pose2, extrinsic);

        // Compute residual
        Eigen::VectorXd residual = factor.compute_residual();

        INFO("Residual vector: " << residual.transpose());

        // Residual norm should be non-negative and finite
        REQUIRE(std::isfinite(residual.norm()));
        REQUIRE(residual.size() == 3);

        // Numeric Jacobian check for each variable
        std::vector<std::shared_ptr<Variable>> vars = factor.variables();
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
        auto pose1 = std::make_shared<PoseVariable>(0, pose1_vec, true); // do_so3_nudge=true
        
        Eigen::Vector3d pos2_W(1.2, 0.8, 0.3);
        Eigen::Vector3d rot2_W(0.05, 0.25, 0.18);
        Eigen::Matrix<double, 6, 1> pose2_vec;
        pose2_vec << pos2_W, rot2_W;
        auto pose2 = std::make_shared<PoseVariable>(1, pose2_vec, true); // do_so3_nudge=true
        
        // Extrinsic rotation between sensor frames
        Eigen::Vector3d extrinsic_rot_vec(0.02, 0.03, -0.01);
        Eigen::Matrix3d extrinsic_rot = ExpMapSO3(extrinsic_rot_vec);
        auto extrinsic_var = std::make_shared<RotationVariable>(2, extrinsic_rot, true);
        
        double sigma = 0.75;
        auto factor = std::make_shared<PoseOrientationBetweenFactor>(0, pose1, pose2, extrinsic_var, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());

        
        // This should pass since PoseOrientationBetweenFactor uses numerical jacobians
        for (size_t i = 0; i < J_analytic.size(); ++i) { 

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
        auto pose1 = std::make_shared<PoseVariable>(0, pose1_vec, true); // do_so3_nudge=true
        
        Eigen::Vector3d pos2_W(-0.2, 1.5, -0.1);
        Eigen::Vector3d rot2_W(-0.1, 0.08, -0.02);
        Eigen::Matrix<double, 6, 1> pose2_vec;
        pose2_vec << pos2_W, rot2_W;
        auto pose2 = std::make_shared<PoseVariable>(1, pose2_vec, true); // do_so3_nudge=true
        
        // Different extrinsic rotation
        Eigen::Vector3d extrinsic_rot_vec(-0.015, 0.025, 0.008);
        Eigen::Matrix3d extrinsic_rot = ExpMapSO3(extrinsic_rot_vec);
        auto extrinsic_var = std::make_shared<RotationVariable>(2, extrinsic_rot, true);
        
        double sigma = 0.12;
        auto factor = std::make_shared<PoseOrientationBetweenFactor>(0, pose1, pose2, extrinsic_var, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_analytic;
        std::vector<Eigen::MatrixXd> J_numeric;
        
        factor->compute_jacobians(J_analytic);
        ComputeNumericalJacobians(*factor, J_numeric);
        
        REQUIRE(J_analytic.size() == J_numeric.size());

        
        // This should pass since PoseOrientationBetweenFactor uses numerical jacobians
        for (size_t i = 0; i < J_analytic.size(); ++i) {
            REQUIRE(J_analytic[i].rows() == J_numeric[i].rows());
            REQUIRE(J_analytic[i].cols() == J_numeric[i].cols());
            REQUIRE((J_analytic[i] - J_numeric[i]).norm() < 1e-4); // Looser tolerance for numerical vs numerical
        }
    }
}

TEST_CASE("PoseVariable apply_increment with and without do_so3_nudge")
{
    // Initial pose: zero translation, small rotation about X axis
    Eigen::Matrix<double, 6, 1> init_pose;
    init_pose << 0, 0, 0, 0.1, 0, 0;

    // Increment: translation + rotation about Y axis
    Eigen::Matrix<double, 6, 1> dx;
    dx << 1, 2, 3, 0, 0.2, 0;

    // PoseVariable with do_so3_nudge = false
    PoseVariable pose_linear(0, init_pose, false);
    pose_linear.apply_increment(dx);

    Eigen::Matrix<double, 6, 1> expected_linear = init_pose + dx;
    REQUIRE(is_approx_equal(pose_linear.value(), expected_linear));

    // PoseVariable with do_so3_nudge = true
    PoseVariable pose_so3(1, init_pose, true);
    pose_so3.apply_increment(dx);

    // Compute expected rotation update: R_new = exp(dR) * R_old
    Eigen::Matrix3d R_init = ExpMapSO3(init_pose.tail<3>());
    Eigen::Matrix3d R_updated = ExpMapSO3(dx.tail<3>()) * R_init;
    Eigen::Vector3d expected_rot = LogMapSO3(R_updated);

    Eigen::Matrix<double, 6, 1> expected_so3;
    expected_so3.head<3>() = init_pose.head<3>() + dx.head<3>();
    expected_so3.tail<3>() = expected_rot;

    REQUIRE(is_approx_equal(pose_so3.value().head<3>(), expected_so3.head<3>()));
    REQUIRE(is_approx_equal(pose_so3.value().tail<3>(), expected_so3.tail<3>()));
}

TEST_CASE("RotationVariable apply_increment with and without do_so3_nudge")
{
    // Initial rotation: 45 degrees about Z axis
    Eigen::Matrix3d R_init = Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Rotation increment: 30 degrees about Y axis
    Eigen::Vector3d dx(0, M_PI / 6, 0);

    // RotationVariable with do_so3_nudge = false
    RotationVariable rot_linear(0, R_init, false);
    rot_linear.apply_increment(dx);

    Eigen::Vector3d expected_vec_linear = LogMapSO3(R_init) + dx;
    Eigen::Matrix3d expected_R_linear = ExpMapSO3(expected_vec_linear);

    REQUIRE(is_approx_equal(rot_linear.value(), expected_vec_linear));
    REQUIRE(is_approx_equal(rot_linear.rotation(), expected_R_linear));

    // RotationVariable with do_so3_nudge = true
    RotationVariable rot_so3(1, R_init, true);
    rot_so3.apply_increment(dx);

    Eigen::Matrix3d expected_R_so3 = ExpMapSO3(dx) * R_init;
    Eigen::Vector3d expected_vec_so3 = LogMapSO3(expected_R_so3);

    REQUIRE(is_approx_equal(rot_so3.value(), expected_vec_so3));
    REQUIRE(is_approx_equal(rot_so3.rotation(), expected_R_so3));
}

TEST_CASE("GenericBetweenFactor residuals and jacobians with LandmarkVariable", "[GenericBetweenFactor]")
{
    using namespace factorama;

    constexpr double kTol = 1e-9;

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

        GenericBetweenFactor factor(42, a, b, expected);

        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero(), kTol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), kTol));
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), kTol));
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

        GenericBetweenFactor factor(43, a, b, expected, sigma);

        Eigen::VectorXd residual = factor.compute_residual();
        Eigen::Vector3d expected_residual = b->value() - a->value();
        REQUIRE(is_approx_equal(residual, expected_residual, kTol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), kTol));
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), kTol));
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
        GenericBetweenFactor factor(44, a, b, expected, sigma);

        Eigen::VectorXd residual = factor.compute_residual();
        Eigen::Vector3d expected_residual = 2.0 * (b->value() - a->value());
        REQUIRE(is_approx_equal(residual, expected_residual, kTol));

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -2.0 * Eigen::Matrix3d::Identity(), kTol));
        REQUIRE(is_approx_equal(J[1], 2.0 * Eigen::Matrix3d::Identity(), kTol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Skip Jacobian for constant variable A")
    {
        auto a = make_landmark(6, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto b = make_landmark(7, Eigen::Vector3d(2.0, 3.0, 4.0));
        a->set_is_constant(true);
        auto expected = std::make_shared<GenericVariable>(45, Eigen::Vector3d::Zero());
        expected->set_is_constant(true);

        GenericBetweenFactor factor(45, a, b, expected);

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(J[0].size() == 0); // Should be skipped
        REQUIRE(is_approx_equal(J[1], Eigen::Matrix3d::Identity(), kTol));
        REQUIRE(J[2].size() == 0); // expected is constant
    }

    SECTION("Skip Jacobian for constant variable B")
    {
        auto a = make_landmark(8, Eigen::Vector3d(1.0, 2.0, 3.0));
        auto b = make_landmark(9, Eigen::Vector3d(4.0, 5.0, 6.0));
        b->set_is_constant(true);
        auto expected = std::make_shared<GenericVariable>(46, Eigen::Vector3d(1.0, 2.0, 3.0));
        expected->set_is_constant(true);

        GenericBetweenFactor factor(46, a, b, expected);

        std::vector<Eigen::MatrixXd> J;
        factor.compute_jacobians(J);
        REQUIRE(J.size() == 3);
        REQUIRE(is_approx_equal(J[0], -Eigen::Matrix3d::Identity(), kTol));
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
        PosePositionBetweenFactor factor(100, pose_a, pose_b, measured, 1.0);
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

        PosePositionBetweenFactor factor(101, pose_a, pose_b, measured2, 1.0);
        Eigen::VectorXd residual = factor.compute_residual();

        Eigen::Vector3d expected_residual = Eigen::Vector3d(2, 1, 4) - measured_vec2; // (1,-1,1)
        REQUIRE(is_approx_equal(residual, expected_residual));
    }

    // --------------------------
    SECTION("Jacobian structure and sign are correct")
    {
        PosePositionBetweenFactor factor(102, pose_a, pose_b, measured, 1.0);

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
        pose_a->set_constant(true);
        PosePositionBetweenFactor factor(103, pose_a, pose_b, measured, 1.0);

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
        PosePositionBetweenFactor factor(104, pose_a, pose_b, measured, sigma);

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
        
        PosePositionPriorFactor factor(0, pose, pos_prior, sigma);
        
        // Check residual
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);
        
        Eigen::Vector3d expected_residual = (1.0 / sigma) * (pose->pos_W() - pos_prior);
        REQUIRE(is_approx_equal(residual, expected_residual, kTol));
        
        // Check Jacobian
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].rows() == 3);
        REQUIRE(jacobians[0].cols() == 6);
        
        // Expected Jacobian: derivative of position residual w.r.t. pose
        Eigen::MatrixXd expected_J = Eigen::MatrixXd::Zero(3, 6);
        expected_J.block<3, 3>(0, 0) = (1.0 / sigma) * Eigen::Matrix3d::Identity();
        REQUIRE(is_approx_equal(jacobians[0], expected_J, kTol));
    }
    
    SECTION("Zero residual for exact position match")
    {
        Eigen::Vector3d pos_exact(3.0, -1.0, 2.5);
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << pos_exact, 0.0, 0.0, 0.0; // position matches exactly
        auto pose = std::make_shared<PoseVariable>(0, pose_init);
        
        PosePositionPriorFactor factor(0, pose, pos_exact, 1.0);
        
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(is_approx_equal(residual, Eigen::Vector3d::Zero(), kTol));
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
            PosePositionPriorFactor factor(0, pose, pos_prior, sigma);
            
            Eigen::VectorXd residual = factor.compute_residual();
            double expected_scale = 1.0 / sigma;
            Eigen::Vector3d expected_residual = expected_scale * (pose->pos_W() - pos_prior);
            
            REQUIRE(is_approx_equal(residual, expected_residual, kTol));
            
            // Check Jacobian scaling
            std::vector<Eigen::MatrixXd> jacobians;
            factor.compute_jacobians(jacobians);
            
            Eigen::MatrixXd expected_J = Eigen::MatrixXd::Zero(3, 6);
            expected_J.block<3, 3>(0, 0) = expected_scale * Eigen::Matrix3d::Identity();
            REQUIRE(is_approx_equal(jacobians[0], expected_J, kTol));
        }
    }
    
    SECTION("Constant pose variable handling")
    {
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;
        auto pose = std::make_shared<PoseVariable>(0, pose_init);
        pose->set_constant(true);
        
        Eigen::Vector3d pos_prior(0.0, 0.0, 0.0);
        PosePositionPriorFactor factor(0, pose, pos_prior, 1.0);
        
        // Residual should still be computed
        Eigen::VectorXd residual = factor.compute_residual();
        REQUIRE(residual.size() == 3);
        REQUIRE(is_approx_equal(residual, pose->pos_W() - pos_prior, kTol));
        
        // Jacobian should be empty for constant variables
        std::vector<Eigen::MatrixXd> jacobians;
        factor.compute_jacobians(jacobians);
        REQUIRE(jacobians.size() == 1);
        REQUIRE(jacobians[0].size() == 0);
    }
}


TEST_CASE("PoseOrientationPriorFactor: behavior with do_so3_nudge modes", "[prior][orientation][so3]")
{
    // Setup: pose with non-trivial initial rotation
    Eigen::Matrix<double, 6, 1> pose_init;
    pose_init.head<3>() = Eigen::Vector3d(1.0, 2.0, 3.0);  // position (irrelevant for this factor)
    pose_init.tail<3>() = Eigen::Vector3d(0.15, -0.2, 0.25); // initial rotation vector
    
    // Prior rotation (different from initial)
    Eigen::Vector3d rot_prior(0.1, -0.15, 0.2);
    double sigma = 0.1;
    
    // Small rotation increment for testing
    Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
    dx.tail<3>() = Eigen::Vector3d(0.03, 0.02, -0.015); // small rotation increment
    
    SECTION("Linear mode (do_so3_nudge = false)")
    {
        // Create pose variable with linear rotation updates
        auto pose_linear = std::make_shared<PoseVariable>(0, pose_init, false);
        PoseOrientationPriorFactor factor(0, pose_linear, rot_prior, sigma, false);
        
        // Compute initial residual
        Eigen::VectorXd r_initial = factor.compute_residual();
        REQUIRE(r_initial.size() == 3);
        
        // Expected initial residual (linear difference)
        Eigen::Vector3d expected_initial = (1.0 / sigma) * (pose_init.tail<3>() - rot_prior);
        REQUIRE(is_approx_equal(r_initial, expected_initial, kTol));
        
        // Apply increment using linear mode
        pose_linear->apply_increment(dx);
        
        // Compute residual after increment
        Eigen::VectorXd r_after = factor.compute_residual();
        
        // Expected residual after linear increment
        Eigen::Vector3d expected_after_linear = (1.0 / sigma) * ((pose_init.tail<3>() + dx.tail<3>()) - rot_prior);
        REQUIRE(is_approx_equal(r_after, expected_after_linear, kTol));
        
        // The change in residual should be linear in the increment
        Eigen::Vector3d residual_change = r_after - r_initial;
        Eigen::Vector3d expected_change = (1.0 / sigma) * dx.tail<3>();
        REQUIRE(is_approx_equal(residual_change, expected_change, kTol));
    }
    
    SECTION("Manifold mode (do_so3_nudge = true)")
    {
        // Create pose variable with manifold rotation updates
        auto pose_manifold = std::make_shared<PoseVariable>(1, pose_init, true);
        PoseOrientationPriorFactor factor(1, pose_manifold, rot_prior, sigma, true);
        
        // Compute initial residual (should be same as linear case initially)
        Eigen::VectorXd r_initial = factor.compute_residual();
        REQUIRE(r_initial.size() == 3);
        
        Eigen::Vector3d expected_initial = (1.0 / sigma) * (pose_init.tail<3>() - rot_prior);
        CAPTURE(r_initial);
        CAPTURE(expected_initial);
        bool is_close_ish = is_approx_equal(r_initial, expected_initial, 0.1);
        bool is_pretty_exact = is_approx_equal(r_initial, expected_initial, kTol);
        // Manifold should not match exactly the linear expectation.
        REQUIRE(is_close_ish);
        REQUIRE(!is_pretty_exact);
        
        // Apply increment using manifold mode
        pose_manifold->apply_increment(dx);
        
        // Compute residual after increment
        Eigen::VectorXd r_after = factor.compute_residual();
        
        // Expected rotation after manifold increment: R_new = exp(dx_rot) * R_init
        Eigen::Matrix3d R_init = ExpMapSO3(pose_init.tail<3>());
        Eigen::Matrix3d R_new = ExpMapSO3(dx.tail<3>()) * R_init;
        Eigen::Vector3d rot_vec_new = LogMapSO3(R_new);
        
        Eigen::Vector3d expected_after_manifold = (1.0 / sigma) * (rot_vec_new - rot_prior);

        CAPTURE(r_after);
        CAPTURE(expected_after_manifold);
        // 0.1 is close enough as the expected value was calculated with linear approximation
        REQUIRE(is_approx_equal(r_after, expected_after_manifold, 0.1));
        
        // The residual should be different from the linear case for non-small increments
        Eigen::Vector3d linear_result = (1.0 / sigma) * ((pose_init.tail<3>() + dx.tail<3>()) - rot_prior);
        
        // For small increments, the difference should be small but detectable
        double difference_norm = (r_after - linear_result).norm();
        
        // The manifold and linear results should differ (unless increment is tiny)
        if (dx.tail<3>().norm() > 1e-6)
        {
            REQUIRE(difference_norm > 1e-10); // Should see some difference
        }
    }
    
    SECTION("Jacobian consistency between modes")
    {
        // Test that Jacobians are computed correctly for both modes
        auto pose_linear = std::make_shared<PoseVariable>(2, pose_init, false);
        auto pose_manifold = std::make_shared<PoseVariable>(3, pose_init, true);
        
        PoseOrientationPriorFactor factor_linear(2, pose_linear, rot_prior, sigma, false);
        PoseOrientationPriorFactor factor_manifold(3, pose_manifold, rot_prior, sigma, true);
        
        std::vector<Eigen::MatrixXd> J_linear, J_manifold;
        factor_linear.compute_jacobians(J_linear);
        factor_manifold.compute_jacobians(J_manifold);
        
        REQUIRE(J_linear.size() == 1);
        REQUIRE(J_manifold.size() == 1);
        REQUIRE(J_linear[0].rows() == 3);
        REQUIRE(J_linear[0].cols() == 6);
        REQUIRE(J_manifold[0].rows() == 3);
        REQUIRE(J_manifold[0].cols() == 6);
        
        // The Jacobians will differ between modes - manifold mode uses inverse right jacobian
        // Print debug info if they differ significantly
        Eigen::MatrixXd diff_modes = J_linear[0] - J_manifold[0];
        double diff_modes_norm = diff_modes.norm();
        
        if (diff_modes_norm > kTol) {
            std::cout << "\nPoseOrientationPriorFactor Jacobian difference between modes (norm: " << diff_modes_norm << "):\n" << diff_modes << std::endl;
            std::cout << "Linear mode Jacobian:\n" << J_linear[0] << std::endl;
            std::cout << "Manifold mode Jacobian:\n" << J_manifold[0] << std::endl;
            std::cout << "pose rotation: " << pose_init.tail<3>().transpose() << std::endl;
            std::cout << "rot_prior: " << rot_prior.transpose() << std::endl;
        }
        
        // Expected Jacobian for linear mode: derivative of (rot_CW - rot_prior) w.r.t. pose
        Eigen::MatrixXd expected_J_linear = Eigen::MatrixXd::Zero(3, 6);
        expected_J_linear.block<3, 3>(0, 3) = (1.0 / sigma) * Eigen::Matrix3d::Identity();
        
        // Check linear mode matches expected
        Eigen::MatrixXd diff_linear_expected = J_linear[0] - expected_J_linear;
        if (diff_linear_expected.norm() > kTol) {
            std::cout << "\nLinear mode Jacobian vs expected (norm: " << diff_linear_expected.norm() << "):\n" << diff_linear_expected << std::endl;
            std::cout << "Linear Jacobian:\n" << J_linear[0] << std::endl;
            std::cout << "Expected Linear:\n" << expected_J_linear << std::endl;
        }
        REQUIRE(is_approx_equal(J_linear[0], expected_J_linear, kTol));
    }
}

TEST_CASE("PoseOrientationPriorFactor: convergence behavior with different do_so3_nudge modes", "[prior][orientation][so3][convergence]")
{
    // Setup: pose with larger initial rotation error
    Eigen::Matrix<double, 6, 1> pose_init;
    pose_init.head<3>() = Eigen::Vector3d::Zero();
    pose_init.tail<3>() = Eigen::Vector3d(0.6, -0.4, 0.5); // larger initial rotation
    
    Eigen::Vector3d rot_prior = Eigen::Vector3d(0.023, -0.005, -0.01); // random ish prior (but close to zero)
    double sigma = 0.1;
    
    // Larger increment to see manifold effects more clearly
    Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
    dx.tail<3>() = Eigen::Vector3d(-0.12, -0.08, -0.1); // move towards prior
    
    SECTION("Residual magnitude comparison")
    {
        auto pose_linear = std::make_shared<PoseVariable>(0, pose_init, false);
        auto pose_manifold = std::make_shared<PoseVariable>(1, pose_init, true);
        
        PoseOrientationPriorFactor factor_linear(0, pose_linear, rot_prior, sigma, false);
        PoseOrientationPriorFactor factor_manifold(1, pose_manifold, rot_prior, sigma, true);
        
        // Initial residuals should be identical
        Eigen::VectorXd r_initial_linear = factor_linear.compute_residual();
        Eigen::VectorXd r_initial_manifold = factor_manifold.compute_residual();
        //REQUIRE(is_approx_equal(r_initial_linear, r_initial_manifold, kTol));
        
        double initial_cost = 0.5 * r_initial_linear.squaredNorm();
        
        // Apply the same increment to both
        pose_linear->apply_increment(dx);
        pose_manifold->apply_increment(dx);
        
        // Check final residuals
        Eigen::VectorXd r_final_linear = factor_linear.compute_residual();
        Eigen::VectorXd r_final_manifold = factor_manifold.compute_residual();
        
        double final_cost_linear = 0.5 * r_final_linear.squaredNorm();
        double final_cost_manifold = 0.5 * r_final_manifold.squaredNorm();
        
        // Both should reduce cost, but manifold should typically be more accurate
        REQUIRE(final_cost_linear < initial_cost);
        REQUIRE(final_cost_manifold < initial_cost);
        
        // For this specific case (moving toward zero rotation), manifold should be better
        // The exact relationship depends on the specific values, but we can at least
        // verify that both methods work and produce different results
        REQUIRE(std::abs(final_cost_linear - final_cost_manifold) > 1e-12);
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
    
    auto factor = std::make_shared<PosePositionPriorFactor>(0, pose, pos_prior, sigma);
    
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
    double sigma = 0.3;
    
    auto factor = std::make_shared<PoseOrientationPriorFactor>(0, pose, rot_prior, sigma);
    
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
    
    if (diff_norm >= 1e-6) {
        std::cout << "\nPoseOrientationPriorFactor Jacobian diff (norm: " << diff_norm << "):\n" << diff << std::endl;
        std::cout << "Analytical Jacobian:\n" << J_analytic[0] << std::endl;
        std::cout << "Numerical Jacobian:\n" << J_numeric[0] << std::endl;
        std::cout << "do_so3_nudge: " << pose->do_so3_nudge() << std::endl;
        std::cout << "pose rotation: " << pose->rot_CW().transpose() << std::endl;
        std::cout << "rot_prior: " << rot_prior.transpose() << std::endl;
    }
    
    REQUIRE(diff_norm < 1e-6);
}

TEST_CASE("Variable clone() method behaves correctly", "[variable][clone]")
{
    SECTION("PoseVariable clone test")
    {
        // Create original pose variable
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3;
        auto original = std::make_shared<PoseVariable>(42, pose_init, true);
        original->set_constant(true);
        
        // Clone the variable
        auto cloned_base = original->clone();
        auto cloned = std::dynamic_pointer_cast<PoseVariable>(cloned_base);
        
        REQUIRE(cloned != nullptr);
        REQUIRE(cloned.get() != original.get()); // Different objects
        
        // Verify all properties are copied correctly
        REQUIRE(cloned->id() == original->id());
        REQUIRE(cloned->size() == original->size());
        REQUIRE(is_approx_equal(cloned->value(), original->value()));
        REQUIRE(is_approx_equal(cloned->pos_W(), original->pos_W()));
        REQUIRE(is_approx_equal(cloned->dcm_CW(), original->dcm_CW()));
        REQUIRE(cloned->is_constant() == original->is_constant());
        REQUIRE(cloned->do_so3_nudge() == original->do_so3_nudge());
        
        // Verify independent modification
        Eigen::Matrix<double, 6, 1> increment;
        increment << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01;
        
        cloned->set_constant(false);
        cloned->apply_increment(increment);
        
        // Original should be unchanged
        REQUIRE(is_approx_equal(original->value(), pose_init));
        REQUIRE(original->is_constant() == true);
        
        // Clone should be modified
        REQUIRE(!is_approx_equal(cloned->value(), original->value()));
        REQUIRE(cloned->is_constant() == false);
    }
    
    SECTION("LandmarkVariable clone test")
    {
        // Create original landmark variable
        Eigen::Vector3d pos_init(5.0, -2.0, 1.5);
        auto original = std::make_shared<LandmarkVariable>(123, pos_init);
        original->set_is_constant(true);
        
        // Clone the variable
        auto cloned_base = original->clone();
        auto cloned = std::dynamic_pointer_cast<LandmarkVariable>(cloned_base);
        
        REQUIRE(cloned != nullptr);
        REQUIRE(cloned.get() != original.get()); // Different objects
        
        // Verify all properties are copied correctly
        REQUIRE(cloned->id() == original->id());
        REQUIRE(cloned->size() == original->size());
        REQUIRE(is_approx_equal(cloned->value(), original->value()));
        REQUIRE(is_approx_equal(cloned->pos_W(), original->pos_W()));
        REQUIRE(cloned->is_constant() == original->is_constant());
        
        // Verify independent modification
        Eigen::Vector3d increment(0.2, -0.3, 0.1);
        
        cloned->set_is_constant(false);
        cloned->apply_increment(increment);
        
        // Original should be unchanged
        REQUIRE(is_approx_equal(original->pos_W(), pos_init));
        REQUIRE(original->is_constant() == true);
        
        // Clone should be modified
        REQUIRE(!is_approx_equal(cloned->pos_W(), original->pos_W()));
        REQUIRE(cloned->is_constant() == false);
    }
    
    SECTION("InverseRangeVariable clone test")
    {
        // Create original inverse range variable
        Eigen::Vector3d origin_W(1.0, 0.0, -1.0);
        Eigen::Vector3d bearing_W(1.0, 1.0, 0.0); // Will be normalized
        double initial_range = 5.0;
        auto original = std::make_shared<InverseRangeVariable>(456, origin_W, bearing_W, initial_range);
        original->set_is_constant(true);
        
        // Clone the variable
        auto cloned_base = original->clone();
        auto cloned = std::dynamic_pointer_cast<InverseRangeVariable>(cloned_base);
        
        REQUIRE(cloned != nullptr);
        REQUIRE(cloned.get() != original.get()); // Different objects
        
        // Verify all properties are copied correctly
        REQUIRE(cloned->id() == original->id());
        REQUIRE(cloned->size() == original->size());
        REQUIRE(is_approx_equal(cloned->value(), original->value()));
        REQUIRE(is_approx_equal(cloned->origin_pos_W(), original->origin_pos_W()));
        REQUIRE(is_approx_equal(cloned->bearing_W(), original->bearing_W()));
        REQUIRE(is_approx_equal(cloned->pos_W(), original->pos_W()));
        REQUIRE(std::abs(cloned->inverse_range() - original->inverse_range()) < kTol);
        REQUIRE(cloned->is_constant() == original->is_constant());
        
        // Verify independent modification
        Eigen::VectorXd increment(1);
        increment[0] = -0.05; // Small change in inverse range
        
        cloned->set_is_constant(false);
        cloned->apply_increment(increment);
        
        // Original should be unchanged
        REQUIRE(std::abs(original->inverse_range() - 1.0/initial_range) < kTol);
        REQUIRE(original->is_constant() == true);
        
        // Clone should be modified
        REQUIRE(std::abs(cloned->inverse_range() - original->inverse_range()) > 1e-8);
        REQUIRE(cloned->is_constant() == false);
    }
    
    SECTION("RotationVariable clone test")
    {
        // Create original extrinsic rotation variable
        Eigen::Matrix3d dcm_CE = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        auto original = std::make_shared<RotationVariable>(789, dcm_CE, true);
        original->set_is_constant(true);
        
        // Clone the variable
        auto cloned_base = original->clone();
        auto cloned = std::dynamic_pointer_cast<RotationVariable>(cloned_base);
        
        REQUIRE(cloned != nullptr);
        REQUIRE(cloned.get() != original.get()); // Different objects
        
        // Verify all properties are copied correctly
        REQUIRE(cloned->id() == original->id());
        REQUIRE(cloned->size() == original->size());
        REQUIRE(is_approx_equal(cloned->value(), original->value()));
        REQUIRE(is_approx_equal(cloned->rotation(), original->rotation()));
        REQUIRE(cloned->is_constant() == original->is_constant());
        
        // Verify independent modification
        Eigen::Vector3d increment(0.05, -0.03, 0.02);
        
        cloned->set_is_constant(false);
        cloned->apply_increment(increment);
        
        // Original should be unchanged
        REQUIRE(is_approx_equal(original->rotation(), dcm_CE));
        REQUIRE(original->is_constant() == true);
        
        // Clone should be modified
        REQUIRE(!is_approx_equal(cloned->rotation(), original->rotation()));
        REQUIRE(cloned->is_constant() == false);
    }
    
    SECTION("Clone preserves polymorphic behavior")
    {
        // Test that cloned variables work correctly in polymorphic contexts
        std::vector<std::shared_ptr<Variable>> variables;
        
        // Create original variables
        auto pose = std::make_shared<PoseVariable>(1, Eigen::Matrix<double, 6, 1>::Zero());
        auto landmark = std::make_shared<LandmarkVariable>(2, Eigen::Vector3d(1, 0, 0));
        auto inv_range = std::make_shared<InverseRangeVariable>(3, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(), 2.0);
        auto extrinsic = std::make_shared<RotationVariable>(4, Eigen::Matrix3d::Identity());
        
        variables.push_back(pose);
        variables.push_back(landmark);
        variables.push_back(inv_range);
        variables.push_back(extrinsic);
        
        // Clone all variables
        std::vector<std::shared_ptr<Variable>> cloned_variables;
        for (const auto& var : variables) {
            cloned_variables.push_back(var->clone());
        }
        
        // Verify polymorphic behavior works
        REQUIRE(cloned_variables.size() == variables.size());
        
        for (size_t i = 0; i < variables.size(); ++i) {
            auto& original = variables[i];
            auto& cloned = cloned_variables[i];
            
            REQUIRE(cloned->id() == original->id());
            REQUIRE(cloned->type() == original->type());
            REQUIRE(cloned->size() == original->size());
            REQUIRE(is_approx_equal(cloned->value(), original->value()));
            REQUIRE(cloned->name() == original->name());
            REQUIRE(cloned.get() != original.get()); // Different objects
        }
    }
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
        graph.finalize_structure();
        
        // Create increment vector (landmark1: 3D, landmark2: 3D = 6D total)
        Eigen::VectorXd dx(6);
        dx << 0.1, 0.2, 0.3,    // landmark1 increment
              1.0, -0.5, 0.8;   // landmark2 increment
              
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
        var1->set_is_constant(true);
        
        FactorGraph graph;
        graph.add_variable(var1);
        graph.add_variable(var2);
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
        graph.finalize_structure();
        
        // Get initial variable vector
        Eigen::VectorXd x0 = graph.get_variable_vector();
        
        // Apply increment
        Eigen::VectorXd dx(6);
        dx << 0.1, 0.2, 0.3,    // landmark1
              0.4, 0.5, 0.6;    // landmark2
        
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
        auto factor1 = std::make_shared<BearingObservationFactor>(10, pose, landmark, bearing, 1.0);
        REQUIRE_NOTHROW(graph.add_factor(factor1));
        
        // Try to add second factor with same ID = 10
        auto factor2 = std::make_shared<BearingObservationFactor>(10, pose, landmark, bearing, 1.0);
        REQUIRE_THROWS_AS(graph.add_factor(factor2), std::runtime_error);
        
        // Different ID should work fine
        auto factor3 = std::make_shared<BearingObservationFactor>(11, pose, landmark, bearing, 1.0);
        REQUIRE_NOTHROW(graph.add_factor(factor3));
    }
}

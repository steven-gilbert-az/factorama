#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

#include "factorama_test/test_utils.hpp"

#include "factorama/generic_variable.hpp"
#include "factorama/inverse_range_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/plane_variable.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/rotation_variable.hpp"


using namespace factorama;


TEST_CASE("PoseVariable apply_increment")
{
    // Initial pose: zero translation, small rotation about X axis
    Eigen::Matrix<double, 6, 1> init_pose;
    init_pose << 0, 0, 0, 0.1, 0, 0;

    // Increment: translation + rotation about Y axis
    Eigen::Matrix<double, 6, 1> dx;
    dx << 1, 2, 3, 0, 0.2, 0;

    // PoseVariable using SO(3) manifold
    PoseVariable pose_so3(1, init_pose);
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


TEST_CASE("RotationVariable apply_increment")
{
    // Initial rotation: 45 degrees about Z axis
    Eigen::Matrix3d R_init = Eigen::AngleAxisd(PI / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Rotation increment: 30 degrees about Y axis
    Eigen::Vector3d dx(0, PI / 6, 0);

    // RotationVariable using SO(3) manifold
    RotationVariable rot_so3(1, R_init);
    rot_so3.apply_increment(dx);

    Eigen::Matrix3d expected_R_so3 = ExpMapSO3(dx) * R_init;
    Eigen::Vector3d expected_vec_so3 = LogMapSO3(expected_R_so3);

    REQUIRE(is_approx_equal(rot_so3.value(), expected_vec_so3));
    REQUIRE(is_approx_equal(rot_so3.rotation(), expected_R_so3));
}


TEST_CASE("Variable clone() method behaves correctly", "[variable][clone]")
{
    SECTION("PoseVariable clone test")
    {
        // Create original pose variable
        Eigen::Matrix<double, 6, 1> pose_init;
        pose_init << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3;
        auto original = std::make_shared<PoseVariable>(42, pose_init);
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
        original->set_constant(true);

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

        cloned->set_constant(false);
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
        original->set_constant(true);

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
        REQUIRE(std::abs(cloned->inverse_range() - original->inverse_range()) < precision_tol);
        REQUIRE(cloned->is_constant() == original->is_constant());

        // Verify independent modification
        Eigen::VectorXd increment(1);
        increment[0] = -0.05; // Small change in inverse range

        cloned->set_constant(false);
        cloned->apply_increment(increment);

        // Original should be unchanged
        REQUIRE(std::abs(original->inverse_range() - 1.0 / initial_range) < precision_tol);
        REQUIRE(original->is_constant() == true);

        // Clone should be modified
        REQUIRE(std::abs(cloned->inverse_range() - original->inverse_range()) > 1e-8);
        REQUIRE(cloned->is_constant() == false);
    }

    SECTION("RotationVariable clone test")
    {
        // Create original extrinsic rotation variable
        Eigen::Matrix3d dcm_CE = Eigen::AngleAxisd(PI / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        auto original = std::make_shared<RotationVariable>(789, dcm_CE);
        original->set_constant(true);

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

        cloned->set_constant(false);
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
        for (const auto &var : variables)
        {
            cloned_variables.push_back(var->clone());
        }

        // Verify polymorphic behavior works
        REQUIRE(cloned_variables.size() == variables.size());

        for (size_t i = 0; i < variables.size(); ++i)
        {
            auto &original = variables[i];
            auto &cloned = cloned_variables[i];

            REQUIRE(cloned->id() == original->id());
            REQUIRE(cloned->type() == original->type());
            REQUIRE(cloned->size() == original->size());
            REQUIRE(is_approx_equal(cloned->value(), original->value()));
            REQUIRE(cloned->name() == original->name());
            REQUIRE(cloned.get() != original.get()); // Different objects
        }
    }
}


TEST_CASE("PlaneVariable basic properties", "[plane][variable]")
{
    // Create plane with normal (0, 0, 1) at distance 5
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    PlaneVariable plane(1, normal, 5.0);

    REQUIRE(plane.id() == 1);
    REQUIRE(plane.size() == 4);
    REQUIRE(is_approx_equal(plane.unit_vector(), normal));
    REQUIRE(std::abs(plane.distance_from_origin() - 5.0) < precision_tol);

    // Test with non-normalized normal
    Eigen::Vector3d unnormalized(3.0, 4.0, 0.0);
    PlaneVariable plane2(2, unnormalized, -2.0);
    REQUIRE(std::abs(plane2.unit_vector().norm() - 1.0) < precision_tol);
}


TEST_CASE("PlaneVariable distance_from_point", "[plane][variable]")
{
    // XY plane at origin (normal = [0, 0, 1], distance = 0)
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    PlaneVariable plane(1, normal, 0.0);

    Eigen::Vector3d point_on_plane(1.0, 2.0, 0.0);
    REQUIRE(std::abs(plane.distance_from_point(point_on_plane)) < precision_tol);

    Eigen::Vector3d point_above(1.0, 2.0, 3.0);
    REQUIRE(std::abs(plane.distance_from_point(point_above) - 3.0) < precision_tol);

    Eigen::Vector3d point_below(1.0, 2.0, -2.5);
    REQUIRE(std::abs(plane.distance_from_point(point_below) + 2.5) < precision_tol);
}


TEST_CASE("PlaneVariable apply_increment", "[plane][variable]")
{
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    PlaneVariable plane(1, normal, 2.0);

    // Increment normal and distance
    Eigen::Vector4d dx;
    dx << 0.1, 0.05, 0.0, 1.5;
    plane.apply_increment(dx);

    // Normal should still be unit length
    REQUIRE(std::abs(plane.unit_vector().norm() - 1.0) < precision_tol);

    // Distance should increase by 1.5
    REQUIRE(std::abs(plane.distance_from_origin() - 3.5) < precision_tol);

    // Value should be consistent
    REQUIRE(is_approx_equal(plane.value().head<3>(), plane.unit_vector()));
    REQUIRE(std::abs(plane.value()(3) - plane.distance_from_origin()) < precision_tol);
}


TEST_CASE("PlaneVariable clone", "[plane][variable][clone]")
{
    Eigen::Vector3d normal(1.0, 1.0, 1.0);
    auto original = std::make_shared<PlaneVariable>(42, normal, -3.5);
    original->set_constant(true);

    auto cloned = std::dynamic_pointer_cast<PlaneVariable>(original->clone());

    REQUIRE(cloned != nullptr);
    REQUIRE(cloned.get() != original.get());
    REQUIRE(cloned->id() == original->id());
    REQUIRE(is_approx_equal(cloned->value(), original->value()));
    REQUIRE(cloned->is_constant() == original->is_constant());

    // Modify clone
    Eigen::Vector4d dx;
    dx << 0.2, 0.1, 0.0, 1.0;
    cloned->set_constant(false);
    cloned->apply_increment(dx);

    // Original should be unchanged
    REQUIRE(is_approx_equal(original->unit_vector(), normal.normalized()));
    REQUIRE(std::abs(original->distance_from_origin() + 3.5) < precision_tol);
    REQUIRE(original->is_constant() == true);
}

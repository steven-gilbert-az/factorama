#pragma once

#include "factorama/base_types.hpp"
#include "factorama/pose_2d_variable.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace factorama
{
    /**
     * @brief Relative constraint between two 2D poses
     *
     * Constrains the relative pose between two Pose2DVariables to match a measured
     * relative transformation. The measured transformation can be optionally solveable.
     * Handles angle wrapping gracefully - angles near -π and π are treated as equivalent.
     *
     * The residual is computed as:
     *   position_error = (pos_b - pos_a) - measured_between[0:2]
     *   angle_error = wrap((theta_b - theta_a) - measured_between[2])
     *
     * @code
     * Eigen::Vector3d measured_between(1.0, 0.5, PI/6);  // dx, dy, dtheta
     * auto between_var = std::make_shared<GenericVariable>(var_id++, measured_between);
     * double position_sigma = 0.1;
     * double angle_sigma = 0.05;
     * auto between_factor = std::make_shared<Pose2DBetweenFactor>(
     *     factor_id++, pose_a, pose_b, between_var.get(), position_sigma, angle_sigma);
     * @endcode
     */
    class Pose2DBetweenFactor final : public Factor
    {
    public:
        /**
         * @brief Construct 2D pose between factor
         * @param id Unique factor identifier
         * @param pose_a First 2D pose variable
         * @param pose_b Second 2D pose variable
         * @param measured_between_variable Measured relative pose [dx, dy, dtheta] (must be 3D)
         *        - If local_frame=true: position measured in pose_a's frame
         *        - If local_frame=false: position measured in world frame (pos_b - pos_a)
         * @param position_sigma Standard deviation of position measurement
         * @param angle_sigma Standard deviation of angular measurement (radians)
         * @param local_frame If true, position measurement is in pose_a's frame; if false, in world frame (default: false)
         */
        Pose2DBetweenFactor(int id,
                           Pose2DVariable* pose_a,
                           Pose2DVariable* pose_b,
                           Variable* measured_between_variable,
                           double position_sigma,
                           double angle_sigma,
                           bool local_frame = false);

        int residual_size() const override { return size_; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;

        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

        std::vector<Variable*> variables() override
        {
            return {pose_a_, pose_b_, measured_between_variable_};
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_2d_between;
        }

        std::string name() const override
        {
            return "Pose2DBetweenFactor(" + pose_a_->name() + ", " +
                   pose_b_->name() + ", " + measured_between_variable_->name() + ")";
        }

        double position_weight() const { return position_weight_; }
        double angle_weight() const { return angle_weight_; }

    private:
        Pose2DVariable* pose_a_;
        Pose2DVariable* pose_b_;
        Variable* measured_between_variable_;
        double position_weight_;  // 1.0 / position_sigma
        double angle_weight_;     // 1.0 / angle_sigma
        bool local_frame_;
        int size_ = 3;

        /**
         * @brief Wrap angle difference to [-π, π]
         * @param angle Input angle in radians
         * @return Wrapped angle in [-π, π]
         */
        static double wrap_angle(double angle);
    };

} // namespace factorama

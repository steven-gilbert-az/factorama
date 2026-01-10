#pragma once

#include "factorama/base_types.hpp"
#include "factorama/pose_2d_variable.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace factorama
{
    /**
     * @brief Prior constraint on 2D pose (position and orientation)
     *
     * Applies a prior constraint to both position and orientation of a Pose2DVariable.
     * Handles angle wrapping gracefully - angles near -π and π are treated as equivalent.
     *
     * @code
     * Eigen::Vector3d prior_pose(1.0, 2.0, PI/4);  // x, y, theta
     * double position_sigma = 0.5;
     * double angle_sigma = 0.1;
     * auto pose_prior = std::make_shared<Pose2DPriorFactor>(
     *     factor_id++, pose_var, prior_pose, position_sigma, angle_sigma);
     * @endcode
     */
    class Pose2DPriorFactor final : public Factor
    {
    public:
        /**
         * @brief Construct 2D pose prior factor
         * @param id Unique factor identifier
         * @param pose_var 2D pose variable
         * @param pose_prior Prior pose [x, y, θ]
         * @param position_sigma Standard deviation of position measurement
         * @param angle_sigma Standard deviation of angular measurement (radians)
         */
        Pose2DPriorFactor(int id,
                         Pose2DVariable* pose_var,
                         const Eigen::Vector3d& pose_prior,
                         double position_sigma,
                         double angle_sigma);

        int residual_size() const override { return size_; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;

        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

        std::vector<Variable*> variables() override
        {
            return {pose_var_};
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_2d_prior;
        }

        std::string name() const override
        {
            return "Pose2DPriorFactor(" + pose_var_->name() + ")";
        }

        double position_weight() const { return position_weight_; }
        double angle_weight() const { return angle_weight_; }

    private:
        Pose2DVariable* pose_var_;
        Eigen::Vector3d pose_prior_;
        double position_weight_;  // 1.0 / position_sigma
        double angle_weight_;     // 1.0 / angle_sigma
        int size_ = 3;

        /**
         * @brief Wrap angle difference to [-π, π]
         * @param angle Input angle in radians
         * @return Wrapped angle in [-π, π]
         */
        static double wrap_angle(double angle);
    };

} // namespace factorama

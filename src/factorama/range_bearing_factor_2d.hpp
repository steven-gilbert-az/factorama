#pragma once

#include "factorama/base_types.hpp"
#include "factorama/pose_2d_variable.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace factorama
{
    /**
     * @brief 2D range and bearing observation factor
     *
     * Measures both the range (distance) and bearing angle from a 2D pose to a 2D
     * landmark. The residual is a 2D vector containing the range error and angular
     * difference between predicted and observed measurements in the pose's local
     * reference frame.
     *
     * @code
     * double range = 5.0;  // meters
     * double bearing_angle = PI/4;  // 45 degrees in pose frame
     * double range_sigma = 0.2;  // m
     * double bearing_sigma = 0.1;  // rad
     * auto factor = std::make_shared<RangeBearingFactor2D>(
     *     factor_id++, pose_var, landmark_var, range, bearing_angle,
     *     range_sigma, bearing_sigma);
     * @endcode
     */
    class RangeBearingFactor2D final : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_POSE = 1e-9;

    public:
        /**
         * @brief Construct 2D range-bearing observation factor
         * @param id Unique factor identifier
         * @param pose_var 2D pose variable
         * @param landmark_var 2D landmark variable (GenericVariable with size=2)
         * @param range_obs Observed range (distance) from pose to landmark
         * @param bearing_angle_obs Observed bearing angle in pose frame (radians)
         * @param range_sigma Standard deviation of range measurement
         * @param bearing_sigma Standard deviation of angular measurement (radians)
         */
        RangeBearingFactor2D(
            int id,
            Pose2DVariable* pose_var,
            Variable* landmark_var,
            double range_obs,
            double bearing_angle_obs,
            double range_sigma = 1.0,
            double bearing_sigma = 1.0);

        int residual_size() const override { return size_; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;

        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

        std::vector<Variable*> variables() override
        {
            return {pose_var_, landmark_var_};
        }

        double range_weight() const { return range_weight_; }
        double bearing_weight() const { return bearing_weight_; }

        double range_obs() const { return range_obs_; }
        double bearing_angle_obs() const { return bearing_angle_obs_; }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::range_bearing_2d;
        }

        std::string name() const override
        {
            return "RangeBearingFactor2D(" + pose_var_->name() + ", " +
                   landmark_var_->name() + ")";
        }

    private:
        Pose2DVariable* pose_var_;
        Variable* landmark_var_;
        double range_obs_;
        double bearing_angle_obs_;
        double range_weight_;
        double bearing_weight_;
        int size_ = 2;

        /**
         * @brief Wrap angle difference to [-π, π]
         * @param angle Input angle in radians
         * @return Wrapped angle in [-π, π]
         */
        static double wrap_angle(double angle);
    };

} // namespace factorama

#pragma once

#include "factorama/base_types.hpp"
#include "factorama/pose_2d_variable.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace factorama
{
    /**
     * @brief 2D bearing angle observation factor
     *
     * Measures the bearing angle from a 2D pose to a 2D landmark. The residual is
     * the angular difference between predicted and observed bearing angles in the
     * pose's local reference frame.
     *
     * @code
     * double bearing_angle = PI/4;  // 45 degrees in pose frame
     * double sigma = 0.1;  // rad
     * auto factor = std::make_shared<BearingObservationFactor2D>(
     *     factor_id++, pose_var, landmark_var, bearing_angle, sigma);
     * @endcode
     */
    class BearingObservationFactor2D final : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_POSE = 1e-9;

    public:
        /**
         * @brief Construct 2D bearing observation factor
         * @param id Unique factor identifier
         * @param pose_var 2D pose variable
         * @param landmark_var 2D landmark variable (GenericVariable with size=2)
         * @param bearing_angle_obs Observed bearing angle in pose frame (radians)
         * @param angle_sigma Standard deviation of angular measurement (radians)
         */
        BearingObservationFactor2D(
            int id,
            Pose2DVariable* pose_var,
            Variable* landmark_var,
            double bearing_angle_obs,
            double angle_sigma = 1.0);

        int residual_size() const override { return size_; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;

        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

        std::vector<Variable*> variables() override
        {
            return {pose_var_, landmark_var_};
        }

        double weight() const { return weight_; }

        double bearing_angle_obs() const { return bearing_angle_obs_; }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::bearing_observation_2d;
        }

        std::string name() const override
        {
            return "BearingObservationFactor2D(" + pose_var_->name() + ", " +
                   landmark_var_->name() + ")";
        }

    private:
        Pose2DVariable* pose_var_;
        Variable* landmark_var_;
        double bearing_angle_obs_;
        double weight_;
        int size_ = 1;

        /**
         * @brief Wrap angle difference to [-π, π]
         * @param angle Input angle in radians
         * @return Wrapped angle in [-π, π]
         */
        static double wrap_angle(double angle);
    };

} // namespace factorama

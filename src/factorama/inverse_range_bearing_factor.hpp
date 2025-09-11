#pragma once

#include <memory>
#include <Eigen/Core>
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/inverse_range_variable.hpp"

namespace factorama
{

    class InverseRangeBearingFactor : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_CAMERA = 1e-9;
        static constexpr double MIN_INVERSE_RANGE = 1e-9;

    public:
        InverseRangeBearingFactor(
            int id,
            PoseVariable* pose_var,
            InverseRangeVariable* inverse_range_variable,
            const Eigen::Vector3d &bearing_C_observed,
            double angle_sigma = 1.0)
            : id_(id),
              pose_var_(pose_var),
              inverse_range_var_(inverse_range_variable),
              bearing_C_obs_(bearing_C_observed.normalized()),
              weight_(1.0 / angle_sigma)
        {
            assert(pose_var != nullptr && "pose_var cannot be nullptr");
            assert(inverse_range_variable != nullptr && "inverse_range_variable cannot be nullptr");
            assert(angle_sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        std::vector<Variable *> variables() override
        {
            return {pose_var_, inverse_range_var_};
        }

        Eigen::VectorXd compute_residual() const override
        {
            // 1. Get world to camera dcm
            const Eigen::Matrix3d &dcm_CW = pose_var_->dcm_CW();
            const Eigen::Vector3d &pos_C_W = pose_var_->pos_W();

            // 2. Get 3D point from inverse range
            Eigen::Vector3d landmark_pos_W = inverse_range_var_->pos_W();

            // 3. Transform to camera frame
            Eigen::Vector3d pos_C = dcm_CW * (landmark_pos_W - pos_C_W);

            // 4. Normalize to unit vector
            Eigen::Vector3d bearing_C = pos_C.normalized();

            // 5. Compute weighted residual
            Eigen::Vector3d res = weight_ * (bearing_C - bearing_C_obs_);
            return res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;

        int residual_size() const override
        {
            int residual = 3;
            return residual;
        }

        double weight() const
        {
            return weight_;
        }

        const Eigen::Vector3d &bearing_C_obs() const
        {
            return bearing_C_obs_;
        }

        std::string name() const override
        {
            return "InverseRangeBearing";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::inverse_range_bearing;
        }

    private:
        int id_;
        PoseVariable* pose_var_;
        InverseRangeVariable* inverse_range_var_;
        Eigen::Vector3d bearing_C_obs_; // In camera frame, normalized
        double weight_;
    };

} // namespace factorama
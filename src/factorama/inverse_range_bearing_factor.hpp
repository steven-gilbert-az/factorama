#pragma once

#include <memory>
#include <Eigen/Core>
#include "factorama/base_types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/inverse_range_variable.hpp"

namespace factorama
{

    /**
     * @brief Bearing observation factor for inverse range landmarks
     *
     * Represents directional measurements from cameras to inverse range landmarks.
     * The factor constrains the predicted bearing (from camera pose to inverse range landmark)
     * to match the observed bearing direction in the camera frame.
     *
     * @code
     * Eigen::Vector3d bearing_vector(0.0, 0.0, 1.0);
     * double bearing_sigma = 0.1;
     * auto bearing_factor = std::make_shared<InverseRangeBearingFactor>(
     *     factor_id++, camera_pose, inv_range_landmark, bearing_vector, bearing_sigma);
     * @endcode
     */
    class InverseRangeBearingFactor final : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_CAMERA = 1e-9;
        static constexpr double MIN_INVERSE_RANGE = 1e-9;

    public:
        /**
         * @brief Construct inverse range bearing factor
         * @param id Unique factor identifier
         * @param pose_var Camera pose variable
         * @param inverse_range_variable Inverse range landmark variable
         * @param bearing_C_observed Unit bearing vector in camera frame
         * @param angle_sigma Standard deviation of angular measurement (radians)
         */
        InverseRangeBearingFactor(
            int id,
            PoseVariable* pose_var,
            InverseRangeVariable* inverse_range_variable,
            const Eigen::Vector3d &bearing_C_observed,
            double angle_sigma = 1.0)
            : pose_var_(pose_var),
              inverse_range_var_(inverse_range_variable),
              bearing_C_obs_(bearing_C_observed.normalized()),
              weight_(1.0 / angle_sigma),
              size_(3)
        {
            id_ = id;
            assert(pose_var != nullptr && "pose_var cannot be nullptr");
            assert(inverse_range_variable != nullptr && "inverse_range_variable cannot be nullptr");
            assert(angle_sigma > 0.0 && "Sigma must be greater than zero");
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

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override
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
            result = weight_ * (bearing_C - bearing_C_obs_);
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;

        int residual_size() const override
        {
            return size_;
        }

        double weight() const
        {
            return weight_;
        }

        const Eigen::Vector3d &bearing_C_obs() const
        {
            return bearing_C_obs_;
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::inverse_range_bearing;
        }

    private:
        PoseVariable* pose_var_;
        InverseRangeVariable* inverse_range_var_;
        Eigen::Vector3d bearing_C_obs_; // In camera frame, normalized
        double weight_;
        int size_;
    };

} // namespace factorama
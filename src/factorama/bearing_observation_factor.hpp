#pragma once
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include <memory>

namespace factorama
{

    /**
     * @brief Bearing observation factor for camera measurements
     *
     * Represents directional measurements from cameras to 3D landmarks. The factor
     * constrains the predicted bearing (from camera pose to landmark) to match
     * the observed bearing direction in the camera frame.
     *
     * @code
     * Eigen::Vector3d bearing_vector(0.0, 0.0, 1.0);
     * double bearing_sigma = 0.1;
     * auto bearing_factor = std::make_shared<BearingObservationFactor>(
     *     factor_id++, camera_pose, landmark, bearing_vector, bearing_sigma);
     * @endcode
     */
    class BearingObservationFactor : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_CAMERA = 1e-9;

    public:
        /**
         * @brief Construct bearing observation factor
         * @param id Unique factor identifier
         * @param pose_var Camera pose variable
         * @param landmark_var 3D landmark variable
         * @param bearing_C_observed Unit bearing vector in camera frame
         * @param angle_sigma Standard deviation of angular measurement (radians)
         */
        BearingObservationFactor(
            int id,
            PoseVariable* pose_var,
            LandmarkVariable* landmark_var,
            const Eigen::Vector3d &bearing_C_observed,
            double angle_sigma = 1.0)
            : id_(id),
              pose_var_(pose_var),
              landmark_var_(landmark_var),
              bearing_C_obs_(bearing_C_observed.normalized()),
              weight_(1.0 / angle_sigma)
        {
            assert(pose_var != nullptr && "pose_var cannot be nullptr");
            assert(landmark_var != nullptr && "landmark_var cannot be nullptr");
            assert(angle_sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::Vector3d pos_W = landmark_var_->pos_W();
            Eigen::Vector3d pos_W_cam = pose_var_->pos_W();
            Eigen::Matrix3d dcm_CW = pose_var_->dcm_CW();

            Eigen::Vector3d delta_W = pos_W - pos_W_cam;
            Eigen::Vector3d bearing_C_pred = (dcm_CW * delta_W).normalized();

            return weight_ * (bearing_C_pred - bearing_C_obs_);
        }

        int residual_size() const override
        {
            return 3;
        }

        std::string name() const override
        {
            return "BearingObs";
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians_out) const override;

        std::vector<Variable *> variables() override
        {
            return {pose_var_, landmark_var_};
        }

        double weight() const override
        {
            return weight_;
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::bearing_observation;
        }

    private:
        int id_;
        PoseVariable* pose_var_;
        LandmarkVariable* landmark_var_;
        Eigen::Vector3d bearing_C_obs_;
        double weight_;
    };
}
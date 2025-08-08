#pragma once
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include <memory>

namespace factorama
{

    class BearingObservationFactor : public Factor
    {
        static constexpr double MIN_DISTANCE_FROM_CAMERA = 1e-9;

    public:
        BearingObservationFactor(
            int id,
            std::shared_ptr<PoseVariable> pose_var,
            std::shared_ptr<LandmarkVariable> landmark_var,
            const Eigen::Vector3d &bearing_C_observed,
            double angle_sigma = 1.0,
            bool do_so3_nudge = true)
            : id_(id),
              pose_var_(std::move(pose_var)),
              landmark_var_(std::move(landmark_var)),
              bearing_C_obs_(bearing_C_observed.normalized()),
              weight_(1.0 / angle_sigma),
              do_so3_nudge_(do_so3_nudge)
        {
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

        std::vector<std::shared_ptr<Variable>> variables() override
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
        std::shared_ptr<PoseVariable> pose_var_;
        std::shared_ptr<LandmarkVariable> landmark_var_;
        Eigen::Vector3d bearing_C_obs_;
        double weight_;
        bool do_so3_nudge_;
    };
}
#pragma once
#include <cassert>
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/random_utils.hpp"
#include "factorama/rotation_variable.hpp"

namespace factorama
{

    class PosePositionBetweenFactor : public Factor
    {
    public:
        PosePositionBetweenFactor(int id,
                                  PoseVariable* pose_a,
                                  PoseVariable* pose_b,
                                  Variable* measured_diff, // expected delta: pos_b - pos_a
                                  double sigma = 1.0)
            : id_(id), pose_a_(pose_a), pose_b_(pose_b), measured_diff_(measured_diff), weight_(1.0 / sigma)
        {
            assert(pose_a != nullptr && "pose_a cannot be nullptr");
            assert(pose_b != nullptr && "pose_b cannot be nullptr");
            assert(measured_diff != nullptr && "measured_diff cannot be nullptr");
            assert(measured_diff_->size() == 3 && "PosePositionBetweenFactor: measured_diff must be 3D");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        int residual_size() const override
        {
            return 3;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::Vector3d diff = pose_b_->pos_W() - pose_a_->pos_W();
            Eigen::Vector3d res = diff - measured_diff_->value();
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            jacobians.clear();

            // Jacobian w.r.t pose_a (negative identity in position block)
            if (pose_a_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                Eigen::MatrixXd J_a = Eigen::MatrixXd::Zero(3, 6);
                J_a.block<3, 3>(0, 0) = -weight_ * Eigen::Matrix3d::Identity();
                jacobians.emplace_back(J_a);
            }

            // Jacobian w.r.t pose_b (positive identity in position block)
            if (pose_b_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                Eigen::MatrixXd J_b = Eigen::MatrixXd::Zero(3, 6);
                J_b.block<3, 3>(0, 0) = weight_ * Eigen::Matrix3d::Identity();
                jacobians.emplace_back(J_b);
            }

            // Jacobian w.r.t measured_diff
            if (measured_diff_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                jacobians.emplace_back(-weight_ * Eigen::Matrix3d::Identity());
            }
        }

        std::vector<Variable *> variables() override
        {
            return {pose_a_, pose_b_, measured_diff_};
        }

        double weight() const override
        {
            return weight_;
        }

        std::string name() const override
        {
            return "PosePositionBetweenFactor(" + pose_a_->name() + ", " + pose_b_->name() + ", " + measured_diff_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_position_between;
        }

    private:
        int id_;
        PoseVariable* pose_a_;
        PoseVariable* pose_b_;
        Variable* measured_diff_;
        double weight_;
    };

    class PoseOrientationBetweenFactor : public Factor
    {
    public:
        PoseOrientationBetweenFactor(
            int id,
            PoseVariable* pose1,
            PoseVariable* pose2,
            RotationVariable* calibration_rotation_12,
            double angle_sigma = 1.0)
            : id_(id),
              pose1_(pose1),
              pose2_(pose2),
              calibration_rotation_12_(calibration_rotation_12),
              weight_(1.0 / angle_sigma)
        {
            assert(pose1 != nullptr && "pose1 cannot be nullptr");
            assert(pose2 != nullptr && "pose2 cannot be nullptr");
            assert(calibration_rotation_12 != nullptr && "calibration_rotation_12 cannot be nullptr");
            assert(angle_sigma > 0.0 && "Sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        // Residual dimension is 3 for so(3)
        int residual_size() const override { return 3; }

        // Residual = LogMapSO3(dcm_S2W * dcm_WS1 * dcm_S1_S2)
        Eigen::VectorXd compute_residual() const override
        {
            return compute_residual(pose1_, pose2_, calibration_rotation_12_);
        }

        Eigen::VectorXd compute_residual(
            PoseVariable *pose1,
            PoseVariable *pose2,
            RotationVariable *calibration_rotation_12) const
        {
            // Let P1 be the coordinate frame of the sensor that corresponds to Pose1
            // Let P2 be the coordinate frame of the sensor that corresponds to pose2
            Eigen::Matrix3d dcm_P1W = pose1->dcm_CW();
            Eigen::Matrix3d dcm_P2W = pose2->dcm_CW();
            Eigen::Matrix3d dcm_P1_P2 = calibration_rotation_12->rotation();
            //                       dcm_S2W * dcm_WS1 * dcm_S1_S2 ~= dcm_S2_S2 ~= Identity
            Eigen::Matrix3d error = dcm_P2W * dcm_P1W.transpose() * dcm_P1_P2;

            Eigen::Vector3d res = LogMapSO3(error);
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;

        std::vector<Variable *> variables() override
        {
            return {pose1_, pose2_, calibration_rotation_12_};
        }

        double weight() const override { return weight_; }

        std::string name() const override { return "PoseOrientationBetweenFactor"; }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_orientation_between;
        }

    private:
        int id_;
        PoseVariable* pose1_;
        PoseVariable* pose2_;
        RotationVariable* calibration_rotation_12_;
        double weight_;
    };
} // namespace factorama
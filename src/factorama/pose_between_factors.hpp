#pragma once
#include <cassert>
#include "factorama/base_types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/random_utils.hpp"
#include "factorama/rotation_variable.hpp"

namespace factorama
{

    /**
     * @brief Position-only relative constraint between pose variables
     *
     * Constrains the position difference between two poses to match a measured difference variable.
     * Only affects the translation components of poses, ignoring rotation.
     *
     * @code
     * auto position_constraint = std::make_shared<PosePositionBetweenFactor>(
     *     factor_id++, pose_a, pose_b, measured_delta_variable, sigma);
     * @endcode
     */
    class PosePositionBetweenFactor final : public Factor
    {
    public:
        /**
         * @brief Construct position between factor
         * @param id Unique factor identifier
         * @param pose_a First pose variable
         * @param pose_b Second pose variable
         * @param measured_diff Variable representing measured position difference
         *        - If local_frame=true: measured in pose_a's frame
         *        - If local_frame=false: measured in world frame (pos_b_W - pos_a_W)
         * @param sigma Standard deviation of measurement
         * @param local_frame If true, measurement is in pose_a's frame; if false, in world frame (default: false)
         */
        PosePositionBetweenFactor(int id,
                                  PoseVariable* pose_a,
                                  PoseVariable* pose_b,
                                  Variable* measured_diff,
                                  double sigma = 1.0,
                                  bool local_frame = false)
            : pose_a_(pose_a), pose_b_(pose_b), measured_diff_(measured_diff),
              weight_(1.0 / sigma), local_frame_(local_frame), size_(3)
        {
            id_ = id;
            assert(pose_a != nullptr && "pose_a cannot be nullptr");
            assert(pose_b != nullptr && "pose_b cannot be nullptr");
            assert(measured_diff != nullptr && "measured_diff cannot be nullptr");
            assert(measured_diff_->size() == 3 && "PosePositionBetweenFactor: measured_diff must be 3D");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
        }

        int residual_size() const override
        {
            return size_;
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::Vector3d diff_W = pose_b_->pos_W() - pose_a_->pos_W();
            Eigen::Vector3d res;

            if (local_frame_)
            {
                // Measurement in pose_a's local frame: transform world difference to pose_a's frame
                Eigen::Matrix3d dcm_AW = pose_a_->dcm_CW();
                Eigen::Vector3d diff_A = dcm_AW * diff_W;
                res = diff_A - measured_diff_->value();
            }
            else
            {
                // Measurement in world frame (original behavior)
                res = diff_W - measured_diff_->value();
            }

            return weight_ * res;
        }

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override
        {
            Eigen::Vector3d diff_W = pose_b_->pos_W() - pose_a_->pos_W();
            Eigen::Vector3d res;

            if (local_frame_)
            {
                // Measurement in pose_a's local frame: transform world difference to pose_a's frame
                Eigen::Matrix3d dcm_AW = pose_a_->dcm_CW();
                Eigen::Vector3d diff_A = dcm_AW * diff_W;
                res = diff_A - measured_diff_->value();
            }
            else
            {
                // Measurement in world frame (original behavior)
                res = diff_W - measured_diff_->value();
            }

            result = weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            // Ensure jacobians vector has correct size for 3 variables
            if(jacobians.size() == 0) {
                jacobians.resize(3);
            }
            else if(jacobians.size() != 3) {
                jacobians.clear();
                jacobians.resize(3);
            }

            if (local_frame_)
            {
                // Local frame: residual = dcm_AW * (pos_b_W - pos_a_W) - measured_diff
                Eigen::Vector3d diff_W = pose_b_->pos_W() - pose_a_->pos_W();
                Eigen::Matrix3d dcm_AW = pose_a_->dcm_CW();

                // Jacobian w.r.t pose_a
                if (pose_a_->is_constant())
                {
                    jacobians[0] = Eigen::MatrixXd();
                }
                else
                {
                    if(jacobians[0].rows() != size_ || jacobians[0].cols() != 6) {
                        jacobians[0].resize(size_, 6);
                    }
                    jacobians[0].setZero();
                    // Position part: d(dcm_AW * diff_W)/d(pos_a_W) = -dcm_AW
                    jacobians[0].block<3, 3>(0, 0) = -weight_ * dcm_AW;
                    // Rotation part: d(dcm_AW * diff_W)/d(rot_a) = -[dcm_AW * diff_W]_×
                    // (negative sign due to left perturbation: R_new = exp([δθ]_×) * R)
                    Eigen::Vector3d diff_A = dcm_AW * diff_W;
                    Eigen::Matrix3d skew_diff_A;
                    skew_diff_A << 0, -diff_A(2), diff_A(1),
                                   diff_A(2), 0, -diff_A(0),
                                   -diff_A(1), diff_A(0), 0;
                    jacobians[0].block<3, 3>(0, 3) = -weight_ * skew_diff_A;
                }

                // Jacobian w.r.t pose_b
                if (pose_b_->is_constant())
                {
                    jacobians[1] = Eigen::MatrixXd();
                }
                else
                {
                    if(jacobians[1].rows() != size_ || jacobians[1].cols() != 6) {
                        jacobians[1].resize(size_, 6);
                    }
                    jacobians[1].setZero();
                    // Position part: d(dcm_AW * diff_W)/d(pos_b_W) = dcm_AW
                    jacobians[1].block<3, 3>(0, 0) = weight_ * dcm_AW;
                    // Rotation part: no dependency on pose_b's rotation
                }
            }
            else
            {
                // World frame: residual = (pos_b_W - pos_a_W) - measured_diff (original behavior)
                // Jacobian w.r.t pose_a (negative identity in position block)
                if (pose_a_->is_constant())
                {
                    jacobians[0] = Eigen::MatrixXd();
                }
                else
                {
                    if(jacobians[0].rows() != size_ || jacobians[0].cols() != 6) {
                        jacobians[0].resize(size_, 6);
                    }
                    jacobians[0].setZero();
                    jacobians[0].block<3, 3>(0, 0).diagonal().array() = -weight_;
                }

                // Jacobian w.r.t pose_b (positive identity in position block)
                if (pose_b_->is_constant())
                {
                    jacobians[1] = Eigen::MatrixXd();
                }
                else
                {
                    if(jacobians[1].rows() != size_ || jacobians[1].cols() != 6) {
                        jacobians[1].resize(size_, 6);
                    }
                    jacobians[1].setZero();
                    jacobians[1].block<3, 3>(0, 0).diagonal().array() = weight_;
                }
            }

            // Jacobian w.r.t measured_diff (same for both modes)
            if (measured_diff_->is_constant())
            {
                jacobians[2] = Eigen::MatrixXd();
            }
            else
            {
                if(jacobians[2].rows() != size_ || jacobians[2].cols() != size_) {
                    jacobians[2].resize(size_, size_);
                }
                jacobians[2].setZero();
                jacobians[2].diagonal().array() = -weight_;
            }
        }

        std::vector<Variable *> variables() override
        {
            return {pose_a_, pose_b_, measured_diff_};
        }

        double weight() const
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
        PoseVariable* pose_a_;
        PoseVariable* pose_b_;
        Variable* measured_diff_;
        double weight_;
        bool local_frame_;
        int size_;
    };

    /**
     * @brief Orientation-only relative constraint between pose variables
     *
     * Constrains the rotation difference between two poses to match a calibration rotation variable.
     * Only affects the rotation components of poses, ignoring translation. Commonly used for
     * camera-IMU calibration where relative orientation is known or estimated.
     *
     * @code
     * auto orientation_constraint = std::make_shared<PoseOrientationBetweenFactor>(
     *     factor_id++, pose1, pose2, calibration_rotation, angle_sigma);
     * @endcode
     */
    class PoseOrientationBetweenFactor final : public Factor
    {
    public:
        /**
         * @brief Construct orientation between factor
         * @param id Unique factor identifier
         * @param pose1 First pose variable
         * @param pose2 Second pose variable
         * @param calibration_rotation_12 Rotation variable representing relative rotation
         * @param angle_sigma Represents the standard deviation to apply to the strength of the rotational constraint (radians)
         */
        PoseOrientationBetweenFactor(
            int id,
            PoseVariable* pose1,
            PoseVariable* pose2,
            RotationVariable* calibration_rotation_12,
            double angle_sigma = 1.0)
            : pose1_(pose1),
              pose2_(pose2),
              calibration_rotation_12_(calibration_rotation_12),
              weight_(1.0 / angle_sigma),
              size_(3)
        {
            id_ = id;
            assert(pose1 != nullptr && "pose1 cannot be nullptr");
            assert(pose2 != nullptr && "pose2 cannot be nullptr");
            assert(calibration_rotation_12 != nullptr && "calibration_rotation_12 cannot be nullptr");
            assert(angle_sigma > 0.0 && "Sigma must be greater than zero");
        }

        // Residual dimension is 3 for so(3)
        int residual_size() const override { return size_; }

        // Residual = LogMapSO3(dcm_S2W * dcm_WS1 * dcm_S1_S2)
        Eigen::VectorXd compute_residual() const override
        {
            return compute_residual(pose1_, pose2_, calibration_rotation_12_);
        }

        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override
        {
            // Let P1 be the coordinate frame of the sensor that corresponds to Pose1
            // Let P2 be the coordinate frame of the sensor that corresponds to pose2
            Eigen::Matrix3d dcm_P1W = pose1_->dcm_CW();
            Eigen::Matrix3d dcm_P2W = pose2_->dcm_CW();
            Eigen::Matrix3d dcm_P1_P2 = calibration_rotation_12_->rotation();
            //                       dcm_S2W * dcm_WS1 * dcm_S1_S2 ~= dcm_S2_S2 ~= Identity
            Eigen::Matrix3d error = dcm_P2W * dcm_P1W.transpose() * dcm_P1_P2;

            Eigen::Vector3d res = LogMapSO3(error);
            result = weight_ * res;
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

        double weight() const { return weight_; }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::pose_orientation_between;
        }

    private:
        PoseVariable* pose1_;
        PoseVariable* pose2_;
        RotationVariable* calibration_rotation_12_;
        double weight_;
        int size_;
    };
} // namespace factorama
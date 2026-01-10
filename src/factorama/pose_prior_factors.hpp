#pragma once
#include <cassert>
#include "factorama/base_types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/random_utils.hpp"

namespace factorama
{

    /**
     * @brief Prior constraint on camera position
     *
     * Applies a prior constraint to the position component of a PoseVariable.
     * Useful for anchoring poses or incorporating GPS measurements.
     *
     * @code
     * double position_sigma = 0.5;
     * auto pose_position_prior = std::make_shared<PosePositionPriorFactor>(
     *     factor_id++, camera_pose, Eigen::Vector3d::Zero(), position_sigma);
     * @endcode
     */
    class PosePositionPriorFactor final : public Factor
    {
    public:
        /**
         * @brief Construct position prior factor
         * @param id Unique factor identifier
         * @param pose Camera pose variable
         * @param pos_prior Prior position in world frame
         * @param sigma Standard deviation of position measurement
         */
        PosePositionPriorFactor(int id,
                                PoseVariable* pose,
                                const Eigen::Vector3d &pos_prior,
                                double sigma = 1.0);

        int residual_size() const override { return size_; }
        double weight() const { return weight_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::pose_position_prior; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;
        std::string name() const override;

    private:
        PoseVariable* pose_;
        Eigen::Vector3d pos_prior_;
        double weight_;
        int size_ = 3;
    };

    /**
     * @brief Prior constraint on camera orientation
     *
     * Applies a prior constraint to the rotation component of a PoseVariable.
     * Useful for incorporating IMU measurements or known orientations.
     *
     * @code
     * double orientation_sigma = 0.1;
     * auto pose_orientation_prior = std::make_shared<PoseOrientationPriorFactor>(
     *     factor_id++, camera_pose, Eigen::Matrix3d::Identity(), orientation_sigma);
     * @endcode
     */
    class PoseOrientationPriorFactor final : public Factor
    {
    public:
        /**
         * @brief Construct orientation prior factor
         * @param id Unique factor identifier
         * @param pose Camera pose variable
         * @param dcm_CW_prior Prior rotation - direction cosine matrix (DCM)
         * @param sigma Standard deviation of angular prior (radians)
         */
        PoseOrientationPriorFactor(int id,
                                   PoseVariable* pose,
                                   const Eigen::Matrix3d &dcm_CW_prior,
                                   double sigma = 1.0);

        int residual_size() const override { return size_; }
        double weight() const { return weight_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::pose_orientation_prior; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;
        std::string name() const override;

    private:
        PoseVariable* pose_;
        Eigen::Vector3d rot_CW_prior_;
        double weight_;
        int size_ = 3;
    };

}
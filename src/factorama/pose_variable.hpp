#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <iostream>
#include "factorama/base_types.hpp" // assuming VariableType is defined here
#include "factorama/random_utils.hpp"

namespace factorama
{
    /**
     * @brief 6-DOF camera pose variable with SE(3) parameterization
     *
     * Represents camera poses as 6-element vectors: [tx, ty, tz, rx, ry, rz]
     * where translation is in world frame and rotation uses rodrigues vector (axis * angle) representation.
     * The pose transforms points from world to camera frame.
     *
     * @code
     * Eigen::Matrix<double, 6, 1> pose_vec = Eigen::Matrix<double, 6, 1>::Zero();
     * auto camera_pose = std::make_shared<PoseVariable>(1, pose_vec);
     * @endcode
     */
    class PoseVariable : public Variable
    {
    public:
        /**
         * @brief Construct pose variable from 6-element vector
         * @param id Unique variable identifier
         * @param pose_CW_init Initial pose as [tx, ty, tz, rx, ry, rz]
         */
        PoseVariable(int id, const Eigen::Matrix<double, 6, 1> &pose_CW_init);

        /**
         * @brief Construct pose variable from position and DCM (rotation matrix)
         * @param id Unique variable identifier
         * @param pos_W Camera position in world frame
         * @param dcm_CW DCM (rotation matrix) from world to camera frame
         */
        PoseVariable(int id, const Eigen::Vector3d pos_W, const Eigen::Matrix3d dcm_CW);

        int size() const override { return 6; }
        const Eigen::VectorXd &value() const override { return pose_CW_; }
        VariableType::VariableTypeEnum type() const override { return VariableType::pose; }

        /**
         * @brief Get camera position in world frame
         * @return 3D position vector
         */
        Eigen::Vector3d pos_W() const { return pose_CW_.segment<3>(0); }

        /**
         * @brief Get rotation rodrigues vector
         * @return 3D rodrigues vector (axis * angle)
         */
        Eigen::Vector3d rot_CW() const { return pose_CW_.segment<3>(3); }

        void set_value_from_vector(const Eigen::VectorXd &x) override;
        void apply_increment(const Eigen::VectorXd &dx) override;

        /**
         * @brief Get DCM (rotation matrix) from rodrigues vector
         * @return 3x3 rotation matrix from world to camera frame
         */
        const Eigen::Matrix3d & dcm_CW() const;

        /**
         * @brief Set pose from 6-element vector
         * @param pose New pose as [tx, ty, tz, rx, ry, rz]
         */
        void set_pose_vector(Eigen::Matrix<double, 6, 1> pose);

        /**
         * @brief Set camera position in world frame
         * @param pos_W New 3D position vector
         */
        void set_pos_W(const Eigen::Vector3d &pos_W);

        /**
         * @brief Set rotation from DCM (rotation matrix)
         * @param dcm_CW New rotation matrix from world to camera frame
         */
        void set_dcm_CW(const Eigen::Matrix3d &dcm_CW);
        void print() const override;
        std::shared_ptr<Variable> clone() const override;

    private:

        void recompute_dcm_CW();
        //Eigen::Matrix<double,6,1> pose_CW_; // [tx, ty, tz, rx, ry, rz] representing pose_CW
        Eigen::VectorXd pose_CW_;
        Eigen::Matrix3d dcm_CW_; // Temporary storage for the dcm.
    };
}

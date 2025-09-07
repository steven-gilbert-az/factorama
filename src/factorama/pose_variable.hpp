#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <iostream>
#include "factorama/types.hpp" // assuming VariableType is defined here
#include "factorama/random_utils.hpp"

namespace factorama
{
    class PoseVariable : public Variable
    {
    public:
        PoseVariable(int id, const Eigen::Matrix<double, 6, 1> &pose_CW_init, bool do_so3_nudge = true)
            : id_(id), pose_CW_(pose_CW_init), do_so3_nudge_(do_so3_nudge) {}

        PoseVariable(int id, const Eigen::Vector3d pos_W, const Eigen::Matrix3d dcm_CW, bool do_so3_nudge = true)
        {
            id_ = id;
            pose_CW_ = Eigen::VectorXd(6);
            pose_CW_.segment<3>(0) = pos_W;

            Eigen::AngleAxisd aa(dcm_CW);
            pose_CW_.segment<3>(3) = aa.axis() * aa.angle();
            do_so3_nudge_ = do_so3_nudge;
        }

        int size() const override { return 6; }

        const Eigen::VectorXd &value() const override { return pose_CW_; }

        void set_value_from_vector(const Eigen::VectorXd &x) override
        {
            pose_CW_ = x;
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            if (dx.size() != size())
            {
                throw std::runtime_error("apply_increment(): size mismatch");
            }

            // Translation increment: linear add
            pose_CW_.segment<3>(0) += dx.segment<3>(0);

            // Rotation increment:
            if (do_so3_nudge_)
            {
                // Current rotation as matrix
                Eigen::Matrix3d R = dcm_CW();

                // Increment rotation vector (tangent space)
                Eigen::Vector3d delta_rot = dx.segment<3>(3);

                // Apply incremental rotation on manifold: R_new = exp(delta_rot) * R
                Eigen::Matrix3d R_new = ExpMapSO3(delta_rot) * R; // right - multiply perturbation convention

                // Update rotation vector (log map)
                Eigen::Vector3d rot_vec_new = LogMapSO3(R_new);

                pose_CW_.segment<3>(3) = rot_vec_new;
            }
            else
            {
                // Linear add for rotation vector (less correct, but easier)
                pose_CW_.segment<3>(3) += dx.segment<3>(3);
            }
        }

        int id() const override { return id_; }

        VariableType::VariableTypeEnum type() const override
        {
            return VariableType::pose;
        }
        bool is_constant() const override { return is_constant_; }
        void set_is_constant(bool val) { is_constant_ = val; }

        std::string name() const override
        {
            return "Pose" + std::to_string(id());
        }

        // Utility: extract translation component (camera position in world frame)
        Eigen::Vector3d pos_W() const
        {
            return pose_CW_.segment<3>(0); // [tx, ty, tz]
        }

        Eigen::Vector3d rot_CW() const
        {
            return pose_CW_.segment<3>(3);
        }

        // Utility: extract rotation matrix dcm_CW (from world to camera frame)
        Eigen::Matrix3d dcm_CW() const
        {
            Eigen::Vector3d rot_CW_tmp = rot_CW(); // [rx, ry, rz]
            double angle = rot_CW_tmp.norm();

            if (angle < 1e-8)
            {
                return Eigen::Matrix3d::Identity();
            }

            Eigen::Vector3d axis = rot_CW_tmp / angle;
            return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }

        void set_pose_vector(Eigen::Matrix<double, 6, 1> pose)
        {
            pose_CW_ = pose;
        }

        void set_pos_W(const Eigen::Vector3d &pos_W)
        {
            pose_CW_.segment<3>(0) = pos_W;
        }

        void set_dcm_CW(const Eigen::Matrix3d &dcm_CW)
        {
            Eigen::AngleAxisd aa(dcm_CW);
            pose_CW_.segment<3>(3) = aa.axis() * aa.angle();
        }

        void print() const override
        {
            std::cout << name() << std::endl;
            std::cout << "pos:" << pos_W() << std::endl;
            std::cout << "dcm_CW: " << dcm_CW() << std::endl;
        }

        bool do_so3_nudge() {
            return do_so3_nudge_;
        }
        
        std::shared_ptr<Variable> clone() const override {
            return std::make_shared<PoseVariable>(*this);
        }

    private:
        int id_;
        Eigen::VectorXd pose_CW_; // [tx, ty, tz, rx, ry, rz] representing pose_CW
        bool is_constant_ = false;
        bool do_so3_nudge_ = true;
    };
}

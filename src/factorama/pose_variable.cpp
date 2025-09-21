#include "pose_variable.hpp"
#include <iostream>
#include <memory>

namespace factorama
{

PoseVariable::PoseVariable(int id, const Eigen::Matrix<double, 6, 1> &pose_CW_init)
    : id_(id), pose_CW_(pose_CW_init) {}

PoseVariable::PoseVariable(int id, const Eigen::Vector3d pos_W, const Eigen::Matrix3d dcm_CW)
{
    id_ = id;
    pose_CW_ = Eigen::VectorXd(6);
    pose_CW_.segment<3>(0) = pos_W;

    Eigen::AngleAxisd aa(dcm_CW);
    pose_CW_.segment<3>(3) = aa.axis() * aa.angle();
}

void PoseVariable::set_value_from_vector(const Eigen::VectorXd &x)
{
    pose_CW_ = x;
}

void PoseVariable::apply_increment(const Eigen::VectorXd &dx)
{
    if (dx.size() != size())
    {
        throw std::runtime_error("apply_increment(): size mismatch");
    }

    // Translation increment: linear add
    pose_CW_.segment<3>(0) += dx.segment<3>(0);

    // Rotation increment:
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

std::string PoseVariable::name() const
{
    return "Pose" + std::to_string(id());
}

Eigen::Matrix3d PoseVariable::dcm_CW() const
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

void PoseVariable::set_pose_vector(Eigen::Matrix<double, 6, 1> pose)
{
    pose_CW_ = pose;
}

void PoseVariable::set_pos_W(const Eigen::Vector3d &pos_W)
{
    pose_CW_.segment<3>(0) = pos_W;
}

void PoseVariable::set_dcm_CW(const Eigen::Matrix3d &dcm_CW)
{
    Eigen::AngleAxisd aa(dcm_CW);
    pose_CW_.segment<3>(3) = aa.axis() * aa.angle();
}

void PoseVariable::print() const
{
    std::cout << name() << std::endl;
    std::cout << "pos:" << pos_W() << std::endl;
    std::cout << "dcm_CW: " << dcm_CW() << std::endl;
}

std::shared_ptr<Variable> PoseVariable::clone() const
{
    return std::make_shared<PoseVariable>(*this);
}

} // namespace factorama
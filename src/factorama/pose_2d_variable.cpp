#include "pose_2d_variable.hpp"
#include "factorama/random_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace factorama
{
    Pose2DVariable::Pose2DVariable(int id, const Eigen::Vector3d& pose_2d)
        : pose_2d_(pose_2d)
    {
        id_ = id;
        // Wrap initial angle to [-π, π]
        pose_2d_(2) = wrap_angle(pose_2d_(2));
    }

    void Pose2DVariable::set_value_from_vector(const Eigen::VectorXd& x)
    {
        if (x.size() != 3)
        {
            throw std::runtime_error("Pose2DVariable::set_value_from_vector(): size must be 3");
        }
        pose_2d_ = x;
        // Wrap angle to avoid accumulation outside [-π, π]
        pose_2d_(2) = wrap_angle(pose_2d_(2));
    }

    void Pose2DVariable::apply_increment(const Eigen::VectorXd& dx)
    {
        if (dx.size() != 3)
        {
            throw std::runtime_error("Pose2DVariable::apply_increment(): size must be 3");
        }

        // Translation: simple addition (Euclidean)
        pose_2d_(0) += dx(0);  // x
        pose_2d_(1) += dx(1);  // y

        // Rotation: additive with wrapping
        pose_2d_(2) += dx(2);  // θ
        pose_2d_(2) = wrap_angle(pose_2d_(2));
    }

    VariableType::VariableTypeEnum Pose2DVariable::type() const
    {
        return VariableType::pose_2d;
    }

    Eigen::Matrix2d Pose2DVariable::dcm_2d() const
    {
        double c = std::cos(pose_2d_(2));
        double s = std::sin(pose_2d_(2));
        Eigen::Matrix2d R;
        R << c, -s,
             s,  c;
        return R;
    }

    void Pose2DVariable::set_pos_2d(const Eigen::Vector2d& pos)
    {
        pose_2d_.head<2>() = pos;
    }

    void Pose2DVariable::set_theta(double theta)
    {
        pose_2d_(2) = wrap_angle(theta);
    }

    void Pose2DVariable::print() const
    {
        std::cout << name() << std::endl;
        std::cout << "Position: [" << pose_2d_(0) << ", " << pose_2d_(1) << "]" << std::endl;
        std::cout << "Theta: " << pose_2d_(2) << " rad (" << (pose_2d_(2) * 180.0 / PI) << " deg)" << std::endl;
    }

    std::shared_ptr<Variable> Pose2DVariable::clone() const
    {
        return std::make_shared<Pose2DVariable>(*this);
    }

    double Pose2DVariable::wrap_angle(double angle)
    {
        // Wrap angle to [-π, π] using atan2 for robust handling
        return std::atan2(std::sin(angle), std::cos(angle));
    }

} // namespace factorama

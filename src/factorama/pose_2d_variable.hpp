#pragma once

#include "factorama/base_types.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

namespace factorama
{
    /**
     * @brief 2D pose variable for planar problems (SE(2))
     *
     * Represents 2D poses as [x, y, θ] where (x,y) is position and θ is orientation.
     * Used for planar SLAM, 2D robot navigation, and other 2D problems.
     *
     * @code
     * Eigen::Vector3d initial_pose(1.0, 2.0, PI/4);  // x, y, theta
     * auto pose = std::make_shared<Pose2DVariable>(1, initial_pose);
     * @endcode
     */
    class Pose2DVariable : public Variable
    {
    public:
        /**
         * @brief Construct 2D pose variable
         * @param id Unique variable identifier
         * @param pose_2d Initial pose [x, y, θ]
         */
        Pose2DVariable(int id, const Eigen::Vector3d& pose_2d);

        int size() const override { return 3; }

        const Eigen::VectorXd& value() const override { return pose_2d_; }

        void set_value_from_vector(const Eigen::VectorXd& x) override;

        void apply_increment(const Eigen::VectorXd& dx) override;

        VariableType::VariableTypeEnum type() const override;

        /**
         * @brief Get 2D position
         * @return Position vector [x, y]
         */
        Eigen::Vector2d pos_2d() const { return pose_2d_.head<2>(); }

        /**
         * @brief Get orientation angle
         * @return Angle in radians
         */
        double theta() const { return pose_2d_(2); }

        /**
         * @brief Get 2D rotation matrix
         * @return 2x2 rotation matrix
         */
        Eigen::Matrix2d dcm_2d() const;

        /**
         * @brief Set 2D position
         * @param pos Position vector [x, y]
         */
        void set_pos_2d(const Eigen::Vector2d& pos);

        /**
         * @brief Set orientation angle
         * @param theta Angle in radians
         */
        void set_theta(double theta);

        void print() const override;

        std::shared_ptr<Variable> clone() const override;

    private:
        Eigen::VectorXd pose_2d_;  // [x, y, θ]

        /**
         * @brief Wrap angle to [-π, π]
         * @param angle Input angle in radians
         * @return Wrapped angle in [-π, π]
         */
        static double wrap_angle(double angle);
    };

} // namespace factorama

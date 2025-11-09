#pragma once

#include "factorama/base_types.hpp"
#include "factorama/random_utils.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace factorama
{

    /**
     * @brief Rotation-only variable for extrinsic calibration
     *
     * Represents rotations between coordinate frames using SO(3) manifold operations.
     * Commonly used for camera-IMU calibration where only relative orientation matters.
     *
     * @code
     * Eigen::Matrix3d initial_rotation = Eigen::Matrix3d::Identity();
     * auto rotation_var = std::make_shared<RotationVariable>(1, initial_rotation);
     * @endcode
     */
    class RotationVariable : public Variable
    {
    public:
        /**
         * @brief Construct rotation variable from DCM
         * @param id Unique variable identifier
         * @param dcm_AB Rotation matrix from frame A to frame B
         */
        RotationVariable(int id, const Eigen::Matrix3d &dcm_AB)
            : dcm_AB_(dcm_AB)
        {
            id_ = id;
            value_ = LogMapSO3(dcm_AB_);
        }

        int size() const override { return 3; }

        virtual VariableType::VariableTypeEnum type() const
        {
            return VariableType::extrinsic_rotation;
        }

        const Eigen::VectorXd &value() const override
        {
            return value_;
        }

        void set_value_from_vector(const Eigen::VectorXd &x) override
        {
            value_ = x;
            dcm_AB_ = ExpMapSO3(x);
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            dcm_AB_ = ExpMapSO3(dx) * dcm_AB_;
            value_ = LogMapSO3(dcm_AB_);
        }

        Eigen::Matrix3d& dcm_AB() {
            return dcm_AB_;
        }

        void print() const override
        {
            std::cout << "RotationVariable ID " << id_ << std::endl;
            std::cout << dcm_AB_ << std::endl;
        }

        const Eigen::Matrix3d &rotation() const { return dcm_AB_; }
        
        std::shared_ptr<Variable> clone() const override {
            return std::make_shared<RotationVariable>(*this);
        }

    private:
        Eigen::VectorXd value_;
        Eigen::Matrix3d dcm_AB_;
    };

} // namespace factorama
#pragma once

#include "factorama/types.hpp"
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
            : id_(id), dcm_AB_(dcm_AB)
        {
            value_ = LogMapSO3(dcm_AB_);
        }

        int size() const override { return 3; }

        int id() const override
        {
            return id_;
        }

        virtual VariableType::VariableTypeEnum type() const
        {
            return VariableType::extrinsic_rotation;
        }

        bool is_constant() const override { return is_constant_; }
        void set_is_constant(bool val) { is_constant_ = val; }

        std::string name() const override
        {
            return "ExtrinsicRotation" + std::to_string(id());
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
            std::cout << "RotationVariable (Camera-to-External) ID " << id_ << std::endl;
            std::cout << dcm_AB_ << std::endl;
        }

        const Eigen::Matrix3d &rotation() const { return dcm_AB_; }
        
        std::shared_ptr<Variable> clone() const override {
            return std::make_shared<RotationVariable>(*this);
        }

    private:
        int id_;
        Eigen::VectorXd value_;
        Eigen::Matrix3d dcm_AB_;
        bool is_constant_ = false;
    };

} // namespace factorama
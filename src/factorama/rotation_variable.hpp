#pragma once

#include "factorama/types.hpp"
#include "factorama/random_utils.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace factorama
{

    // The RotationVariable is assumed to track a rotation
    // (i.e DCM - direction cosine matrix) between pose B and pose A
    // 
    class RotationVariable : public Variable
    {
    public:
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
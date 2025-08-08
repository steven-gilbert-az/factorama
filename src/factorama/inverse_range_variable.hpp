#pragma once

#include <Eigen/Core>
#include <memory>
#include <iostream>
#include "factorama/types.hpp"

namespace factorama
{

    class InverseRangeVariable : public Variable
    {
    public:



        InverseRangeVariable(int variable_id,
                             const Eigen::Vector3d &origin_pos_W,
                             const Eigen::Vector3d &bearing_W,
                             double initial_range)
            : id_(variable_id), origin_pos_W_(origin_pos_W), bearing_W_(bearing_W.normalized())
        {
            // Store inverse range
            // inverse_range_ = 1.0 / initial_range;
            inverse_range_value_ = Eigen::VectorXd(1);
            inverse_range_value_[0] = 1.0 / initial_range;

            clip_inverse_range();
        }

        // Default min/max values for inverse range - domain specific - user can change these if they want.
        double minimum_inverse_range_ = 1e-6; // Assume meters, max range = 1million meters (1000 km) - obviously too small for satelite stuff
        double maximum_inverse_range_ = 1e3;  // minimum range = 1mm

        int id() const override
        {
            return id_;
        }
        int size() const override
        {
            return 1;
        }

        const Eigen::VectorXd &value() const override
        {
            return inverse_range_value_;
        }

        void clip_inverse_range() {
            // TODO: if we ever get logger integration, add an info or warning if it does clip it.
            inverse_range_value_[0] = std::max(inverse_range_value_[0], minimum_inverse_range_);
            inverse_range_value_[0] = std::min(inverse_range_value_[0], maximum_inverse_range_);
        }

        void set_value_from_vector(const Eigen::VectorXd &x) override
        {
            inverse_range_value_ = x;
            clip_inverse_range();
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            if (dx.size() != size())
            {
                throw std::runtime_error("apply_increment(): size mismatch");
            }
            inverse_range_value_ += dx;
            clip_inverse_range();
        }

        Eigen::Vector3d pos_W() const
        {
            const double inv_range = inverse_range_value_[0];
            if (!std::isfinite(inv_range) || std::abs(inv_range) < 1e-9)
            {
                std::cerr << "inverse range NAN or Zero" << std::endl;
                return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
            }
            return origin_pos_W_ + (1.0 / inv_range) * bearing_W_;
        }

        const Eigen::Vector3d &origin_pos_W() const
        {
            return origin_pos_W_;
        }

        const Eigen::Vector3d &bearing_W() const
        {
            return bearing_W_;
        }

        VariableType::VariableTypeEnum type() const override
        {
            return VariableType::inverse_range_landmark;
        }
        bool is_constant() const override { return is_constant_; }
        void set_is_constant(bool val) { is_constant_ = val; }

        double inverse_range() const
        {
            return inverse_range_value_[0];
        }

        std::string name() const override
        {
            return "InverseRangeVariable" + std::to_string(id());
        }

        void print() const override
        {
            std::cout << name() << std::endl;
            std::cout << "Origin: " << origin_pos_W() << std::endl;
            std::cout << "Range: " << 1.0 / inverse_range() << std::endl;
            std::cout << "Pos: " << pos_W() << std::endl;
        }
        
        std::shared_ptr<Variable> clone() const override {
            return std::make_shared<InverseRangeVariable>(*this);
        }

    private:
        int id_;
        Eigen::Vector3d origin_pos_W_;
        Eigen::Vector3d bearing_W_;
        Eigen::VectorXd inverse_range_value_;
        bool is_constant_ = false;
    };

    using InverseRangeVariablePtr = std::shared_ptr<InverseRangeVariable>;

} // namespace factorama
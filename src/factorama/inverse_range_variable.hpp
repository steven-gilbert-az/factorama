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
                             double initial_range);

        // Default min/max values for inverse range - domain specific - user can change these if they want.
        double minimum_inverse_range_ = 1e-6; // Assume meters, max range = 1million meters (1000 km) - obviously too small for satelite stuff
        double maximum_inverse_range_ = 1e3;  // minimum range = 1mm

        int id() const override { return id_; }
        int size() const override { return 1; }
        const Eigen::VectorXd &value() const override { return inverse_range_value_; }
        const Eigen::Vector3d &origin_pos_W() const { return origin_pos_W_; }
        const Eigen::Vector3d &bearing_W() const { return bearing_W_; }
        VariableType::VariableTypeEnum type() const override { return VariableType::inverse_range_landmark; }
        bool is_constant() const override { return is_constant_; }
        void set_is_constant(bool val) { is_constant_ = val; }
        double inverse_range() const { return inverse_range_value_[0]; }

        void clip_inverse_range();
        void set_value_from_vector(const Eigen::VectorXd &x) override;
        void apply_increment(const Eigen::VectorXd &dx) override;
        Eigen::Vector3d pos_W() const;
        std::string name() const override;
        void print() const override;
        std::shared_ptr<Variable> clone() const override;

    private:
        int id_;
        Eigen::Vector3d origin_pos_W_;
        Eigen::Vector3d bearing_W_;
        Eigen::VectorXd inverse_range_value_;
        bool is_constant_ = false;
    };

    using InverseRangeVariablePtr = std::shared_ptr<InverseRangeVariable>;

} // namespace factorama
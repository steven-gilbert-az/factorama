#pragma once
#include <iostream>
#include "factorama/types.hpp"

namespace factorama
{
    /**
     * @brief Generic N-dimensional linear variable
     *
     * Flexible variable type that can represent arbitrary vector quantities.
     * Uses simple linear updates (no special manifold operations like poses).
     *
     * @code
     * Eigen::VectorXd initial_bias(3);
     * initial_bias << 0.1, -0.05, 0.02;
     * auto bias_var = std::make_shared<GenericVariable>(1, initial_bias);
     * @endcode
     */
    class GenericVariable : public Variable
    {
    public:
        /**
         * @brief Construct generic variable with initial value
         * @param id Unique variable identifier
         * @param initial_value Initial vector value (any dimension > 0)
         */
        GenericVariable(int id, const Eigen::VectorXd &initial_value)
            : id_(id), value_(initial_value)
        {
            if (initial_value.size() == 0)
            {
                throw std::runtime_error("GenericVariable: initial_value must have non-zero size");
            }
        }

        int size() const override { return value_.size(); }

        const Eigen::VectorXd &value() const override { return value_; }

        void set_value_from_vector(const Eigen::VectorXd &x) override 
        {
            if (x.size() != size())
            {
                throw std::runtime_error("set_value_from_vector(): size mismatch");
            }
            value_ = x;
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            if (dx.size() != size())
            {
                throw std::runtime_error("apply_increment(): size mismatch");
            }
            value_ += dx;
        }

        int id() const override { return id_; }

        VariableType::VariableTypeEnum type() const override
        {
            return VariableType::generic;
        }

        bool is_constant() const override { return is_constant_; }
        void set_is_constant(bool val) { is_constant_ = val; }

        std::string name() const override
        {
            return "Generic" + std::to_string(id());
        }

        void print() const override
        {
            std::cout << name() << std::endl;
            std::cout << "Value: " << value_.transpose() << std::endl;
        }
        
        std::shared_ptr<Variable> clone() const override 
        {
            return std::make_shared<GenericVariable>(*this);
        }

    private:
        int id_;
        Eigen::VectorXd value_;
        bool is_constant_ = false;
    };
}
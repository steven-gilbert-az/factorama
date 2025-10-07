#pragma once
#include <iostream>
#include "factorama/base_types.hpp"

namespace factorama
{
    /**
     * @brief 3D landmark variable in Euclidean space
     *
     * Represents 3D world landmarks as simple position vectors. Commonly used
     * for bundle adjustment problems with camera observations.
     *
     * @code
     * Eigen::Vector3d landmark_pos(0.0, 1.0, 6.0);
     * auto landmark = std::make_shared<LandmarkVariable>(2, landmark_pos);
     * @endcode
     */
    class LandmarkVariable : public Variable
    {
    public:
        /**
         * @brief Construct landmark variable with initial 3D position
         * @param id Unique variable identifier
         * @param pos_W_init Initial position in world frame
         */
        LandmarkVariable(int id, const Eigen::Vector3d &pos_W_init)
            : pos_W_(pos_W_init)
        {
            id_ = id;
        }

        int size() const override { return 3; }

        const Eigen::VectorXd &value() const override { return pos_W_; }

        void set_value_from_vector(const Eigen::VectorXd &x) override
        {
            pos_W_ = x;
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            if (dx.size() != size())
            {
                throw std::runtime_error("apply_increment(): size mismatch");
            }
            pos_W_ += dx;
        }

        VariableType::VariableTypeEnum type() const override
        {
            return VariableType::landmark;
        }

        void set_is_constant(bool val) { is_constant_ = val; }

        /**
         * @brief Get 3D position in world frame
         * @return Position vector
         */
        Eigen::Vector3d pos_W() const
        {
            return pos_W_;
        }

        /**
         * @brief Set 3D position in world frame
         * @param pos New position vector
         */
        void set_pos_W(Eigen::Vector3d pos)
        {
            pos_W_ = pos;
        }

        void print() const override
        {
            std::cout << name() << std::endl;
            std::cout << "Pos: \n" << pos_W() << std::endl;
        }

        std::shared_ptr<Variable> clone() const override
        {
            return std::make_shared<LandmarkVariable>(*this);
        }

    private:
        Eigen::VectorXd pos_W_; // 3D position in world frame
    };
}

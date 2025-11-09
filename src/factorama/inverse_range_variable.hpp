#pragma once

#include <Eigen/Core>
#include <memory>
#include <iostream>
#include "factorama/base_types.hpp"

namespace factorama
{

    /**
     * @brief Inverse range parameterization
     *
     * Represents 3D landmarks using inverse range (inverse depth) parameterization.
     *
     * The landmark position is computed as: pos_W = origin_pos_W + bearing_W / inverse_range
     *
     * @code
     * Eigen::Vector3d origin(0.0, 0.0, 0.0);
     * Eigen::Vector3d bearing(0.0, 0.0, 1.0);
     * double initial_range = 5.0;
     * auto inv_range_var = std::make_shared<InverseRangeVariable>(
     *     1, origin, bearing, initial_range);
     * @endcode
     */
    class InverseRangeVariable : public Variable
    {
    public:
        /**
         * @brief Construct inverse range variable
         * @param variable_id Unique variable identifier
         * @param origin_pos_W Bearing origin position in world frame
         * @param bearing_W Unit bearing direction in world frame
         * @param initial_range Initial range estimate (positive distance)
         */
        InverseRangeVariable(int variable_id,
                             const Eigen::Vector3d &origin_pos_W,
                             const Eigen::Vector3d &bearing_W,
                             double initial_range);

        // Default min/max values for inverse range - domain specific - user can change these if they want.
        double minimum_inverse_range_ = 1e-6; // Assume meters, max range = 1million meters (1000 km) - obviously too small for satelite stuff
        double maximum_inverse_range_ = 1e3;  // minimum range = 1mm

        int size() const override { return 1; }
        const Eigen::VectorXd &value() const override { return inverse_range_value_; }
        const Eigen::Vector3d &origin_pos_W() const { return origin_pos_W_; }
        const Eigen::Vector3d &bearing_W() const { return bearing_W_; }
        VariableType::VariableTypeEnum type() const override { return VariableType::inverse_range_landmark; }
        double inverse_range() const { return inverse_range_value_[0]; }
        void clip_inverse_range();
        void set_value_from_vector(const Eigen::VectorXd &x) override;
        void apply_increment(const Eigen::VectorXd &dx) override;
        void print() const override;
        Eigen::Vector3d pos_W() const;
        std::shared_ptr<Variable> clone() const override;

    private:
        Eigen::Vector3d origin_pos_W_;
        Eigen::Vector3d bearing_W_;
        Eigen::VectorXd inverse_range_value_;
    };

    using InverseRangeVariablePtr = std::shared_ptr<InverseRangeVariable>;

} // namespace factorama
#pragma once

#include "factorama/base_types.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace factorama
{

    /**
     * @brief Plane Variable - represents a 3D plane in Hessian normal form
     *
     * The plane is parameterized as: n^T * p + d = 0
     * where:
     *   - n is a unit normal vector (3 components)
     *   - d is the signed distance from origin
     *
     * Total degrees of freedom: 3 (geometrically)
     * Internal representation: 4 parameters (unit vector + distance) - slightly underdetermined
     *
     * The plane equation is: unit_vector^T * point + distance_from_origin = 0
     * Points with positive values are on the side the normal points to.
     */
    class PlaneVariable : public Variable
    {
    public:
        /**
         * @brief Construct Plane variable from normal vector and distance
         * @param id Unique variable identifier
         * @param normal_vector Normal vector to the plane (will be normalized)
         * @param distance Signed distance from the origin to the plane
         */
        PlaneVariable(int id, const Eigen::Vector3d &normal_vector, double distance)
            : unit_vector_(normal_vector.normalized()),
              distance_from_origin_(distance),
              value_(4)
        {
            id_ = id;
            assert(normal_vector.squaredNorm() > 0.0 && "normal vector must be nonzero");
            value_ << unit_vector_, distance_from_origin_;
        }

        int size() const override { return 4; }  // 4 parameters (slightly underdetermined)

        VariableType::VariableTypeEnum type() const override
        {
            return VariableType::plane;
        }

        const Eigen::VectorXd &value() const override
        {
            return value_;
        }

        void set_value_from_vector(const Eigen::VectorXd &x) override
        {
            if (x.size() != 4)
            {
                throw std::runtime_error("PlaneVariable::set_value_from_vector(): expected size 4");
            }
            unit_vector_ = x.head<3>().normalized();
            distance_from_origin_ = x(3);
            value_ << unit_vector_, distance_from_origin_;
        }

        void apply_increment(const Eigen::VectorXd &dx) override
        {
            if (dx.size() != size())
            {
                throw std::runtime_error("PlaneVariable::apply_increment(): size mismatch");
            }
            // Apply increment to unit vector and distance, then renormalize
            unit_vector_ += dx.head<3>();
            distance_from_origin_ += dx(3);
            assert(unit_vector_.squaredNorm() > 0.0 && "normal vector must be nonzero after increment");
            unit_vector_.normalize();
            value_ << unit_vector_, distance_from_origin_;
        }

        /**
         * @brief Get the unit normal vector
         * @return Unit normal vector to the plane
         */
        const Eigen::Vector3d& unit_vector() const
        {
            return unit_vector_;
        }

        /**
         * @brief Get signed distance from origin
         * @return Distance from origin to plane
         */
        double distance_from_origin() const
        {
            return distance_from_origin_;
        }

        /**
         * @brief Compute signed distance from a point to the plane
         * @param point_pos_W Point position in world frame
         * @return Signed distance (positive on normal side, negative on opposite side)
         */
        double distance_from_point(const Eigen::Vector3d &point_pos_W) const
        {
            // Hessian form: n^T * p + d = 0
            // Distance = n^T * p + d
            return unit_vector_.dot(point_pos_W) + distance_from_origin_;
        }

        void print() const override
        {
            std::cout << name() << std::endl;
            std::cout << "Normal: " << unit_vector_.transpose() << std::endl;
            std::cout << "Distance: " << distance_from_origin_ << std::endl;
        }

        std::shared_ptr<Variable> clone() const override
        {
            return std::make_shared<PlaneVariable>(*this);
        }

    private:
        Eigen::Vector3d unit_vector_;        // Unit normal vector to the plane
        double distance_from_origin_;         // Signed distance from origin
        Eigen::VectorXd value_;              // 4-element vector [nx, ny, nz, d]
    };

} // namespace factorama

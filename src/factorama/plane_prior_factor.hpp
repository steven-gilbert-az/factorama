#pragma once
#include <cassert>
#include "factorama/base_types.hpp"
#include "factorama/plane_variable.hpp"

namespace factorama
{

    /**
     * @brief Prior constraint on plane parameters
     *
     * Applies a prior constraint to both the normal vector and distance of a PlaneVariable.
     * Uses separate sigmas for orientation and distance since they have different units.
     *
     * @code
     * Eigen::Vector3d prior_normal(0.0, 0.0, 1.0);
     * double prior_distance = -5.0;
     * auto plane_prior = std::make_shared<PlanePriorFactor>(
     *     factor_id++, plane_var, prior_normal, prior_distance, 0.1, 0.5);
     * @endcode
     */
    class PlanePriorFactor final : public Factor
    {
    public:
        /**
         * @brief Construct plane prior factor
         * @param id Unique factor identifier
         * @param plane Plane variable
         * @param normal_prior Prior unit normal vector (will be normalized)
         * @param distance_prior Prior distance from origin
         * @param normal_sigma Standard deviation of normal vector prior (radians)
         * @param distance_sigma Standard deviation of distance prior (meters)
         */
        PlanePriorFactor(int id,
                        PlaneVariable* plane,
                        const Eigen::Vector3d &normal_prior,
                        double distance_prior,
                        double normal_sigma = 1.0,
                        double distance_sigma = 1.0);

        int residual_size() const override { return size_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::plane_prior; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;
        std::string name() const override;

    private:
        PlaneVariable* plane_;
        Eigen::Vector3d normal_prior_;
        double distance_prior_;
        double weight_normal_;
        double weight_distance_;
        int size_ = 4;
    };

} // namespace factorama

#pragma once
#include "factorama/base_types.hpp"
#include "factorama/plane_variable.hpp"
#include <cassert>

namespace factorama
{

    /**
     * @brief Factor constraining a point to lie on or near a plane
     *
     * Measures the signed distance from a 3D point to a plane. The residual is the
     * perpendicular distance, positive on the side the normal points to.
     * Works with any 3D variable (LandmarkVariable, GenericVariable, etc.).
     *
     * @code
     * auto plane_constraint = std::make_shared<PlaneFactor>(
     *     factor_id++, point_var, plane_var, sigma);
     * @endcode
     */
    class PlaneFactor final : public Factor
    {
    public:
        /**
         * @brief Construct plane factor
         * @param id Unique factor identifier
         * @param point_var Point variable (must be 3D)
         * @param plane_var Plane variable
         * @param sigma Standard deviation of distance measurement
         */
        PlaneFactor(int id,
                    Variable* point_var,
                    PlaneVariable* plane_var,
                    double sigma = 1.0);

        PlaneFactor(int id,
                    Variable* point_var,
                    PlaneVariable* plane_var,
                    double sigma,
                    bool do_distance_scaling,
                    double dist_scaling_r0,
                    Eigen::Vector3d dist_scaling_p0);

        int residual_size() const override { return size_; }
        double weight() const { return weight_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::plane_factor; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;
        std::string name() const override;

    private:
        Variable* point_var_;
        PlaneVariable* plane_var_;
        double weight_;
        bool do_distance_scaling_;
        double dist_scaling_r0_;
        Eigen::Vector3d dist_scaling_p0_;
        int size_ = 1;
    };

} // namespace factorama

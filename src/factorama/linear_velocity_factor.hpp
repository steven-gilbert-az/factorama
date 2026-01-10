#pragma once
#include "factorama/base_types.hpp"
#include <cassert>

namespace factorama
{

    /**
     * @brief linear velocity between variables
     * initialized with variable1, variable2, and a generic velocity variable
     * NOTE: The velocity variable could have a smaller dimensionality than var1 / var2
     * If so, it is assumed to only constrain the first N states.
     * E.g. var1 - pose, var2 - pose
     * velocity_var = 3d (position only) velocity
     * this works because the pose's first three states are position (linear)
     * 
     * @code
     * auto vel_factor = std::make_shared<VelocityFactor>(
     *     factor_id++, var_1, var_2, velocity_variable, dt, sigma);
     * @endcode
     */
    class LinearVelocityFactor final : public Factor
    {
    public:
        /**
         * @brief Construct linear velocity factor
         * @param id Unique factor identifier
         * @param var_1 First variable - state at the first timestamp
         * @param var_2 Second variable - state at the second timestamp
         * @param velocity_variable Variable representing the approximate velocity state
         * @param dt - Delta time between variable1 and variable2
         * @param sigma Standard deviation of the velocity constraint
         */
        LinearVelocityFactor(int id,
                             Variable* var_1,
                             Variable* var_2,
                             Variable* velocity_variable,
                             double dt,
                             double sigma = 1.0,
                             int initial_index = 0);

        int residual_size() const override { return size_; }
        double weight() const { return weight_; }
        FactorType::FactorTypeEnum type() const override { return FactorType::generic_between; }

        Eigen::VectorXd compute_residual() const override;
        void compute_residual(Eigen::Ref<Eigen::VectorXd> result) const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override;
        std::vector<Variable *> variables() override;

    private:
        Variable* var_1_;
        Variable* var_2_;
        Variable* velocity_variable_;
        double dt_;  // delta time from var1 to var2
        double weight_;
        int initial_index_;
        int size_;
    };

} // namespace factorama
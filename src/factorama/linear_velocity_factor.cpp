#include "factorama/linear_velocity_factor.hpp"

namespace factorama
{

    LinearVelocityFactor::LinearVelocityFactor(int id,
                                               Variable *var_1,
                                               Variable *var_2,
                                               Variable *velocity_variable,
                                               double dt,
                                               double sigma) : var_1_(var_1),
                                                               var_2_(var_2),
                                                               velocity_variable_(velocity_variable),
                                                               dt_(dt),
                                                               weight_(1.0 / sigma)
    {
        id_ = id;
        assert(var_1 != nullptr && "var_1 cannot be nullptr");
        assert(var_2 != nullptr && "var_2 cannot be nullptr");
        assert(velocity_variable != nullptr && "velocity_variable cannot be nullptr");
        assert(var_1->size() == var_2->size() &&
               "LinearVelocityFactor: variable size mismatch");

        assert(var_1_->size() >= velocity_variable->size() &&
               "LinearVelocityFactor: velocity size must be <= the variable sizes");

        assert(sigma > 0.0 && "LinearVelocityFactor: sigma must be greater than zero");
    }


    // The residual is essentially in units of var1 (e.g. for landmarks this would be meters)
    Eigen::VectorXd LinearVelocityFactor::compute_residual() const
    {
        Eigen::VectorXd diff = var_2_->value() - var_1_->value();
        Eigen::VectorXd res = diff - velocity_variable_->value() * dt_;
        return weight_ * res;
    }

    void LinearVelocityFactor::compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const
    {
        jacobians.clear();

        const int dim = velocity_variable_->size();

        if (var_1_->is_constant())
        {
            jacobians.emplace_back(); // empty Jacobian
        }
        else
        {
            jacobians.emplace_back(-weight_ * Eigen::MatrixXd::Identity(dim, dim));
        }

        if (var_2_->is_constant())
        {
            jacobians.emplace_back();
        }
        else
        {
            jacobians.emplace_back(weight_ * Eigen::MatrixXd::Identity(dim, dim));
        }

        if (velocity_variable_->is_constant())
        {
            jacobians.emplace_back();
        }
        else
        {
            jacobians.emplace_back(-weight_  * dt_* Eigen::MatrixXd::Identity(dim, dim));
        }
    }

    std::vector<Variable *> LinearVelocityFactor::variables()
    {
        return {var_1_, var_2_, velocity_variable_};
    }

}
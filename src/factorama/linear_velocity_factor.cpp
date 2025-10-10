#include "factorama/linear_velocity_factor.hpp"

namespace factorama
{

    LinearVelocityFactor::LinearVelocityFactor(int id,
                                               Variable *var_1,
                                               Variable *var_2,
                                               Variable *velocity_variable,
                                               double dt,
                                               double sigma,
                                               int initial_index) : var_1_(var_1),
                                                                    var_2_(var_2),
                                                                    velocity_variable_(velocity_variable),
                                                                    dt_(dt),
                                                                    weight_(1.0 / sigma),
                                                                    initial_index_(initial_index)
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
        int residual_dim = residual_size();
        Eigen::VectorXd diff = var_2_->value().segment(initial_index_, residual_dim) - var_1_->value().segment(initial_index_, residual_dim);
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
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(dim,var_1_->size());
            J.block(0, initial_index_, dim, dim).diagonal().setConstant(-weight_);
            jacobians.emplace_back(J);
        }

        if (var_2_->is_constant())
        {
            jacobians.emplace_back();
        }
        else
        {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(dim, var_2_->size());
            J.block(0, initial_index_, dim, dim).diagonal().setConstant(weight_);
            jacobians.emplace_back(J);
        }

        if (velocity_variable_->is_constant())
        {
            jacobians.emplace_back();
        }
        else
        {
            jacobians.emplace_back(-weight_ * dt_ * Eigen::MatrixXd::Identity(dim, dim));
        }
    }

    std::vector<Variable *> LinearVelocityFactor::variables()
    {
        return {var_1_, var_2_, velocity_variable_};
    }

}
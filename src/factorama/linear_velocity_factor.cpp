#include "factorama/linear_velocity_factor.hpp"

namespace factorama
{

    LinearVelocityFactor::LinearVelocityFactor(int id, Variable *var_1, Variable *var_2, Variable *velocity_variable,
                                               double dt, double sigma, int initial_index)
        : var_1_(var_1)
        , var_2_(var_2)
        , velocity_variable_(velocity_variable)
        , dt_(dt)
        , weight_(1.0 / sigma)
        , initial_index_(initial_index)
        , size_(velocity_variable->size())
    {
        id_ = id;
        assert(var_1 != nullptr && "var_1 cannot be nullptr");
        assert(var_2 != nullptr && "var_2 cannot be nullptr");
        assert(velocity_variable != nullptr && "velocity_variable cannot be nullptr");
        assert(var_1->size() == var_2->size() && "LinearVelocityFactor: variable size mismatch");

        assert(var_1_->size() >= velocity_variable->size() &&
               "LinearVelocityFactor: velocity size must be <= the variable sizes");

        assert(sigma > 0.0 && "LinearVelocityFactor: sigma must be greater than zero");
    }

    // The residual is essentially in units of var1 (e.g. for landmarks this would be meters)
    Eigen::VectorXd LinearVelocityFactor::compute_residual() const
    {
        int residual_dim = residual_size();
        Eigen::VectorXd diff = var_2_->value().segment(initial_index_, residual_dim) -
                               var_1_->value().segment(initial_index_, residual_dim);
        Eigen::VectorXd res = diff - velocity_variable_->value() * dt_;
        return weight_ * res;
    }

    void LinearVelocityFactor::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
    {
        int residual_dim = residual_size();
        Eigen::VectorXd diff = var_2_->value().segment(initial_index_, residual_dim) -
                               var_1_->value().segment(initial_index_, residual_dim);
        result = weight_ * (diff - velocity_variable_->value() * dt_);
    }

    void LinearVelocityFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 3 variables
        if (jacobians.size() == 0) {
            jacobians.resize(3);
        } else if (jacobians.size() != 3) {
            jacobians.clear();
            jacobians.resize(3);
        }

        const int dim = velocity_variable_->size();

        if (var_1_->is_constant()) {
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != var_1_->size()) {
                jacobians[0].resize(size_, var_1_->size());
            }
            jacobians[0].setZero();
            jacobians[0].block(0, initial_index_, dim, dim).diagonal().setConstant(-weight_);
        }

        if (var_2_->is_constant()) {
            jacobians[1] = Eigen::MatrixXd();
        } else {
            if (jacobians[1].rows() != size_ || jacobians[1].cols() != var_2_->size()) {
                jacobians[1].resize(size_, var_2_->size());
            }
            jacobians[1].setZero();
            jacobians[1].block(0, initial_index_, dim, dim).diagonal().setConstant(weight_);
        }

        if (velocity_variable_->is_constant()) {
            jacobians[2] = Eigen::MatrixXd();
        } else {
            if (jacobians[2].rows() != size_ || jacobians[2].cols() != size_) {
                jacobians[2].resize(size_, size_);
            }
            jacobians[2].setZero();
            jacobians[2].diagonal().array() = -weight_ * dt_;
        }
    }

    std::vector<Variable *> LinearVelocityFactor::variables()
    {
        return {var_1_, var_2_, velocity_variable_};
    }

} // namespace factorama
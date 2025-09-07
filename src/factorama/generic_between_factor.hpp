#pragma once
#include "factorama/types.hpp"
#include <cassert>

namespace factorama
{

    class GenericBetweenFactor : public Factor
    {
    public:
        GenericBetweenFactor(int id,
                             Variable* var_a,
                             Variable* var_b,
                             Variable* measured_diff,
                             double sigma = 1.0)
            : id_(id), var_a_(var_a), var_b_(var_b), measured_diff_(measured_diff), weight_(1.0 / sigma)
        {
            assert(var_a != nullptr && "var_a cannot be nullptr");
            assert(var_b != nullptr && "var_b cannot be nullptr");
            assert(measured_diff != nullptr && "measured_diff cannot be nullptr");
            assert(var_a_->size() == var_b_->size() &&
                   "BetweenFactor: variable size mismatch");

            assert(var_a_->size() == measured_diff_->size() &&
                   "BetweenFactor: measured_diff size mismatch");

            assert(sigma > 0.0 && "BetweenFactor: sigma must be greater than zero");
        }

        int id() const override
        {
            return id_;
        }

        int residual_size() const override
        {
            return measured_diff_->size();
        }

        Eigen::VectorXd compute_residual() const override
        {
            Eigen::VectorXd diff = var_b_->value() - var_a_->value();
            Eigen::VectorXd res = diff - measured_diff_->value();
            return weight_ * res;
        }

        void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const override
        {
            jacobians.clear();

            const int dim = measured_diff_->size();

            if (var_a_->is_constant())
            {
                jacobians.emplace_back(); // empty Jacobian
            }
            else
            {
                jacobians.emplace_back(-weight_ * Eigen::MatrixXd::Identity(dim, dim));
            }

            if (var_b_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                jacobians.emplace_back(weight_ * Eigen::MatrixXd::Identity(dim, dim));
            }

            if (measured_diff_->is_constant())
            {
                jacobians.emplace_back();
            }
            else
            {
                jacobians.emplace_back(-weight_ * Eigen::MatrixXd::Identity(dim, dim));
            }
        }

        std::vector<Variable *> variables() override
        {
            return {var_a_, var_b_, measured_diff_};
        }

        double weight() const override
        {
            return weight_;
        }

        std::string name() const override
        {
            return "GenericBetweenFactor(" + var_a_->name() + ", " + var_b_->name() + ", " + measured_diff_->name() + ")";
        }

        FactorType::FactorTypeEnum type() const override
        {
            return FactorType::generic_between;
        }

    private:
        int id_;
        Variable* var_a_;
        Variable* var_b_;
        Variable* measured_diff_;
        double weight_;
    };

} // namespace factorama
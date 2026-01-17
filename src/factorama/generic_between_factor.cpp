#include "generic_between_factor.hpp"
#include <cassert>

namespace factorama
{

    GenericBetweenFactor::GenericBetweenFactor(int id, Variable *var_a, Variable *var_b, Variable *measured_diff,
                                               double sigma)
        : var_a_(var_a)
        , var_b_(var_b)
        , measured_diff_(measured_diff)
        , weight_(1.0 / sigma)
    {
        id_ = id;
        assert(var_a != nullptr && "var_a cannot be nullptr");
        assert(var_b != nullptr && "var_b cannot be nullptr");
        assert(measured_diff != nullptr && "measured_diff cannot be nullptr");
        assert(var_a_->size() == var_b_->size() && "BetweenFactor: variable size mismatch");

        assert(var_a_->size() == measured_diff_->size() && "BetweenFactor: measured_diff size mismatch");

        assert(sigma > 0.0 && "BetweenFactor: sigma must be greater than zero");

        size_ = measured_diff_->size();
    }

    Eigen::VectorXd GenericBetweenFactor::compute_residual() const
    {
        Eigen::VectorXd diff = var_b_->value() - var_a_->value();
        Eigen::VectorXd res = diff - measured_diff_->value();
        return weight_ * res;
    }

    void GenericBetweenFactor::compute_residual(Eigen::Ref<Eigen::VectorXd> result) const
    {
        Eigen::VectorXd diff = var_b_->value() - var_a_->value();
        result = weight_ * (diff - measured_diff_->value());
    }

    void GenericBetweenFactor::compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const
    {
        // Ensure jacobians vector has correct size for 3 variables
        if (jacobians.size() == 0) {
            jacobians.resize(3);
        } else if (jacobians.size() != 3) {
            jacobians.clear();
            jacobians.resize(3);
        }

        if (var_a_->is_constant()) {
            jacobians[0] = Eigen::MatrixXd();
        } else {
            if (jacobians[0].rows() != size_ || jacobians[0].cols() != size_) {
                jacobians[0].resize(size_, size_);
            }
            jacobians[0].setZero();
            jacobians[0].diagonal().array() = -weight_;
        }

        if (var_b_->is_constant()) {
            jacobians[1] = Eigen::MatrixXd();
        } else {
            if (jacobians[1].rows() != size_ || jacobians[1].cols() != size_) {
                jacobians[1].resize(size_, size_);
            }
            jacobians[1].setZero();
            jacobians[1].diagonal().array() = weight_;
        }

        if (measured_diff_->is_constant()) {
            jacobians[2] = Eigen::MatrixXd();
        } else {
            if (jacobians[2].rows() != size_ || jacobians[2].cols() != size_) {
                jacobians[2].resize(size_, size_);
            }
            jacobians[2].setZero();
            jacobians[2].diagonal().array() = -weight_;
        }
    }

    std::vector<Variable *> GenericBetweenFactor::variables()
    {
        return {var_a_, var_b_, measured_diff_};
    }

    std::string GenericBetweenFactor::name() const
    {
        return "GenericBetweenFactor(" + var_a_->name() + ", " + var_b_->name() + ", " + measured_diff_->name() + ")";
    }

} // namespace factorama
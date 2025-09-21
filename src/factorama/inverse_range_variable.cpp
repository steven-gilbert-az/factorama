#include "inverse_range_variable.hpp"
#include <iostream>
#include <limits>
#include <memory>

namespace factorama
{

InverseRangeVariable::InverseRangeVariable(int variable_id,
                                         const Eigen::Vector3d &origin_pos_W,
                                         const Eigen::Vector3d &bearing_W,
                                         double initial_range)
    : id_(variable_id), origin_pos_W_(origin_pos_W), bearing_W_(bearing_W.normalized())
{
    // Store inverse range
    inverse_range_value_ = Eigen::VectorXd(1);
    inverse_range_value_[0] = 1.0 / initial_range;

    clip_inverse_range();
}

void InverseRangeVariable::clip_inverse_range()
{
    // TODO: if we ever get logger integration, add an info or warning if it does clip it.
    inverse_range_value_[0] = std::max(inverse_range_value_[0], minimum_inverse_range_);
    inverse_range_value_[0] = std::min(inverse_range_value_[0], maximum_inverse_range_);
}

void InverseRangeVariable::set_value_from_vector(const Eigen::VectorXd &x)
{
    inverse_range_value_ = x;
    clip_inverse_range();
}

void InverseRangeVariable::apply_increment(const Eigen::VectorXd &dx)
{
    if (dx.size() != size())
    {
        throw std::runtime_error("apply_increment(): size mismatch");
    }
    inverse_range_value_ += dx;
    clip_inverse_range();
}

Eigen::Vector3d InverseRangeVariable::pos_W() const
{
    const double inv_range = inverse_range_value_[0];
    if (!std::isfinite(inv_range) || std::abs(inv_range) < 1e-9)
    {
        std::cerr << "inverse range NAN or Zero" << std::endl;
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    return origin_pos_W_ + (1.0 / inv_range) * bearing_W_;
}

std::string InverseRangeVariable::name() const
{
    return "InverseRangeVariable" + std::to_string(id());
}

void InverseRangeVariable::print() const
{
    std::cout << name() << std::endl;
    std::cout << "Origin: " << origin_pos_W() << std::endl;
    std::cout << "Range: " << 1.0 / inverse_range() << std::endl;
    std::cout << "Pos: " << pos_W() << std::endl;
}

std::shared_ptr<Variable> InverseRangeVariable::clone() const
{
    return std::make_shared<InverseRangeVariable>(*this);
}

} // namespace factorama
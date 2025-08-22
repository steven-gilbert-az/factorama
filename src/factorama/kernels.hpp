#pragma once
#include <Eigen/Dense>
#include <vector>

namespace factorama
{
    /**
     * Base class for robust kernels that downweight outliers in factor graph optimization.
     */
    class RobustKernel
    {
    public:
        virtual ~RobustKernel() = default;
        
        /**
         * Apply robust weighting to residual vector.
         */
        virtual Eigen::VectorXd apply_to_residual(const Eigen::VectorXd& residual) = 0;
        
        /**
         * Apply robust weighting to jacobians with proper chain rule.
         */
        virtual void apply_to_jacobians(
            const Eigen::VectorXd& residual,
            std::vector<Eigen::MatrixXd>& jacobians) = 0;
    };

    /**
     * Huber robust kernel - linear for small residuals, quadratic for large ones.
     * Loss function: rho(r) = r^2/2 if ||r|| <= delta, delta*||r|| - delta^2/2 otherwise
     */
    class HuberKernel : public RobustKernel
    {
    private:
        double delta_;

    public:
        explicit HuberKernel(double delta) : delta_(delta) {}
        
        Eigen::VectorXd apply_to_residual(const Eigen::VectorXd& residual) override
        {
            double r_norm = residual.norm();
            if (r_norm <= delta_) {
                return residual;
            }
            return residual * std::sqrt(delta_ / r_norm);
        }
        
        void apply_to_jacobians(
            const Eigen::VectorXd& residual,
            std::vector<Eigen::MatrixXd>& jacobians) override
        {
            double r_norm = residual.norm();
            if (r_norm <= delta_ || r_norm < 1e-12) {
                return;
            }
            
            // Chain rule: w * J_r + r * (w_deriv) * (r^T * J_r) / ||r||
            double w = std::sqrt(delta_ / r_norm);
            double w_deriv = -0.5 * delta_ / (r_norm * r_norm * r_norm);
            
            for (auto& jac : jacobians) {
                jac = w * jac + w_deriv * residual * (residual.transpose() * jac);
            }
        }

        double delta() const { return delta_; }
    };

    /*
     * Example factor integration:
     * 
     * virtual Eigen::VectorXd compute_weighted_residual() const {
     *     auto residual = compute_residual();
     *     if (kernel_) {
     *         residual = kernel_->apply_to_residual(residual);
     *     }
     *     return residual * std::sqrt(weight());
     * }
     * 
     * virtual void compute_weighted_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const {
     *     auto residual = compute_residual();
     *     compute_jacobians(jacobians);
     *     if (kernel_) {
     *         kernel_->apply_to_jacobians(residual, jacobians);
     *     }
     *     double w = std::sqrt(weight());
     *     for (auto& jac : jacobians) {
     *         jac *= w;
     *     }
     * }
     */
}
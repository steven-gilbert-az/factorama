#pragma once
#include <unordered_map>
#include <vector>
#include <memory>
#include <type_traits>
#include <iostream>
#include <Eigen/Sparse>
#include "factorama/types.hpp"

namespace factorama
{
    struct VariablePlacement
    {
        std::shared_ptr<Variable> variable;
        int index; // Start column in Jacobian
        int dim;   // Variable dimension
    };

    struct FactorPlacement
    {
        std::shared_ptr<Factor> factor;
        int residual_index; // Start row in residual vector
        int residual_dim;

        std::vector<int> variable_column_indices;
        std::vector<int> variable_dims;
    };

    class FactorGraph
    {
    public:
        FactorGraph() = default;

        void add_variable(const std::shared_ptr<Variable> &variable);
        void add_factor(const std::shared_ptr<Factor> &factor);

        void finalize_structure();
        Eigen::VectorXd &compute_full_residual_vector();
        Eigen::MatrixXd &compute_full_jacobian_matrix();
        Eigen::SparseMatrix<double>& compute_sparse_jacobian_matrix();
        void compute_full_jacobian_and_residual();

        std::shared_ptr<Variable> get_variable(int id) const;
        std::vector<std::shared_ptr<Variable>> get_all_variables() const;
        const std::vector<std::shared_ptr<Factor>> &get_all_factors() const;

        Eigen::VectorXd get_variable_vector() const;
        void set_variable_values_from_vector(const Eigen::VectorXd &x);
        void apply_increment(const Eigen::VectorXd &dx);

        // Safe public access to cached results
        const Eigen::MatrixXd &jacobian() const
        {
            if (!jacobian_valid_)
            {
                throw std::runtime_error("Jacobian not valid — call compute_full_jacobian_and_residual() first.");
            }
            return cached_jacobian_;
        }

        const Eigen::SparseMatrix<double> &sparse_jacobian() const
        {
            if (!sparse_jacobian_valid_)
            {
                throw std::runtime_error("Sparse Jacobian not valid — call compute_sparse_jacobian_matrix() first.");
            }
            return sparse_jacobian_; 
        }

        const Eigen::VectorXd &residual() const
        {
            if (!residual_valid_)
            {
                throw std::runtime_error("Residual not valid — call compute_full_jacobian_and_residual() first.");
            }
            return cached_residual_;
        }

        bool jacobian_valid() const { return jacobian_valid_; }
        bool residual_valid() const { return residual_valid_; }

        // Type-safe variable lookup
        template <typename T>
        std::vector<std::shared_ptr<T>> get_variables_of_type() const
        {
            std::vector<std::shared_ptr<T>> result;
            for (const auto &var : variables_vector_)
            {
                if (auto casted = std::dynamic_pointer_cast<T>(var))
                {
                    result.push_back(casted);
                }
            }
            return result;
        }

        // Print helpers
        void print_structure() const;
        void print_variables() const;
        void print_jacobian_and_residual(bool detailed = false);

        void set_sparse_jacobians(bool value){
            do_sparse_jacobian_ = value;
        } 

        // Basic stats
        int num_variables() const { return static_cast<int>(variables_vector_.size()); }
        int num_values() const { return num_values_; }
        int num_residuals() const { return num_residuals_; }

        void set_verbose(bool verbose) { verbose_ = verbose; }

        VariablePlacement variable_placement(int id, bool& valid_out) const {
            auto it = variable_placement_.find(id);
            if (it == variable_placement_.end()) {
                valid_out = false;
                return VariablePlacement();  // or maybe throw if that’s your style
            }
            valid_out = true;
            return it->second;
        }

        FactorPlacement factor_placement(int id, bool& valid_out) const {
            auto it = factor_placement_.find(id);
            if (it == factor_placement_.end()) {
                valid_out = false;
                return FactorPlacement();
            }
            valid_out = true;
            return it->second;
        }

        const std::unordered_map<int, VariablePlacement> &variable_placement_map() const {
            return variable_placement_;
        }
        const std::unordered_map<int, FactorPlacement> &factor_placement_map() const {
            return factor_placement_;
        }

        // detailed_factor_test: 
        // Test all factors to ensure that they seem healthy.
        // will return pass/fail, and will print out detailed
        // diagnostic info
        bool detailed_factor_test(double jacobian_tol, bool verbose = false);

    private:

        // Main storage
        std::unordered_map<int, std::shared_ptr<Variable>> variables_map_;
        std::vector<std::shared_ptr<Variable>> variables_vector_;
        std::unordered_map<int, std::shared_ptr<Factor>> factors_map_;
        std::vector<std::shared_ptr<Factor>> factors_;

        // Structure bookkeeping
        bool structure_finalized_ = false;
        std::unordered_map<int, VariablePlacement> variable_placement_;
        std::unordered_map<int, FactorPlacement> factor_placement_;

        int num_residuals_ = 0;
        int num_values_ = 0;

        // Cached computation
        Eigen::MatrixXd cached_jacobian_;
        bool jacobian_valid_ = false;

        Eigen::VectorXd cached_residual_;
        bool residual_valid_ = false;

        bool do_sparse_jacobian_ = false;
        Eigen::SparseMatrix<double> sparse_jacobian_;
        bool sparse_jacobian_initialized_ = false;
        bool sparse_jacobian_valid_ = false;

        // Sparse jacobian indices and data.
        // Each element in the "outer" layer is a column
        std::vector<std::vector<int>> sparse_jacobian_row_indices_;
        std::vector<std::vector<double>> sparse_jacobian_data_;

        // Misc
        bool verbose_ = false;

        // Helper functions
        void copy_sparse_data_to_eigen_matrix();
    };

} // namespace factorama
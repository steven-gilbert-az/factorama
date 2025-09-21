#pragma once
#include <unordered_map>
#include <vector>
#include <memory>
#include <type_traits>
#include <Eigen/Sparse>
#include "factorama/types.hpp"

namespace factorama
{
    /**
     * @brief Internal structure tracking variable placement in the optimization problem
     */
    struct VariablePlacement
    {
        Variable* variable;
        int index; // Start column in Jacobian
        int dim;   // Variable dimension
    };

    /**
     * @brief Internal structure tracking factor placement in the optimization problem
     */
    struct FactorPlacement
    {
        Factor* factor;
        int residual_index; // Start row in residual vector
        int residual_dim;

        std::vector<int> variable_column_indices;
        std::vector<int> variable_dims;
    };

    /**
     * @brief Central container for factor graph optimization problems
     *
     * FactorGraph manages variables (poses, landmarks, etc.) and factors (constraints, measurements)
     * for non-linear least squares optimization. Call finalize_structure() after adding all
     * variables and factors, then use with SparseOptimizer for solving.
     *
     * @code
     * FactorGraph graph;
     * graph.add_variable(pose_var);
     * graph.add_factor(measurement_factor);
     * graph.finalize_structure();
     * @endcode
     */
    class FactorGraph
    {
    public:
        FactorGraph() = default;

        /**
         * @brief Add a variable to the factor graph
         * @param variable Shared pointer to any Variable subclass (PoseVariable, LandmarkVariable, etc.)
         */
        void add_variable(const std::shared_ptr<Variable> &variable);

        /**
         * @brief Add a factor (constraint/measurement) to the factor graph
         * @param factor Shared pointer to any Factor subclass (BearingObservationFactor, PriorFactor, etc.)
         */
        void add_factor(const std::shared_ptr<Factor> &factor);

        /**
         * @brief Finalize the graph structure for optimization
         * Must be called after adding all variables and factors, before optimization
         */
        void finalize_structure();

        /**
         * @brief Compute residual vector for all factors
         * @return Reference to computed residual vector
         */
        Eigen::VectorXd &compute_full_residual_vector();

        /**
         * @brief Compute dense Jacobian matrix
         * @return Reference to computed Jacobian matrix
         */
        Eigen::MatrixXd &compute_full_jacobian_matrix();

        /**
         * @brief Compute sparse Jacobian matrix (more efficient for large problems)
         * @return Reference to computed sparse Jacobian matrix
         */
        Eigen::SparseMatrix<double>& compute_sparse_jacobian_matrix();

        /**
         * @brief Compute both Jacobian and residual in one call
         */
        void compute_full_jacobian_and_residual();

        /**
         * @brief Get variable by ID
         * @param id Variable ID
         * @return Pointer to variable or nullptr if not found
         */
        Variable* get_variable(int id) const;

        /**
         * @brief Get all variables in the graph
         * @return Vector of shared pointers to all variables
         */
        const std::vector<std::shared_ptr<Variable>> &get_all_variables() const;

        /**
         * @brief Get all factors in the graph
         * @return Vector of shared pointers to all factors
         */
        const std::vector<std::shared_ptr<Factor>> &get_all_factors() const;

        /**
         * @brief Get current variable values as a single vector
         * @return Concatenated vector of all variable values
         */
        Eigen::VectorXd get_variable_vector() const;

        /**
         * @brief Set all variable values from a concatenated vector
         * @param x Vector containing new values for all variables
         */
        void set_variable_values_from_vector(const Eigen::VectorXd &x);

        /**
         * @brief Apply optimization increment to all variables
         * @param dx Increment vector (typically from optimizer)
         */
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

        // Main storage - vector (like an array) and map
        //    variables_vector_, factors_ - shared_ptr, will manage memory
        //    variables_map_, factors_map_  - will store bare ptr, doesn't need to manage memory
        std::unordered_map<int, Variable*> variables_map_;
        std::vector<std::shared_ptr<Variable>> variables_vector_;
        std::unordered_map<int, Factor*> factors_map_;
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
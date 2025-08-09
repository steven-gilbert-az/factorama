#include "factorama/factor_graph.hpp"
#include "factorama/numerical_jacobian.hpp"
#include <stdexcept>
#include <iostream>

namespace factorama
{
    void FactorGraph::add_variable(const std::shared_ptr<Variable> &variable)
    {
        if (variables_map_.find(variable->id()) != variables_map_.end())
        {
            throw std::runtime_error("Variable with ID " + std::to_string(variable->id()) + 
                                   " already exists in factor graph. Variable IDs must be unique.");
        }
        variables_map_[variable->id()] = variable;
        variables_vector_.push_back(variable);
    }

    void FactorGraph::add_factor(const std::shared_ptr<Factor> &factor)
    {
        if (factors_map_.find(factor->id()) != factors_map_.end())
        {
            throw std::runtime_error("Factor with ID " + std::to_string(factor->id()) + 
                                   " already exists in factor graph. Factor IDs must be unique.");
        }
        factors_map_[factor->id()] = factor;
        factors_.push_back(factor);
    }

    std::shared_ptr<Variable> FactorGraph::get_variable(int id) const
    {
        auto it = variables_map_.find(id);
        return (it != variables_map_.end()) ? it->second : nullptr;
    }

    std::vector<std::shared_ptr<Variable>> FactorGraph::get_all_variables() const
    {
        return variables_vector_;
    }

    const std::vector<std::shared_ptr<Factor>> &FactorGraph::get_all_factors() const
    {
        return factors_;
    }

    Eigen::VectorXd FactorGraph::get_variable_vector() const
    {
        Eigen::VectorXd x(num_values());
        int offset = 0;
        for (const auto &var : variables_vector_)
        {
            if (var->is_constant())
            {
                continue;
            }
            bool placement_valid = false;
            auto var_placement = variable_placement(var->id(), placement_valid);

            if (!placement_valid)
            {
                throw std::runtime_error("FactorGraph: Variable placement not found for variable ID " + std::to_string(var->id()));
            }

            if (var_placement.dim != var->size())
            {
                throw std::runtime_error("FactorGraph: Variable size mismatch for ID " + std::to_string(var->id()) + 
                                        ". Expected: " + std::to_string(var_placement.dim) + 
                                        ", Got: " + std::to_string(var->size()));
            }

            x.segment(offset, var->size()) = var->value();
            offset += var->size();
        }
        return x;
    }

    void FactorGraph::set_variable_values_from_vector(const Eigen::VectorXd &x)
    {
        int offset = 0;
        for (const auto &var : variables_vector_)
        {
            if (!var->is_constant())
            {
                var->set_value_from_vector(x.segment(offset, var->size()));
                offset += var->size();
            }
        }
    }

    void FactorGraph::apply_increment(const Eigen::VectorXd &dx)
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("apply_increment: must call finalize_structure() first");
        }
        
        if (dx.size() != num_values_)
        {
            throw std::runtime_error("apply_increment: increment vector size (" + 
                                    std::to_string(dx.size()) + 
                                    ") does not match expected size (" + 
                                    std::to_string(num_values_) + ")");
        }
        
        const auto &variables = get_all_variables();
        for (const auto &var : variables)
        {
            if (var->is_constant())
            {
                continue;
            }

            bool valid = false;
            VariablePlacement placement = variable_placement(var->id(), valid);
            if (!valid)
            {
                std::cerr << "[FactorGraph] Skipping variable id=" << var->id() << " (no placement found)\n";
                continue;
            }

            if (placement.index + placement.dim > dx.size()) {
                throw std::runtime_error("apply_increment: variable placement exceeds vector bounds for variable " + std::to_string(var->id()));
            }

            const Eigen::VectorXd local_dx = dx.segment(placement.index, placement.dim);
            var->apply_increment(local_dx);
        }
    }

    Eigen::VectorXd &FactorGraph::compute_full_residual_vector()
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("Called before finalize_structure().");
        }

        if (!residual_valid_)
        {
            cached_residual_ = Eigen::VectorXd::Zero(num_residuals_);
        }

        for (const auto &it : factor_placement_)
        {
            auto factor_placement = it.second;
            const auto &factor = factor_placement.factor;
            const int row_offset = factor_placement.residual_index;

            Eigen::VectorXd r = factor->compute_residual();
            cached_residual_.segment(row_offset, r.size()) = r;
        }

        residual_valid_ = true;

        return cached_residual_;
    }

    Eigen::MatrixXd &FactorGraph::compute_full_jacobian_matrix()
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("Called 'compute_full_jacobian_matrix' before finalize_structure().");
        }

        if (!jacobian_valid_)
        {
            cached_jacobian_ = Eigen::MatrixXd::Zero(num_residuals_, num_values_);
        }

        for (const auto &it : factor_placement_)
        {
            auto factor_placement = it.second;
            const auto &factor = factor_placement.factor;
            const int row_offset = factor_placement.residual_index;

            std::vector<Eigen::MatrixXd> jacobians;
            factor->compute_jacobians(jacobians);
            const auto &variables = factor->variables();

            for (size_t i = 0; i < variables.size(); ++i)
            {
                const auto &var = variables[i];


                const auto &Ji = jacobians[i];
                // int col_offset = variable_offset(var->id());
                
                if(Ji.rows() == 0 || Ji.cols() == 0) {
                    // empty jacobian. ensure that this variable is meant to be constant, then move on
                    if(!var->is_constant()) {
                        throw std::runtime_error("FactorGraph: Jacobian block empty for non-constant variable: " + var->name());
                    }
                    continue;
                }

                bool var_valid = false;
                auto var_placement = variable_placement(var->id(), var_valid);
                if (!var_valid)
                {
                    std::cerr << "var id: " << var->id() << "not found" << std::endl;
                    continue;
                }
                int col_offset = var_placement.index;

                // Bounds and dimension validation
                if (row_offset + Ji.rows() > cached_jacobian_.rows() || 
                    col_offset + Ji.cols() > cached_jacobian_.cols()) {
                    throw std::runtime_error("FactorGraph: Jacobian block bounds exceeded. factor " + factor->name() + ", var " + var->name());
                }
                
                if (Ji.rows() != factor_placement.residual_dim) {
                    throw std::runtime_error("FactorGraph: Jacobian row mismatch for. factor " + factor->name() + ", var " + var->name());
                }
                
                if (Ji.cols() != var_placement.dim) {
                    throw std::runtime_error("FactorGraph: Jacobian column mismatch. factor " + factor->name() + ", var " + var->name());
                }

                cached_jacobian_.block(row_offset, col_offset, Ji.rows(), Ji.cols()) = Ji;
            }
        }

        jacobian_valid_ = true;
        return cached_jacobian_;
    }

    void FactorGraph::compute_full_jacobian_and_residual()
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("Called 'compute_full_jacobian_and_residual' before finalize_structure().");
        }
        
        compute_full_jacobian_matrix();
        compute_full_residual_vector();
    }

    void FactorGraph::finalize_structure()
    {
        variable_placement_.clear();
        factor_placement_.clear();

        int col_offset = 0;

        // Assign column offsets for variables

        for (size_t i = 0; i < variables_vector_.size(); i++)
        {
            const auto &var = variables_vector_[i];
            VariablePlacement placement;
            placement.variable = var;
            placement.index = col_offset;
            if (var->is_constant())
            {
                placement.dim = 0;
            }
            else
            {
                placement.dim = var->size();
            }
            variable_placement_[var->id()] = placement;
            col_offset += placement.dim;
        }

        int row_offset = 0;

        // Assign row offsets and column mappings for each factor
        for (size_t i = 0; i < factors_.size(); i++)
        {
            const auto &factor = factors_[i];
            FactorPlacement placement;
            placement.factor = factor;
            placement.residual_index = row_offset;
            placement.residual_dim = factor->residual_size();

            for (const auto &var : factor->variables())
            {
                if (var->is_constant()) {
                    continue;
                }

                bool var_placement_valid;
                auto var_placement = variable_placement(var->id(), var_placement_valid);
                if (!var_placement_valid)
                {
                    throw std::runtime_error("Variable placement not found for variable: " + 
                                           var->name() + 
                                           ". Variable may not have been added to the factor graph.");
                }

                placement.variable_column_indices.push_back(var_placement.index);
                placement.variable_dims.push_back(var_placement.dim);
            }

            factor_placement_[i] = placement;
            row_offset += placement.residual_dim;
        }

        num_residuals_ = row_offset;
        num_values_ = col_offset;

        // ======= Sparse structure preparation =======
        if (do_sparse_jacobian_)
        {

            // First, figure out the number of elements in each column
            // std::vector<int> num_sparse_rows(num_values_, 0); // num sparse rows in each column
            Eigen::VectorXi num_sparse_rows = Eigen::VectorXi::Zero(num_values_);
            //for (const auto &fp : factor_placement_)
            for(size_t i = 0; i < factors_.size(); i++)
            {
                auto& fp = factor_placement_[i]; // TODO : switch to ID eventually
                // const int row_base = fp.residual_index;
                const int row_dim = fp.residual_dim;
                for (auto &var : fp.factor->variables())
                {
                    if (var->is_constant())
                    {
                        continue;
                    }
                    auto &vp = variable_placement_[var->id()];

                    // for each column - add the num rows
                    int col_base = vp.index;
                    int col_dim = vp.dim;

                    for (int i = col_base; i < col_base + col_dim; i++)
                    {
                        num_sparse_rows[i] += row_dim;
                    }
                }
            }

            sparse_jacobian_row_indices_ = std::vector<std::vector<int>>(num_values_);
            sparse_jacobian_data_ = std::vector<std::vector<double>>(num_values_);

            // preallocate the sparse data based on # of rows
            for (int i = 0; i < num_values_; i++)
            {
                sparse_jacobian_row_indices_[i] = std::vector<int>(num_sparse_rows[i]);
                sparse_jacobian_data_[i] = std::vector<double>(num_sparse_rows[i]);
            }

            std::vector<int> current_row_index(num_values_, 0);

            for(size_t i = 0; i < factors_.size(); i++)
            {
                auto& fp = factor_placement_[i]; // TODO : switch to ID eventually
                const int row_base = fp.residual_index;
                const int row_dim = fp.residual_dim;

                const auto &col_indices = fp.variable_column_indices;
                const auto &col_dims = fp.variable_dims;

                for (size_t j = 0; j < col_indices.size(); ++j)
                {
                    int col_base = col_indices[j];
                    int col_dim = col_dims[j];

                    for (int r = 0; r < row_dim; ++r)
                    {
                        for (int c = 0; c < col_dim; ++c)
                        {
                            int dense_row = row_base + r;
                            int dense_col = col_base + c;
                            int sparse_row = current_row_index[dense_col];
                            sparse_jacobian_row_indices_[dense_col][sparse_row] = dense_row;
                            current_row_index[dense_col]++;
                        }
                    }
                }
            }

            // Preallocate the sparse jacobian
            sparse_jacobian_ = Eigen::SparseMatrix<double>(num_residuals_, num_values_);
            sparse_jacobian_.reserve(num_sparse_rows);

            sparse_jacobian_initialized_ = true;
        }

        structure_finalized_ = true;
    }

    Eigen::SparseMatrix<double> &FactorGraph::compute_sparse_jacobian_matrix()
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("Called 'compute_sparse_jacobian_matrix' before finalize_structure().");
        }

        if (!do_sparse_jacobian_)
        {
            throw std::runtime_error("Called 'compute_sparse_jacobian_matrix' This only works if you are doing sparse jacobians!!");
        }

        if (!sparse_jacobian_initialized_)
        {
            throw std::runtime_error("Called 'compute_sparse_jacobian_matrix' without the sparse structure being initalized!!");
        }

        //sparse_jacobian_.setZero(); // Clean existing values but preserve structure
        std::vector<int> current_row_index(num_values_, 0);
        for (size_t factor_ind = 0; factor_ind < factors_.size(); factor_ind++)
        //for (const auto &it : factor_placement_)
        {
            auto factor_placement = factor_placement_[factor_ind];
            //auto factor_placement = it.second;
            const auto &factor = factor_placement.factor;
            //const int row_offset = factor_placement.residual_index;

            std::vector<Eigen::MatrixXd> jacobians;
            factor->compute_jacobians(jacobians);
            const auto &variables = factor->variables();

            
            for (size_t i = 0; i < variables.size(); i++)
            {
                const auto &var = variables[i];
                const auto &Ji = jacobians[i];

                bool var_valid = false;
                auto var_placement = variable_placement(var->id(), var_valid);
                if (!var_valid)
                {
                    std::cerr << "var id: " << var->id() << " not found" << std::endl;
                    continue;
                }
                int col_offset = var_placement.index;

                // Iterate through each element of the Ji block and add non-zeros
                for (int r = 0; r < Ji.rows(); ++r)
                {
                    for (int c = 0; c < Ji.cols(); ++c)
                    {
                        double value = Ji(r, c);
                        // int dense_row = row_offset + r;
                        int dense_col = col_offset + c;
                        int sparse_row = current_row_index[dense_col];
                        // sparse_jacobian_.coeffRef(row_offset + r, col_offset + c) = value;
                        // sparse_jacobian_.insert(row_offset + r, col_offset + c) = value;

                        sparse_jacobian_data_[dense_col][sparse_row] = value;
                        current_row_index[dense_col]++;
                    }
                }
            }
        }

        // // TODO: add to a verbose check
        // std::cout << "[DEBUG] Sparse Jacobian build sanity check:\n";
        // std::cout << "  Matrix rows: " << sparse_jacobian_.rows() << "\n";
        // std::cout << "  Matrix cols: " << sparse_jacobian_.cols() << "\n";
        // std::cout << "  Expected num_values_: " << num_values_ << "\n";
        // std::cout << "  Row indices vector size: " << sparse_jacobian_row_indices_.size() << "\n";
        // std::cout << "  Data vector size: " << sparse_jacobian_data_.size() << "\n";
        // Fill in the "eigen sparse"

        bool initialize_every_time = false;
        // By "initializing every time" we can check for incidental zeros and sparse them out
        // The sparsity pattern might theoretically be different
        // each time

        if (!sparse_jacobian_valid_ || initialize_every_time)
        {
            if(initialize_every_time) {
                //sparse_jacobian_.setZero();
            }
            
            for (int col = 0; col < num_values_; ++col)
            {
                //std::cout << "startvec: " << col << std::endl;
                //sparse_jacobian_.startVec(col);
                auto &this_col_indices = sparse_jacobian_row_indices_[col];

                for (size_t i = 0; i < this_col_indices.size(); ++i)
                {  
                    double value = sparse_jacobian_data_[col][i];
                    if(initialize_every_time  && fabs(value) < 1e-12) {
                        continue;
                    }
                    int row_index = this_col_indices[i];
                    
                    //sparse_jacobian_.insertBack(row_index, col) = value;
                    sparse_jacobian_.coeffRef(row_index, col) = value;
                }
            }
            //sparse_jacobian_.makeCompressed();s
            sparse_jacobian_.finalize();
        }
        else
        {
            // subsequent population of sparse_jacobian_
            // use the following template, except calculate the index from row/col
            // double* values = J.valuePtr(); // Fast pointer access
            // for (int i : indices) {
            //     values[i] = calculated_jacobian_value; // Or += / *= etc
            // }

            copy_sparse_data_to_eigen_matrix();
        }
        sparse_jacobian_valid_ = true;

        return sparse_jacobian_;
    }

    void FactorGraph::print_structure() const
    {
        if (!structure_finalized_)
        {
            throw std::runtime_error("Cannot print structure — finalize_structure() hasn't been called.");
        }

        std::cout << "Factor graph structure:\n";
        std::cout << "  Variables: " << variables_vector_.size() << "\n";
        std::cout << "  Factors: " << factors_.size() << "\n";
        std::cout << "  Residuals: " << num_residuals_ << "\n";
        std::cout << "  Values: " << num_values_ << "\n";
    }

    void FactorGraph::print_variables() const
    {
        std::cout << "Variables:\n";

        for (const auto &var : variables_vector_)
        {
            var->print();
        }
    }

    void FactorGraph::print_jacobian_and_residual(bool detailed)
    {
        compute_full_jacobian_and_residual();
        if (!jacobian_valid_)
        {
            throw std::runtime_error("Jacobian not valid — call compute_full_jacobian_and_residual() first.");
        }

        std::cout << "Jacobian: " << cached_jacobian_.rows()
                  << " x " << cached_jacobian_.cols() << "\n";

        for (int i = 0; i < cached_jacobian_.cols(); ++i)
        {
            if (cached_jacobian_.col(i).norm() < 1e-12)
            {
                std::cout << "  ⚠️  Near-zero column in Jacobian: param " << i << "\n";
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(cached_jacobian_);
        std::cout << "Jacobian rank: " << svd.rank() << "\n";
        std::cout << "Condition estimate: "
                  << svd.singularValues()(0) / svd.singularValues().tail(1)(0) << "\n";

        std::cout << "Residual norm: " << cached_residual_.norm() << "\n";

        if (detailed)
        {
            Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
            std::cout << "Full Jacobian:\n"
                      << cached_jacobian_.format(fmt) << "\n";
            std::cout << "Full Residual:\n"
                      << cached_residual_.format(fmt) << "\n";
        }
    }

    bool FactorGraph::detailed_factor_test(double jacobian_tol, bool verbose) {
        if (!structure_finalized_) {
            throw std::runtime_error("detailed_factor_test: must call finalize_structure() first");
        }
        
        if (verbose) {
            std::cout << "\n=== DETAILED FACTOR TEST ===" << std::endl;
            std::cout << "Testing " << factors_.size() << " factors with tolerance " << jacobian_tol << std::endl;
        }
        
        int num_passed = 0;
        int num_failed = 0;
        
        for (size_t factor_idx = 0; factor_idx < factors_.size(); ++factor_idx) {
            const auto& factor = factors_[factor_idx];
            
            // Get analytical jacobians
            std::vector<Eigen::MatrixXd> J_analytic;
            factor->compute_jacobians(J_analytic);
            
            // Get numerical jacobians
            std::vector<Eigen::MatrixXd> J_numeric;
            ComputeNumericalJacobians(*factor, J_numeric);
            
            // Compare sizes
            if (J_analytic.size() != J_numeric.size()) {
                if (verbose) {
                    std::cout << "\n--- Factor " << factor_idx << " (" << factor->name() << ", ID=" << factor->id() << ") ---" << std::endl;
                    std::cout << "FAIL: Jacobian count mismatch - analytical: " << J_analytic.size() 
                             << ", numerical: " << J_numeric.size() << std::endl;
                }
                num_failed++;
                continue;
            }
            
            bool factor_passed = true;
            const auto& variables = factor->variables();
            
            for (size_t var_idx = 0; var_idx < J_analytic.size(); ++var_idx) {
                const auto& var = variables[var_idx];
                const auto& J_a = J_analytic[var_idx];
                const auto& J_n = J_numeric[var_idx];
                
                // Handle constant variables (empty jacobians)
                if (var->is_constant()) {
                    if (J_a.size() == 0 && J_n.size() == 0) {
                        continue;
                    } else {
                        if (verbose) {
                            if (factor_passed) {
                                std::cout << "\n--- Factor " << factor_idx << " (" << factor->name() << ", ID=" << factor->id() << ") ---" << std::endl;
                            }
                            std::cout << "  FAIL: Variable " << var_idx << " (" << var->name() << ", ID=" << var->id() 
                                     << ") - constant but jacobians not empty" << std::endl;
                        }
                        factor_passed = false;
                        continue;
                    }
                }
                
                // Check dimensions
                if (J_a.rows() != J_n.rows() || J_a.cols() != J_n.cols()) {
                    if (verbose) {
                        if (factor_passed) {
                            std::cout << "\n--- Factor " << factor_idx << " (" << factor->name() << ", ID=" << factor->id() << ") ---" << std::endl;
                        }
                        std::cout << "  FAIL: Variable " << var_idx << " (" << var->name() << ", ID=" << var->id() 
                                 << ") - dimension mismatch" << std::endl;
                        std::cout << "     Analytical: " << J_a.rows() << "x" << J_a.cols() << std::endl;
                        std::cout << "     Numerical:  " << J_n.rows() << "x" << J_n.cols() << std::endl;
                    }
                    factor_passed = false;
                    continue;
                }
                
                // Compare values
                Eigen::MatrixXd diff = J_n - J_a;  // numerical - analytical
                double max_error = diff.cwiseAbs().maxCoeff();
                double sum_abs_error = diff.cwiseAbs().sum();
                
                if (max_error >= jacobian_tol) {
                    if (verbose) {
                        if (factor_passed) {
                            std::cout << "\n--- Factor " << factor_idx << " (" << factor->name() << ", ID=" << factor->id() << ") ---" << std::endl;
                        }
                        std::cout << "  FAIL: Variable " << var_idx << " (" << var->name() << ", ID=" << var->id() 
                                 << ") - jacobian comparison FAILED" << std::endl;
                        std::cout << "     Max error: " << max_error << " (tolerance: " << jacobian_tol << ")" << std::endl;
                        std::cout << "     Sum |error|: " << sum_abs_error << std::endl;
                        
                        // Print detailed jacobian info
                        Eigen::IOFormat fmt(6, 0, ", ", "\n", "[", "]");
                        std::cout << "     Numerical jacobian:\n" << J_n.format(fmt) << std::endl;
                        std::cout << "     Analytical jacobian:\n" << J_a.format(fmt) << std::endl;
                        std::cout << "     Diff (numerical - analytical):\n" << diff.format(fmt) << std::endl;
                    }
                    
                    factor_passed = false;
                }
            }
            
            if (factor_passed) {
                num_passed++;
            } else {
                num_failed++;
            }
        }
        
        // Print summary
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Factors passed: " << num_passed << "/" << factors_.size() << std::endl;
        std::cout << "Factors failed: " << num_failed << "/" << factors_.size() << std::endl;
        
        bool all_passed = (num_failed == 0);
        if (all_passed) {
            std::cout << "ALL FACTORS PASSED!" << std::endl;
        } else {
            std::cout << num_failed << " factors have jacobian issues" << std::endl;
        }
        
        return all_passed;
    }

    void FactorGraph::copy_sparse_data_to_eigen_matrix()
    {
        double *values = sparse_jacobian_.valuePtr();
        int running_index = 0;
        int total_nonzeros = sparse_jacobian_.nonZeros();

        // Validate preconditions
        if (!values) {
            throw std::runtime_error("FactorGraph: Sparse matrix valuePtr() returned null");
        }
        if (total_nonzeros < 0) {
            throw std::runtime_error("FactorGraph: Invalid nonzeros count");
        }

        for (int col = 0; col < num_values_; ++col)
        {
            const size_t num_elem = sparse_jacobian_row_indices_[col].size();
            const auto& source_data = sparse_jacobian_data_[col];
            
            // Comprehensive bounds checking
            if (source_data.size() != num_elem) {
                throw std::runtime_error("FactorGraph: Data/indices size mismatch for column " + std::to_string(col));
            }
            if (source_data.empty()) {
                continue; // Skip empty columns
            }
            
            // Check for size_t to int conversion safety
            if (num_elem > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("FactorGraph: Column has too many elements for int indexing");
            }
            
            const int num_elem_int = static_cast<int>(num_elem);
            
            // Bounds checking with safe arithmetic
            if (running_index < 0 || running_index >= total_nonzeros) {
                throw std::runtime_error("FactorGraph: Invalid running_index " + std::to_string(running_index));
            }
            if (running_index + num_elem_int > total_nonzeros) {
                throw std::runtime_error("FactorGraph: Sparse matrix bounds exceeded");
            }
            
            // Safe copy using iterators 
            std::copy(source_data.begin(), source_data.end(), &values[running_index]);
            running_index += num_elem_int;
        }
    }
}

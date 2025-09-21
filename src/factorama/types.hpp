#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace factorama
{
    namespace VariableType
    {
        enum VariableTypeEnum
        {
            none = 0,
            pose,
            landmark,
            inverse_range_landmark,
            extrinsic_rotation,
            generic,
            num_variable_types
        };

        // Bulletproof pattern: constexpr function ensures compile-time safety
        static constexpr std::array<const char *, num_variable_types> get_variable_names()
        {
            std::array<const char *, num_variable_types> names{};
            names[none] = "none";
            names[pose] = "pose";
            names[landmark] = "landmark";
            names[inverse_range_landmark] = "inverse_range_landmark";
            names[extrinsic_rotation] = "extrinsic_rotation";
            names[generic] = "generic";
            return names;
        }

        static constexpr auto variable_names = get_variable_names();

        // Compile-time safety: ensure we didn't miss any enum values
        static_assert(variable_names.size() == num_variable_types,
                      "VariableType: variable_names array size must match num_variable_types");

        // Additional safety: ensure no strings are null
        static_assert(variable_names[none] != nullptr &&
                          variable_names[pose] != nullptr &&
                          variable_names[landmark] != nullptr &&
                          variable_names[inverse_range_landmark] != nullptr &&
                          variable_names[extrinsic_rotation] != nullptr &&
                          variable_names[generic] != nullptr,
                      "VariableType: all variable names must be non-null");
    }

    namespace FactorType
    {
        enum FactorTypeEnum
        {
            none = 0,
            bearing_observation,
            inverse_range_bearing,
            generic_prior,
            generic_between,
            pose_position_prior,
            pose_orientation_prior,
            pose_position_between,
            pose_orientation_between,
            num_factor_types
        };

        // Bulletproof pattern: constexpr function ensures compile-time safety
        static constexpr std::array<const char *, num_factor_types> get_factor_names()
        {
            std::array<const char *, num_factor_types> names{};
            names[none] = "none";
            names[bearing_observation] = "bearing_observation";
            names[inverse_range_bearing] = "inverse_range_bearing";
            names[generic_prior] = "generic_prior";
            names[generic_between] = "generic_between";
            names[pose_position_prior] = "pose_position_prior";
            names[pose_orientation_prior] = "pose_orientation_prior";
            names[pose_position_between] = "pose_position_between";
            names[pose_orientation_between] = "pose_orientation_between";
            return names;
        }

        static constexpr auto factor_names = get_factor_names();

        // Compile-time safety: ensure we didn't miss any enum values
        static_assert(factor_names.size() == num_factor_types,
                      "FactorType: factor_names array size must match num_factor_types");

        // Additional safety: ensure no strings are null
        static_assert(factor_names[none] != nullptr &&
                          factor_names[bearing_observation] != nullptr &&
                          factor_names[inverse_range_bearing] != nullptr &&
                          factor_names[generic_prior] != nullptr &&
                          factor_names[generic_between] != nullptr &&
                          factor_names[pose_position_prior] != nullptr &&
                          factor_names[pose_orientation_prior] != nullptr &&
                          factor_names[pose_position_between] != nullptr &&
                          factor_names[pose_orientation_between] != nullptr,
                      "FactorType: all factor names must be non-null");
    }

    /// Base class for all optimization variables in the factor graph
    /// 
    /// Variables represent the state being optimized (poses, landmarks, etc.)
    /// Each variable has a vector representation for efficient optimization.
    class Variable
    {
    public:
        virtual ~Variable() = default;
        
        /// Get the unique identifier for this variable
        /// @return Variable ID used for indexing and identification
        virtual int id() const = 0;
        
        /// Get the dimension of this variable
        /// @return Dimension (e.g. 3 for rotations, 6 for poses)
        virtual int size() const = 0;
        
        /// Get the variable's current value as a vector
        /// @return Current variable value
        virtual const Eigen::VectorXd &value() const = 0;
        
        /// Set the variable from a new value vector
        /// @param x New value (used by optimizer)
        virtual void set_value_from_vector(const Eigen::VectorXd &x) = 0;
        
        /// Apply an increment to the current value
        /// @param dx Increment to apply (used by optimization algorithms)
        virtual void apply_increment(const Eigen::VectorXd &dx) = 0;

        /// Get the variable type enumeration
        /// @return Type identifier for this variable class
        virtual VariableType::VariableTypeEnum type() const = 0;
        
        /// Get a human-readable name for this variable
        /// @return String description of this variable
        virtual std::string name() const = 0;
        
        /// Print variable information to stdout
        virtual void print() const = 0;
        
        /// Check if this variable is held constant during optimization
        /// @return True if variable should not be optimized
        virtual bool is_constant() const = 0;

        /// Create a deep copy of this variable
        /// @return Shared pointer to cloned variable (used for numerical Jacobians)
        virtual std::shared_ptr<Variable> clone() const = 0;
    };

    /**
     * @brief Base class for all constraints and measurements in the factor graph
     *
     * Factors represent relationships between variables (measurements, priors, relative constraints).
     * Each factor contributes a residual vector and jacobians to the optimization problem.
     */
    class Factor
    {
    public:
        virtual ~Factor() = default;
        virtual int id() const = 0;

        /**
         * @brief Get the dimension of this factor's residual vector
         * @return Number of residual elements this factor contributes
         */
        virtual int residual_size() const = 0;

        /**
         * @brief Compute the residual vector for current variable values
         * @return Residual vector
         */
        virtual Eigen::VectorXd compute_residual() const = 0;

        /**
         * @brief Compute jacobians with respect to connected variables
         * @param jacobians Output vector of jacobian matrices (one per variable).
         *                  Empty (0x0) matrices indicate constant variables by convention.
         */
        virtual void compute_jacobians(std::vector<Eigen::MatrixXd> &jacobians) const = 0;

        /**
         * @brief Get pointers to all variables this factor depends on
         * @return Vector of variable pointers in jacobian order
         */
        virtual std::vector<Variable *> variables() = 0;

        virtual double weight() const = 0;
        virtual std::string name() const = 0;
        virtual FactorType::FactorTypeEnum type() const = 0;
    };
}
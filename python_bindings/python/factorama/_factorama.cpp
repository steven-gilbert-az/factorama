#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

// Factorama includes
#include <factorama/types.hpp>
#include <factorama/factor_graph.hpp>
#include <factorama/sparse_optimizer.hpp>
#include <factorama/pose_variable.hpp>
#include <factorama/landmark_variable.hpp>
#include <factorama/generic_variable.hpp>
#include <factorama/bearing_observation_factor.hpp>
#include <factorama/generic_prior_factor.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_factorama, m) {
    m.doc() = "Factorama Python bindings - factor graph optimization library";

    // Bind enums
    py::enum_<factorama::VariableType::VariableTypeEnum>(m, "VariableType")
        .value("none", factorama::VariableType::none)
        .value("pose", factorama::VariableType::pose)
        .value("landmark", factorama::VariableType::landmark)
        .value("inverse_range_landmark", factorama::VariableType::inverse_range_landmark)
        .value("extrinsic_rotation", factorama::VariableType::extrinsic_rotation)
        .value("generic", factorama::VariableType::generic);

    py::enum_<factorama::FactorType::FactorTypeEnum>(m, "FactorType")
        .value("none", factorama::FactorType::none)
        .value("bearing_observation", factorama::FactorType::bearing_observation)
        .value("inverse_range_bearing", factorama::FactorType::inverse_range_bearing)
        .value("generic_prior", factorama::FactorType::generic_prior)
        .value("generic_between", factorama::FactorType::generic_between)
        .value("pose_position_prior", factorama::FactorType::pose_position_prior)
        .value("pose_orientation_prior", factorama::FactorType::pose_orientation_prior)
        .value("pose_position_between", factorama::FactorType::pose_position_between)
        .value("pose_orientation_between", factorama::FactorType::pose_orientation_between);

    // Bind base classes
    py::class_<factorama::Variable, std::shared_ptr<factorama::Variable>>(m, "Variable")
        .def("id", &factorama::Variable::id)
        .def("size", &factorama::Variable::size)
        .def("value", &factorama::Variable::value, py::return_value_policy::reference)
        .def("set_value_from_vector", &factorama::Variable::set_value_from_vector)
        .def("apply_increment", &factorama::Variable::apply_increment)
        .def("type", &factorama::Variable::type)
        .def("name", &factorama::Variable::name)
        .def("is_constant", &factorama::Variable::is_constant)
        .def("clone", &factorama::Variable::clone);

    py::class_<factorama::Factor, std::shared_ptr<factorama::Factor>>(m, "Factor")
        .def("id", &factorama::Factor::id)
        .def("residual_size", &factorama::Factor::residual_size)
        .def("compute_residual", &factorama::Factor::compute_residual)
        .def("variables", &factorama::Factor::variables)
        .def("weight", &factorama::Factor::weight)
        .def("name", &factorama::Factor::name)
        .def("type", &factorama::Factor::type);

    // Bind concrete variable classes
    py::class_<factorama::PoseVariable, std::shared_ptr<factorama::PoseVariable>, factorama::Variable>(m, "PoseVariable")
        .def(py::init<int, const Eigen::Matrix<double, 6, 1>&, bool>(),
             "Create a PoseVariable with SE(3) pose",
             py::arg("id"), py::arg("pose_CW_init"), py::arg("do_so3_nudge") = true)
        .def(py::init<int, const Eigen::Vector3d&, const Eigen::Matrix3d&, bool>(),
             "Create a PoseVariable with position and rotation matrix",
             py::arg("id"), py::arg("pos_W"), py::arg("dcm_CW"), py::arg("do_so3_nudge") = true)
        .def("pos_W", &factorama::PoseVariable::pos_W)
        .def("rot_CW", &factorama::PoseVariable::rot_CW)
        .def("dcm_CW", &factorama::PoseVariable::dcm_CW)
        .def("set_is_constant", &factorama::PoseVariable::set_is_constant);

    py::class_<factorama::LandmarkVariable, std::shared_ptr<factorama::LandmarkVariable>, factorama::Variable>(m, "LandmarkVariable")
        .def(py::init<int, const Eigen::Vector3d&>(),
             "Create a LandmarkVariable with 3D position",
             py::arg("id"), py::arg("pos_W_init"))
        .def("pos_W", &factorama::LandmarkVariable::pos_W)
        .def("set_is_constant", &factorama::LandmarkVariable::set_is_constant);

    py::class_<factorama::GenericVariable, std::shared_ptr<factorama::GenericVariable>, factorama::Variable>(m, "GenericVariable")
        .def(py::init<int, const Eigen::VectorXd&>(),
             "Create a GenericVariable with arbitrary dimension",
             py::arg("id"), py::arg("initial_value"))
        .def("set_is_constant", &factorama::GenericVariable::set_is_constant);

    // Bind factor classes
    py::class_<factorama::BearingObservationFactor, std::shared_ptr<factorama::BearingObservationFactor>, factorama::Factor>(m, "BearingObservationFactor")
        .def(py::init<int, factorama::PoseVariable*, factorama::LandmarkVariable*, 
                      const Eigen::Vector3d&, double, bool>(),
             "Create a BearingObservationFactor",
             py::arg("id"), py::arg("pose_var"), py::arg("landmark_var"), 
             py::arg("bearing_C_observed"), py::arg("angle_sigma") = 1.0, py::arg("do_so3_nudge") = true);

    py::class_<factorama::GenericPriorFactor, 
               std::shared_ptr<factorama::GenericPriorFactor>, 
               factorama::Factor>(m, "GenericPriorFactor")
        .def(py::init<int, factorama::Variable*, const Eigen::VectorXd&, double>(),
             "Create a GenericPriorFactor",
             py::arg("id"), py::arg("variable"), py::arg("prior_value"), py::arg("sigma") = 1.0);

    // Optimizer enums and settings
    py::enum_<factorama::OptimizerMethod>(m, "OptimizerMethod")
        .value("GaussNewton", factorama::OptimizerMethod::GaussNewton)
        .value("LevenbergMarquardt", factorama::OptimizerMethod::LevenbergMarquardt);

    py::class_<factorama::OptimizerSettings>(m, "OptimizerSettings")
        .def(py::init<>())
        .def_readwrite("method", &factorama::OptimizerSettings::method)
        .def_readwrite("max_num_iterations", &factorama::OptimizerSettings::max_num_iterations)
        .def_readwrite("step_tolerance", &factorama::OptimizerSettings::step_tolerance)
        .def_readwrite("residual_tolerance", &factorama::OptimizerSettings::residual_tolerance)
        .def_readwrite("initial_lambda", &factorama::OptimizerSettings::initial_lambda)
        .def_readwrite("max_lambda", &factorama::OptimizerSettings::max_lambda)
        .def_readwrite("lambda_up_factor", &factorama::OptimizerSettings::lambda_up_factor)
        .def_readwrite("lambda_down_factor", &factorama::OptimizerSettings::lambda_down_factor)
        .def_readwrite("learning_rate", &factorama::OptimizerSettings::learning_rate)
        .def_readwrite("verbose", &factorama::OptimizerSettings::verbose)
        .def_readwrite("check_rank_deficiency", &factorama::OptimizerSettings::check_rank_deficiency);

    py::class_<factorama::OptimizerStats>(m, "OptimizerStats")
        .def(py::init<>())
        .def_readwrite("valid", &factorama::OptimizerStats::valid)
        .def_readwrite("chi2", &factorama::OptimizerStats::chi2)
        .def_readwrite("delta_norm", &factorama::OptimizerStats::delta_norm)
        .def_readwrite("residual_norm", &factorama::OptimizerStats::residual_norm)
        .def_readwrite("current_iteration", &factorama::OptimizerStats::current_iteration)
        .def_readwrite("rank", &factorama::OptimizerStats::rank)
        .def_readwrite("damping_parameter", &factorama::OptimizerStats::damping_parameter);

    // Bind main classes
    py::class_<factorama::FactorGraph, std::shared_ptr<factorama::FactorGraph>>(m, "FactorGraph")
        .def(py::init<>())
        .def("add_variable", &factorama::FactorGraph::add_variable)
        .def("add_factor", &factorama::FactorGraph::add_factor)
        .def("finalize_structure", &factorama::FactorGraph::finalize_structure)
        .def("compute_full_residual_vector", &factorama::FactorGraph::compute_full_residual_vector,
             py::return_value_policy::reference)
        .def("compute_full_jacobian_matrix", &factorama::FactorGraph::compute_full_jacobian_matrix,
             py::return_value_policy::reference)
        .def("compute_sparse_jacobian_matrix", &factorama::FactorGraph::compute_sparse_jacobian_matrix,
             py::return_value_policy::reference)
        .def("compute_full_jacobian_and_residual", &factorama::FactorGraph::compute_full_jacobian_and_residual)
        .def("get_variable", &factorama::FactorGraph::get_variable)
        .def("get_all_variables", &factorama::FactorGraph::get_all_variables)
        .def("get_all_factors", &factorama::FactorGraph::get_all_factors)
        .def("get_variable_vector", &factorama::FactorGraph::get_variable_vector)
        .def("set_variable_values_from_vector", &factorama::FactorGraph::set_variable_values_from_vector)
        .def("apply_increment", &factorama::FactorGraph::apply_increment)
        .def("jacobian", &factorama::FactorGraph::jacobian, py::return_value_policy::reference)
        .def("sparse_jacobian", &factorama::FactorGraph::sparse_jacobian, py::return_value_policy::reference)
        .def("residual", &factorama::FactorGraph::residual, py::return_value_policy::reference)
        .def("jacobian_valid", &factorama::FactorGraph::jacobian_valid)
        .def("residual_valid", &factorama::FactorGraph::residual_valid)
        .def("print_structure", &factorama::FactorGraph::print_structure)
        .def("print_variables", &factorama::FactorGraph::print_variables)
        .def("print_jacobian_and_residual", &factorama::FactorGraph::print_jacobian_and_residual,
             py::arg("detailed") = false)
        .def("num_variables", &factorama::FactorGraph::num_variables)
        .def("num_values", &factorama::FactorGraph::num_values)
        .def("num_residuals", &factorama::FactorGraph::num_residuals)
        .def("set_verbose", &factorama::FactorGraph::set_verbose)
        .def("detailed_factor_test", &factorama::FactorGraph::detailed_factor_test,
             py::arg("jacobian_tol"), py::arg("verbose") = false);

    py::class_<factorama::SparseOptimizer>(m, "SparseOptimizer")
        .def(py::init<>())
        .def("setup", &factorama::SparseOptimizer::setup)
        .def("optimize", &factorama::SparseOptimizer::optimize)
        .def("settings", &factorama::SparseOptimizer::settings, py::return_value_policy::reference)
        .def_readwrite("initial_stats", &factorama::SparseOptimizer::initial_stats_)
        .def_readwrite("current_stats", &factorama::SparseOptimizer::current_stats_);
}
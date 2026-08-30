#include <iostream>
#include <chrono>

// #include <factorama/base_types.hpp>
#include <factorama/factor_graph.hpp>
#include <factorama/generic_variable.hpp>
#include <factorama/generic_prior_factor.hpp>
#include <factorama/sparse_optimizer.hpp>
// #include "factorama/landmark_variable.hpp"
// #include "factorama/generic_prior_factor.hpp"
// #include "factorama/generic_variable.hpp"

using namespace factorama;

double JacobianBenchTest(int num_iter, int dim)
{
    // This test runs residual/jacobians N times and then returns the average time it took
    Eigen::VectorXd initial_position = Eigen::VectorXd::Zero(dim);
    int var_id = 0;
    int factor_id = 0;
    auto position_variable = std::make_shared<GenericVariable>(var_id++, initial_position);
    auto gps_factor_1 = std::make_shared<GenericPriorFactor>(
        factor_id++,                    // id
        position_variable.get(),        // variable
        initial_position, // estimate value
        0.5);                           // sigma
    std::vector<Eigen::MatrixXd> jacobian;

    Eigen::VectorXd residual;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; i++)
    {
        // Calculate the residual and jacobian
        gps_factor_1->compute_residual(residual);
        gps_factor_1->compute_jacobians(jacobian);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double total_time_s = double(duration.count()) / 1e9;

    return total_time_s / double(num_iter);
}


double VirtualJacobianTest(int num_iter, int dim)
{
    // This test runs residual/jacobians N times and then returns the average time it took
    Eigen::VectorXd initial_position = Eigen::VectorXd::Zero(dim);
    int var_id = 0;
    int factor_id = 0;
    auto position_variable = std::make_shared<GenericVariable>(var_id++, initial_position);
    auto gps_factor_1 = std::make_shared<GenericPriorFactor>(
        factor_id++,                    // id
        position_variable.get(),        // variable
        initial_position, // estimate value
        0.5);                           // sigma
    std::vector<Eigen::MatrixXd> jacobian;

    Eigen::VectorXd residual;

    Factor* gps_factor = gps_factor_1.get(); 

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; i++)
    {
        // Calculate the residual and jacobian
        gps_factor->compute_jacobians(jacobian);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double total_time_s = double(duration.count()) / 1e9;

    return total_time_s / double(num_iter);
}


int main()
{

    int num_var = 10;
    int num_factor = 10;

    int num_iter = 100000000;
    int num_dim = 9;

    double avg_time = JacobianBenchTest(num_iter, num_dim);

    auto avg_time_nanosecond = avg_time * 1e9;

    std::cout << "average residua/jacobian time (nanosecond) - " << avg_time_nanosecond << std::endl;


    double avg_time2 = VirtualJacobianTest(num_iter, num_dim);
    auto avg_time2_nanosecond = avg_time2 * 1e9;
    std::cout << "average residua/jacobian time  - virtual (nanosecond) - " << avg_time2_nanosecond << std::endl;

    //
}
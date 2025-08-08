#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <Eigen/Dense>
#include "factorama_test/test_utils.hpp"

#include <chrono>
#include <iostream>
#include <vector>

using Vector6d = Eigen::Matrix<double, 6, 1>;

// Assume CreateGTSAMGraph(...) is defined already

// std::pair<gtsam::NonlinearFactorGraph, gtsam::Values>
// CreateGTSAMGraph(const std::vector<Vector6d>& poses,
//                  const std::vector<Eigen::Vector3d>& landmarks,
//                  const std::vector<std::pair<int, int>>& sparsity);
std::tuple<gtsam::NonlinearFactorGraph, gtsam::Values>
CreateGTSAMGraph(
    const std::vector<Eigen::Matrix<double, 6, 1>> &poses,
    const std::vector<Eigen::Vector3d> &landmarks,
    const std::vector<std::pair<size_t, size_t>> &sparsity = {},
    bool verbose = false)
{
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    auto K = std::make_shared<gtsam::Cal3_S2>(500.0, 500.0, 0.0, 320.0, 240.0);
    auto measurement_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

    // Prior noise (fairly loose priors)
    auto pose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2).finished());
    auto landmark_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-3);

    const bool full_dense = sparsity.empty();

    if (verbose)
    {
        std::cout << "[CreateGTSAMGraph] Adding " << poses.size() << " poses and "
                  << landmarks.size() << " landmarks\n";
    }

    // === Add camera poses and pose priors ===
    for (size_t i = 0; i < poses.size(); ++i)
    {
        gtsam::Symbol cam_sym('x', i);
        gtsam::Pose3 cam_pose = gtsam::Pose3::Expmap(poses[i]);
        initial.insert(cam_sym, cam_pose);

        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            cam_sym, cam_pose, pose_prior_noise);

        if (verbose)
        {
            auto t = cam_pose.translation();
            auto r = cam_pose.rotation().rpy(); // roll, pitch, yaw
            std::cout << "  Pose x" << i << ": "
                      << "pos = [" << t.x() << ", " << t.y() << ", " << t.z() << "], "
                      << "rpy = [" << r.x() << ", " << r.y() << ", " << r.z() << "]\n";
        }
    }

    // === Add landmarks and landmark priors ===
    for (size_t j = 0; j < landmarks.size(); ++j)
    {
        gtsam::Symbol lm_sym('l', j);
        gtsam::Point3 point(landmarks[j]);
        initial.insert(lm_sym, point);

        graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
            lm_sym, point, landmark_prior_noise);

        if (verbose)
        {
            std::cout << "  Landmark l" << j << ": pos = ["
                      << point.x() << ", " << point.y() << ", " << point.z() << "]\n";
        }
    }

    // === Add projection factors ===
    for (size_t i = 0; i < poses.size(); ++i)
    {
        gtsam::Symbol cam_sym('x', i);
        gtsam::Pose3 cam_pose = initial.at<gtsam::Pose3>(cam_sym);
        gtsam::PinholeCamera<gtsam::Cal3_S2> cam(cam_pose, *K);

        for (size_t j = 0; j < landmarks.size(); ++j)
        {
            if (!full_dense &&
                std::find(sparsity.begin(), sparsity.end(), std::make_pair(i, j)) == sparsity.end())
                continue;

            gtsam::Symbol lm_sym('l', j);
            gtsam::Point3 pt = initial.at<gtsam::Point3>(lm_sym);

            try
            {
                gtsam::Point2 z = cam.project(pt);

                graph.emplace_shared<gtsam::GenericProjectionFactor<
                    gtsam::Pose3, gtsam::Point3>>(z, measurement_noise, cam_sym, lm_sym, K);

                if (verbose)
                {
                    std::cout << "    Factor: x" << i << " -> l" << j
                              << " | proj = [" << z.x() << ", " << z.y() << "]\n";
                }
            }
            catch (const gtsam::CheiralityException &e)
            {
                if (verbose)
                {
                    std::cout << "    ⚠️  CheiralityException for x" << i << " -> l" << j
                              << ": landmark behind camera\n";
                }
            }
        }
    }

    return {graph, initial};
}

void PrintJacobianSummary(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &values)
{
    gtsam::GaussianFactorGraph::shared_ptr linear_graph = graph.linearize(values);

    size_t total_rows = 0;
    size_t total_cols = 0;

    for (const gtsam::GaussianFactor::shared_ptr &factor : *linear_graph)
    {
        std::shared_ptr<gtsam::JacobianFactor> jf = std::dynamic_pointer_cast<gtsam::JacobianFactor>(factor);
        if (jf)
        {
            const Eigen::MatrixXd &A = jf->getA();
            total_rows += A.rows();
            total_cols += A.cols(); // Accumulating all column dimensions; note this may include redundancy
        }
        else
        {
            std::cout << "[JacobianSummary] Skipped non-Jacobian factor.\n";
        }
    }

    std::cout << "[JacobianSummary] Total residuals (rows): " << total_rows << "\n";
    std::cout << "[JacobianSummary] Total parameter dims (cols, possibly redundant): " << total_cols << "\n";
}

int main(int argc, char *argv[])
{

    // Default values
    int num_iterations = 10;
    int num_landmarks = 10;

    // Parse arguments
    if (argc > 1)
        num_iterations = std::atoi(argv[1]);
    if (argc > 2)
        num_landmarks = std::atoi(argv[2]);

    std::cout << " Using " << num_iterations << " iterations and "
              << num_landmarks << " landmarks \n";

    // Dummy test data - 5 initial cameras
    std::vector<Eigen::Matrix<double, 6, 1>> poses = {
        (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 1, 0, 0).finished(),
        (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 2, 0, 0).finished(),
        (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 1.3, 0, 0).finished(),
        (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 1.4, 0, 0).finished(),
        (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 1.5, 0, 0).finished()};

    // 9 initial landmarks
    std::vector<Eigen::Vector3d> landmarks = {
        {0.0, 0.0, 6.0},
        {1.0, 1.0, 7.0},
        {-1.0, -1.0, 8.0},
        {0.5, -0.5, 6.5},
        {-0.8, 0.9, 7.2},
        {1.1, -1.1, 7.8},
        {-1.3, 0.4, 6.9},
        {0.7, 1.2, 8.1},
        {-0.4, -0.7, 6.3}};

    auto extra_landmarks = factorama::CreateLandmarksInVolume(
        Eigen::Vector3d(-5.0, -5.0, 10.0),
        Eigen::Vector3d(5.0, 5.0, 15.0), num_landmarks);

    landmarks.insert(landmarks.end(),
                     extra_landmarks.begin(), extra_landmarks.end());

    std::vector<std::pair<size_t, size_t>> sparsity; // Empty = full dense

    // Construct graph and initial values
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    std::tie(graph, values) = CreateGTSAMGraph(poses, landmarks, sparsity, true);

    PrintJacobianSummary(graph, values);

    // Initial residual norm
    double initial_error = graph.error(values);
    std::cout << "[GTSAM] Initial residual norm (chi²): " << initial_error << "\n";

    gtsam::Values current_values = values;
    gtsam::GaussNewtonParams params;
    params.maxIterations = 1;

    std::vector<double> iteration_times;
    auto total_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_iterations; ++i)
    {
        // std::cout << "Iteration " << i << std::endl;
        auto iter_start = std::chrono::steady_clock::now();

        gtsam::GaussNewtonOptimizer optimizer(graph, current_values, params);
        current_values = optimizer.optimize();

        auto iter_end = std::chrono::steady_clock::now();
        double iteration_sec = double(std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end - iter_start).count()) * 1e-9;
        iteration_times.push_back(iteration_sec);
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_sec = double(std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count()) * 1e-9;

    // Final residual norm
    double final_error = graph.error(current_values);

    std::cout << "[GTSAM] Final residual norm (chi²): " << final_error << "\n";
    std::cout << "[GTSAM] Delta norm: " << (initial_error - final_error) << "\n";
    std::cout << "[GTSAM] Time for first step: " << iteration_times[0] * 1e6 << " us\n";
    std::cout << "[GTSAM] Total time for " << num_iterations << " iterations: " << total_sec * 1e6 << " us\n";

    return 0;
}